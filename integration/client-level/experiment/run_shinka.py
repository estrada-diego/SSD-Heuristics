#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

ALGORITHM = "shinka"
FILE = Path(__file__).resolve()
EXPERIMENT_ROOT = FILE.parent
CLIENT_LEVEL_ROOT = EXPERIMENT_ROOT.parent
SHINKA_WORKFLOW_ROOT = CLIENT_LEVEL_ROOT / "shinka"
PREPARE_SCRIPT = SHINKA_WORKFLOW_ROOT / "prepare_dataset.py"
RUN_EVO_SCRIPT = SHINKA_WORKFLOW_ROOT / "run_evo.py"
EXPORT_SCRIPT = SHINKA_WORKFLOW_ROOT / "export_heuristic.py"


def get_output_dir(trace_dir: str, devices: List[str]) -> str:
    dev_names = [os.path.basename(dev_path) for dev_path in devices]
    return os.path.join(str(trace_dir), "...".join(dev_names), ALGORITHM)


def get_training_results_dir(trace_dir: str, devices: List[str]) -> Path:
    return Path(get_output_dir(trace_dir, devices)) / "training_results"


def get_exported_header_paths(trace_dir: str, devices: List[str]) -> List[Path]:
    export_root = get_training_results_dir(trace_dir, devices) / "exported_heuristics"
    return [export_root / f"shinka_dev_{index}.h" for index in range(len(devices))]


def get_best_program_paths(trace_dir: str, devices: List[str]) -> List[Path]:
    training_root = get_training_results_dir(trace_dir, devices)
    return [
        training_root / f"evolution_device_{index}" / "best" / "main.py"
        for index in range(len(devices))
    ]


def run_shell_command(command: str) -> None:
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Error running command: {exc}")


def run_checked(cmd: List[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def get_duration_from_trace(trace_path: str) -> str:
    with open(trace_path, encoding="utf-8") as handle:
        for line in handle:
            if "Duration" in line:
                value_raw = line.split("=")[2]
                if "." in value_raw:
                    return re.findall(r"-?\d+\.\d+", value_raw)[0]
                return re.findall(r"-?\d+", value_raw)[0]
    raise ValueError(f"Could not find duration in {trace_path}")


def start_processing(trace_dir: str, args: argparse.Namespace, specific_workplace: str) -> None:
    print("Processing " + str(trace_dir))
    output_dir = get_output_dir(trace_dir, args.devices)

    commands = []
    devices_list_str = "-".join(args.devices)
    print("The devices_list_str is {}".format(devices_list_str))

    for idx, _device in enumerate(args.devices):
        cmd = "echo 'Starting client ' " + str(idx) + "; "
        cmd += "cd " + specific_workplace + "/; "
        trace_name = "trace_" + str(idx + 1) + ".trace"
        stats_name = "trace_" + str(idx + 1) + ".stats"
        trace_path = os.path.join(trace_dir, trace_name)
        stats_path = os.path.join(trace_dir, stats_name)
        duration = get_duration_from_trace(stats_path)
        cmd += (
            "sudo ./replay.sh -user $USER -original_device_index "
            + str(idx)
            + " -devices_list "
            + devices_list_str
            + " -trace "
            + trace_path
            + " -output_dir "
            + output_dir
            + " -duration "
            + duration
            + "; exit"
        )
        commands.append(cmd)

    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        for command in commands:
            executor.submit(run_shell_command, command)
    print("Output dir = " + output_dir)
    subprocess.run("stty sane", shell=True, check=True)


def delete_dir(path: str) -> bool:
    try:
        shutil.rmtree(path)
        print(f"Directory '{path}' has been deleted.")
    except OSError as exc:
        print(f"Error: {exc}")
        return False
    return True


def prepare_datasets(trace_dir: str, devices: List[str], skip_existing: bool) -> None:
    cmd = [
        sys.executable,
        str(PREPARE_SCRIPT),
        "-devices",
        *devices,
        "-trace_dir",
        str(Path(trace_dir).resolve()),
    ]
    if skip_existing:
        cmd.append("--skip_existing")
    run_checked(cmd)


def train_device_heuristic(
    dataset_path: Path,
    results_dir: Path,
    args: argparse.Namespace,
) -> None:
    cmd = [
        sys.executable,
        str(RUN_EVO_SCRIPT),
        "--dataset_path",
        str(dataset_path),
        "--results_dir",
        str(results_dir),
        "--num_generations",
        str(args.num_generations),
        "--train_eval_split",
        args.train_eval_split,
        "--split_section",
        args.split_section,
        "--max_parallel_jobs",
        str(args.max_parallel_jobs),
    ]
    if args.disable_meta:
        cmd.append("--disable_meta")
    run_checked(cmd)


def export_device_heuristic(program_path: Path, output_header: Path, device_idx: int) -> None:
    cmd = [
        sys.executable,
        str(EXPORT_SCRIPT),
        "--program_path",
        str(program_path),
        "--output_header",
        str(output_header),
        "--function_name",
        f"shinka_predict_dev_{device_idx}",
    ]
    run_checked(cmd)


def ensure_exported_headers(trace_dir: str, devices: List[str], args: argparse.Namespace) -> bool:
    training_root = get_training_results_dir(trace_dir, devices)
    header_paths = get_exported_header_paths(trace_dir, devices)
    program_paths = get_best_program_paths(trace_dir, devices)
    export_root = training_root / "exported_heuristics"
    export_root.mkdir(parents=True, exist_ok=True)

    ok = True
    for index, (program_path, header_path) in enumerate(zip(program_paths, header_paths)):
        if not program_path.exists():
            print(f"Best Shinka program not found: {program_path}")
            ok = False
            continue
        if header_path.exists() and args.resume:
            print(f"Exported header already exists, skipping: {header_path}")
            continue
        export_device_heuristic(program_path, header_path, index)
    return ok


def train_shinka(trace_dir: str, devices: List[str], args: argparse.Namespace) -> bool:
    prepare_datasets(trace_dir, devices, skip_existing=args.resume)
    training_root = get_training_results_dir(trace_dir, devices)

    for device_idx in range(len(devices)):
        dataset_path = training_root / f"dataset_device_{device_idx}.csv"
        results_dir = training_root / f"evolution_device_{device_idx}"
        best_program = results_dir / "best" / "main.py"

        if args.resume and best_program.exists():
            print(f"Evolution already exists for device {device_idx}, skipping: {best_program}")
            continue

        if not dataset_path.exists():
            print(f"Dataset is missing for device {device_idx}: {dataset_path}")
            return False

        train_device_heuristic(dataset_path, results_dir, args)

    return ensure_exported_headers(trace_dir, devices, args)


def make_shinka(trace_dir: str, devices_list: List[str], specific_workplace: str) -> bool:
    header_paths = get_exported_header_paths(trace_dir, devices_list)
    for header_path in header_paths:
        if not header_path.exists():
            print("Exported Shinka header not found: {}".format(header_path))
            return False

    try:
        tmp_header_dir = Path(specific_workplace) / "2ssds_heuristics_header"
        tmp_header_dir.mkdir(parents=True, exist_ok=True)
        for header_path in header_paths:
            subprocess.run(["cp", str(header_path), str(tmp_header_dir)], check=True)
            print(f"File {header_path} copied to {tmp_header_dir}")
    except subprocess.CalledProcessError as exc:
        print(f"Error while copying generated headers: {exc}")
        return False

    original_directory = os.getcwd()
    try:
        os.chdir(specific_workplace)
        try:
            subprocess.run(["make"], check=True)
        except subprocess.CalledProcessError as make_error:
            print(f"Error running 'make': {make_error}")
            for file in Path("/tmp").glob("*.o"):
                file.unlink()
            subprocess.run(["make"], check=True)
        os.chdir(original_directory)
    except FileNotFoundError:
        print(f"Directory '{specific_workplace}' not found.")
        os.chdir(original_directory)
        return False

    return True


def do_sleep(timer_mins: int, time_start: "pd.Timestamp") -> None:
    import pandas as pd

    time_end = pd.Timestamp.now()
    time_elapsed = (time_end - time_start).seconds
    print("time_elapsed = " + str(time_elapsed))
    if time_elapsed < timer_mins * 60:
        print("Sleeping for " + str(timer_mins * 60 - time_elapsed) + " seconds")
        time.sleep(timer_mins * 60 - time_elapsed)


if __name__ == "__main__":
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("-devices", nargs="+", type=str, required=True)
    parser.add_argument("-trace_dir", type=str)
    parser.add_argument("-trace_dirs", nargs="+", type=str)
    parser.add_argument("-resume", action="store_true")
    parser.add_argument("-timer_mins", type=int, default=0)
    parser.add_argument("-only_training", action="store_true", default=False)
    parser.add_argument("-only_replaying", action="store_true", default=False)
    parser.add_argument("-if_model_updated", action="store_true", default=False)
    parser.add_argument("--num_generations", type=int, default=int(os.getenv("SHINKA_NUM_GENERATIONS", "40")))
    parser.add_argument("--train_eval_split", type=str, default=os.getenv("SHINKA_TRAIN_EVAL_SPLIT", "100_0"))
    parser.add_argument("--split_section", type=str, choices=["train", "eval", "full"], default=os.getenv("SHINKA_SPLIT_SECTION", "train"))
    parser.add_argument("--max_parallel_jobs", type=int, default=int(os.getenv("SHINKA_MAX_PARALLEL_JOBS", "4")))
    parser.add_argument("--disable_meta", action="store_true")
    args = parser.parse_args()

    if not args.trace_dir and not args.trace_dirs:
        print("ERROR: You must provide -trace_dir or -trace_dirs.")
        raise SystemExit(-1)
    if len(args.devices) != 2:
        print("ERROR: The Shinka client-level integration currently expects exactly 2 devices.")
        raise SystemExit(-1)

    trace_dirs = args.trace_dirs or [args.trace_dir]
    print("trace_paths = " + str(trace_dirs))
    print("algo = " + ALGORITHM)
    print("devices = " + str(args.devices))
    print("Found " + str(len(trace_dirs)) + " trace dirs")

    time_start = pd.Timestamp.now()
    for idx, trace_dir in enumerate(trace_dirs):
        print("\nProcessing trace dir " + str(idx + 1) + " out of " + str(len(trace_dirs)))
        output_dir = get_output_dir(trace_dir, args.devices)
        output_stat_path = os.path.join(output_dir, "trace_1.trace.stats")
        header_paths = get_exported_header_paths(trace_dir, args.devices)

        if not args.only_replaying:
            if args.resume and all(path.exists() for path in header_paths):
                print("     The exported Shinka heuristics already exist, skipping training")
            else:
                train_result = train_shinka(trace_dir, args.devices, args)
                if not train_result:
                    print("\n[Train Shinka Error], dir: {}".format(trace_dir))
                    raise SystemExit(-1)
        else:
            if not ensure_exported_headers(trace_dir, args.devices, args):
                print("\n[Export Shinka Error], dir: {}".format(trace_dir))
                raise SystemExit(-1)

        if args.timer_mins > 0:
            do_sleep(args.timer_mins, time_start)

        if args.only_training:
            continue

        if not all(path.exists() for path in header_paths):
            print("     WARNING: The exported headers are not ready, skipping")
            continue

        if os.path.isfile(output_stat_path) and args.if_model_updated:
            header_modified_time = max(os.path.getmtime(path) for path in header_paths)
            stat_modified_time = os.path.getmtime(output_stat_path)
            if header_modified_time < stat_modified_time:
                print("     The exported heuristic is older than the replayed traces, skipping")
                continue

        specific_workplace = str((EXPERIMENT_ROOT / "tmp_running" / "{}_{}...{}".format(
            ALGORITHM,
            args.devices[0].split("/")[2],
            args.devices[1].split("/")[2],
        )).resolve())
        if os.path.exists(specific_workplace):
            if delete_dir(specific_workplace) is False:
                print("Workplace delete error: {}".format(specific_workplace))
                raise SystemExit(-1)
        print("The specific workplace is {}".format(specific_workplace))
        try:
            subprocess.run(["cp", "-r", str((EXPERIMENT_ROOT / ALGORITHM).resolve()), specific_workplace], check=True)
        except subprocess.CalledProcessError:
            print("cp workplace wrong!")
            raise SystemExit(-1)

        make_result = make_shinka(trace_dir, args.devices, specific_workplace)
        if make_result is False:
            print("\n[Make ERROR: Exported heuristics are not complete], dir: {}".format(trace_dir))
            raise SystemExit(-1)

        subprocess.run("stty sane", shell=True, check=True)
        start_processing(trace_dir, args, specific_workplace)

        if delete_dir(specific_workplace) is False:
            print("Workplace delete error: {}".format(specific_workplace))
            raise SystemExit(-1)
