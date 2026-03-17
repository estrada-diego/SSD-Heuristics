#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run evolutionary search for N generations, then evaluate the best "
            "program with run_best_on_dataset.py."
        )
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        required=True,
        help="Number of generations for run_evo.py.",
    )
    parser.add_argument(
        "--split_tag",
        type=str,
        required=True,
        help="Split tag forwarded to run_best_on_dataset.py (e.g., 80_20, 100_0).",
    )
    parser.add_argument(
        "-dataset",
        type=str,
        required=True,
        help="Dataset path used for evolution and final evaluation.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="results_heimdall/final_eval",
        help="Output root for run_best_on_dataset.py.",
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="results_heimdall/best/main.py",
        help="Program path to evaluate after evolution (default: current best).",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    dataset_path = str(Path(args.dataset).resolve())

    env = os.environ.copy()
    env["SHINKA_DATASET_PATH"] = dataset_path
    env["SHINKA_NUM_GENERATIONS"] = str(args.num_generations)
    env["SHINKA_TRAIN_EVAL_SPLIT"] = args.split_tag
    env["SHINKA_SPLIT_SECTION"] = "train"

    evo_cmd = [sys.executable, str(here / "run_evo.py")]
    print("Running evolution:", " ".join(evo_cmd))
    subprocess.run(evo_cmd, cwd=str(here), env=env, check=True)

    eval_cmd = [
        sys.executable,
        str(here / "run_best_on_dataset.py"),
        "--program_path",
        args.program_path,
        "-dataset",
        dataset_path,
        "--split_tag",
        args.split_tag,
        "--split_section",
        "eval",
        "--output_root",
        args.output_root,
    ]
    print("Running final eval:", " ".join(eval_cmd))
    subprocess.run(eval_cmd, cwd=str(here), env=env, check=True)

    print("Done.")


if __name__ == "__main__":
    main()
