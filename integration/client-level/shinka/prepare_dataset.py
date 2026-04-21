#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

FILE = Path(__file__).resolve()
SHINKA_ROOT = FILE.parent
CLIENT_LEVEL_ROOT = SHINKA_ROOT.parent
FLASHNET_TRAINING_ROOT = CLIENT_LEVEL_ROOT / "experiment" / "flashnet" / "training"
TAIL_SCRIPT = FLASHNET_TRAINING_ROOT / "TailAlgorithms" / "tail_v1.py"
FEATURE_SCRIPT = FLASHNET_TRAINING_ROOT / "FeatureExtractors" / "feat_v6.py"

SHINKA_DATASET_COLUMNS = [
    "size",
    "queue_len",
    "prev_queue_len_1",
    "prev_queue_len_2",
    "prev_queue_len_3",
    "prev_latency_1",
    "prev_latency_2",
    "prev_latency_3",
    "prev_throughput_1",
    "prev_throughput_2",
    "prev_throughput_3",
    "latency",
    "reject",
]


def _device_names(devices: List[str]) -> List[str]:
    return [os.path.basename(device) for device in devices]


def _bundle_dir(trace_dir: Path, devices: List[str]) -> Path:
    return trace_dir / "...".join(_device_names(devices))


def _baseline_dir(trace_dir: Path, devices: List[str]) -> Path:
    return _bundle_dir(trace_dir, devices) / "baseline"


def _training_root(trace_dir: Path, devices: List[str]) -> Path:
    return _bundle_dir(trace_dir, devices) / "shinka" / "training_results"


def _run_checked(cmd: List[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _build_dataset(feature_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(feature_csv)
    if "io_type" in df.columns:
        df = df[df["io_type"] == 1].copy()

    missing = [column for column in SHINKA_DATASET_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {feature_csv}: {missing}")

    # Relabel using p70 read latency threshold — direct alignment with goal of
    # minimizing tail latency, replacing the GC-throughput-drop heuristic from tail_v1.py.
    p70 = df["latency"].quantile(0.70)
    df["reject"] = (df["latency"] > p70).astype(int)
    print(f"Relabeled with p70 latency threshold = {p70:.1f} µs  "
          f"(reject rate: {df['reject'].mean():.1%})")

    df = df[SHINKA_DATASET_COLUMNS].dropna().reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"===== output file : {output_csv}")


def prepare_trace_dir(trace_dir: Path, devices: List[str], skip_existing: bool = False) -> Dict[str, str]:
    baseline_dir = _baseline_dir(trace_dir, devices)
    if not baseline_dir.exists():
        raise FileNotFoundError(
            f"Baseline directory does not exist for {trace_dir}: {baseline_dir}. "
            "Run run_baseline.py first."
        )

    training_root = _training_root(trace_dir, devices)
    intermediate_dir = training_root / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    per_device_paths: List[Path] = []

    for device_idx in range(len(devices)):
        baseline_trace = baseline_dir / f"trace_{device_idx + 1}.trace"
        if not baseline_trace.exists():
            raise FileNotFoundError(f"Missing baseline trace: {baseline_trace}")

        dataset_output = training_root / f"dataset_device_{device_idx}.csv"
        per_device_paths.append(dataset_output)
        if skip_existing and dataset_output.exists():
            print(f"Dataset already exists, skipping: {dataset_output}")
            continue

        labeled_csv = intermediate_dir / f"device_{device_idx}_labeled.csv"
        feature_prefix = intermediate_dir / f"device_{device_idx}_features"
        feature_csv = feature_prefix.with_suffix(".csv")

        _run_checked(
            [
                sys.executable,
                str(TAIL_SCRIPT),
                "-file",
                str(baseline_trace),
                "-output",
                str(labeled_csv),
            ]
        )
        _run_checked(
            [
                sys.executable,
                str(FEATURE_SCRIPT),
                "-files",
                str(labeled_csv),
                "-output",
                str(feature_prefix),
                "-device",
                str(device_idx),
            ]
        )
        _build_dataset(feature_csv, dataset_output)

    combined_output = training_root / "dataset_combined.csv"
    if not (skip_existing and combined_output.exists()):
        dataframes = [pd.read_csv(path) for path in per_device_paths if path.exists()]
        if not dataframes:
            raise FileNotFoundError(f"No per-device datasets were created under {training_root}")
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df.to_csv(combined_output, index=False)
        print(f"===== output file : {combined_output}")

    manifest = {
        "trace_dir": str(trace_dir),
        "devices": devices,
        "training_root": str(training_root),
        "dataset_combined": str(combined_output),
        "dataset_device_0": str(per_device_paths[0]) if len(per_device_paths) > 0 else None,
        "dataset_device_1": str(per_device_paths[1]) if len(per_device_paths) > 1 else None,
    }
    manifest_path = training_root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"===== output file : {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Shinka-ready datasets from client-level baseline replay traces."
    )
    parser.add_argument(
        "-devices",
        nargs="+",
        required=True,
        help="Storage devices used in the client-level experiment.",
    )
    parser.add_argument("-trace_dir", type=str, help="Single trace directory.")
    parser.add_argument("-trace_dirs", nargs="+", type=str, help="Multiple trace directories.")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip dataset regeneration when the Shinka CSV already exists.",
    )
    args = parser.parse_args()

    if not args.trace_dir and not args.trace_dirs:
        raise SystemExit("Provide -trace_dir or -trace_dirs.")

    trace_dirs = [Path(path).resolve() for path in (args.trace_dirs or [args.trace_dir])]
    for trace_dir in trace_dirs:
        print(f"\nPreparing datasets for {trace_dir}")
        prepare_trace_dir(trace_dir=trace_dir, devices=args.devices, skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
