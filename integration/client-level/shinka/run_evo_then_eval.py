#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Shinka evolution, then evaluate the best heuristic on the held-out split."
    )
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--num_generations", type=int, required=True)
    parser.add_argument("--split_tag", type=str, required=True, help="Split tag like 80_20.")
    parser.add_argument(
        "--program_path",
        type=str,
        default=None,
        help="Program to evaluate after evolution. Defaults to <results_dir>/best/main.py.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Output root for the final evaluation files.",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    dataset_path = str(Path(args.dataset_path).resolve())
    results_dir = Path(args.results_dir).resolve()
    program_path = Path(args.program_path).resolve() if args.program_path else results_dir / "best" / "main.py"
    output_root = Path(args.output_root).resolve() if args.output_root else results_dir / "final_eval"

    env = os.environ.copy()
    env["SHINKA_NUM_GENERATIONS"] = str(args.num_generations)
    env["SHINKA_TRAIN_EVAL_SPLIT"] = args.split_tag
    env["SHINKA_SPLIT_SECTION"] = "train"

    evo_cmd = [
        sys.executable,
        str(here / "run_evo.py"),
        "--dataset_path",
        dataset_path,
        "--results_dir",
        str(results_dir),
        "--num_generations",
        str(args.num_generations),
        "--train_eval_split",
        args.split_tag,
        "--split_section",
        "train",
    ]
    print("Running evolution:", " ".join(evo_cmd))
    subprocess.run(evo_cmd, cwd=str(here), env=env, check=True)

    eval_cmd = [
        sys.executable,
        str(here / "run_best_on_dataset.py"),
        "--program_path",
        str(program_path),
        "-dataset",
        dataset_path,
        "--split_tag",
        args.split_tag,
        "--split_section",
        "eval",
        "--output_root",
        str(output_root),
    ]
    print("Running final eval:", " ".join(eval_cmd))
    subprocess.run(eval_cmd, cwd=str(here), env=env, check=True)


if __name__ == "__main__":
    main()
