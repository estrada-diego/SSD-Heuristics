#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import evaluate as ev

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def create_output_dir(output_path: str) -> str:
    os.makedirs(output_path, exist_ok=True)
    return output_path


def write_stats(file_path: str, text: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    print("===== output file : " + file_path)


def _load_program(program_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("program", str(program_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load program module: {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plot_combined_eval_figure(
    figure_path: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    latencies: np.ndarray,
) -> Dict[str, int]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    raw_lat = np.asarray(latencies, dtype=float)
    accepted_lat = raw_lat[y_pred == 0]
    if raw_lat.size == 0:
        return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

    x_raw = np.sort(raw_lat)
    y_raw = np.arange(len(x_raw)) / float(len(x_raw))
    x_acc = np.sort(accepted_lat) if accepted_lat.size else np.array([0.0])
    y_acc = np.arange(len(x_acc)) / float(len(x_acc)) if accepted_lat.size else np.array([0.0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax_cm = axes[0]
    ax_cdf = axes[1]

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fast", "Slow"])
    disp.plot(ax=ax_cm, cmap=plt.cm.Blues, values_format="g", colorbar=False)
    ax_cm.set_title("Confusion Matrix (Fast/Slow)")

    ax_cdf.set_xlabel("Latency (us)")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_title("Read-IO Latency CDF (Best Heuristic)")
    p70 = np.percentile(x_raw, 70)
    ax_cdf.set_xlim(0, max(p70 * 3, 1000))
    ax_cdf.set_ylim(0, 1)
    ax_cdf.plot(x_raw, y_raw, label="Raw Latency", color="red")
    if accepted_lat.size:
        ax_cdf.plot(
            x_acc, y_acc, label="Heuristic Accepted", linestyle="dashdot", color="green"
        )
    ax_cdf.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    print("===== output figure : " + figure_path)

    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _evaluate_single_dataset(
    program_path: Path,
    dataset_path: str,
    output_root: str,
    split_tag: str,
    split_section: str,
    num_generations: str,
) -> None:
    resolved_dataset = ev._resolve_dataset_path(dataset_path)
    features, labels = ev._load_dataset(resolved_dataset)
    features, labels = ev.split_dataset(
        features=features,
        labels=labels,
        train_eval_split=split_tag,
        split_section=split_section,
        split_seed=ev.BASE_SEED,
    )
    latencies = np.array([float(f.get("latency", 0.0)) for f in features], dtype=float)

    program = _load_program(program_path)
    if not hasattr(program, "run_experiment"):
        raise AttributeError(f"{program_path} does not define run_experiment")

    result = program.run_experiment(features=features, labels=labels, seed=ev.BASE_SEED)
    is_valid, err = ev.validate_fn(result)
    if not is_valid:
        raise ValueError(f"Program output is invalid: {err}")

    y_true = np.asarray(result["labels"], dtype=int)
    y_pred = np.asarray(result["predictions"], dtype=int)
    metrics = ev._compute_metrics(result["labels"], result["predictions"])

    dataset_name = Path(resolved_dataset).stem
    model_name = "evolved_heuristic"
    out_dir = os.path.join(output_root, dataset_name, model_name, f"split_{split_tag}", "best")
    create_output_dir(out_dir)

    eval_fig = os.path.join(out_dir, "eval.png")
    cm_vals = _plot_combined_eval_figure(eval_fig, y_true, y_pred, latencies)

    try:
        roc = float(roc_auc_score(y_true, y_pred))
    except ValueError:
        roc = 0.0
    try:
        pr = float(average_precision_score(y_true, y_pred))
    except ValueError:
        pr = 0.0

    fpr = cm_vals["fp"] / max(cm_vals["fp"] + cm_vals["tn"], 1)
    fnr = cm_vals["fn"] / max(cm_vals["fn"] + cm_vals["tp"], 1)

    stats_lines: List[str] = []
    stats_lines.append(f"Dataset: {resolved_dataset}")
    stats_lines.append(f"Split: {split_tag} ({split_section})")
    if num_generations:
        stats_lines.append(f"Num generations: {num_generations}")
    stats_lines.append(f"%Profile rejection : {float(np.mean(y_true == 1))}")
    stats_lines.append(f"%Model rejection   : {float(np.mean(y_pred == 1))}")
    stats_lines.append(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Fast", "Slow"],
            zero_division=0,
        )
    )
    stats_lines.append(f"FPR = {round(fpr,3)}  ({round(fpr*100,1)}%)")
    stats_lines.append(f"FNR = {round(fnr,3)}  ({round(fnr*100,1)}%)")
    stats_lines.append(f"ROC-AUC = {round(roc,3)}  ({round(roc*100,1)}%)")
    stats_lines.append(f"PR-AUC = {round(pr,3)}  ({round(pr*100,1)}%)")
    stats_lines.append(f"combined_score = {metrics['combined_score']}")
    write_stats(os.path.join(out_dir, "eval.stats"), "\n".join(stats_lines))

    preds_csv = os.path.join(out_dir, "predictions.csv")
    with open(preds_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "label", "prediction", "latency"])
        for i, (yt, yp, lat) in enumerate(zip(y_true, y_pred, latencies)):
            writer.writerow([i, int(yt), int(yp), float(lat)])
    print("===== output file : " + preds_csv)

    fast_csv = os.path.join(out_dir, "fast_latency.csv")
    with open(fast_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["latency"])
        for lat in latencies[y_pred == 0]:
            writer.writerow([float(lat)])
    print("===== output file : " + fast_csv)

    metrics_json = os.path.join(out_dir, "final_metrics.json")
    payload: Dict[str, Any] = {
        "program_path": str(program_path),
        "dataset_path": resolved_dataset,
        "split_tag": split_tag,
        "split_section": split_section,
        "metrics": metrics,
        "confusion_matrix": {
            "tn": cm_vals["tn"],
            "fp": cm_vals["fp"],
            "fn": cm_vals["fn"],
            "tp": cm_vals["tp"],
            "matrix": [[cm_vals["tn"], cm_vals["fp"]], [cm_vals["fn"], cm_vals["tp"]]],
        },
        "roc_auc": roc,
        "pr_auc": pr,
    }
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("===== output file : " + metrics_json)

    print("===== output dir : " + out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate evolved best heuristic in a train_and_eval-like output format."
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="results_heimdall/best/main.py",
        help="Path to evolved program (default: results_heimdall/best/main.py).",
    )
    parser.add_argument("-dataset", type=str, help="Single dataset path.")
    parser.add_argument("-datasets", nargs="+", type=str, help="Multiple dataset paths.")
    parser.add_argument(
        "--output_root",
        type=str,
        default="results_heimdall/final_eval",
        help="Root directory for evaluation outputs.",
    )
    parser.add_argument(
        "--split_tag",
        type=str,
        default="100_0",
        help="Tag used in output path, e.g., 80_20 or 100_0.",
    )
    parser.add_argument(
        "--split_section",
        type=str,
        default="auto",
        choices=["auto", "full", "train", "eval"],
        help="Which data partition to evaluate. 'auto' uses eval for x_y with y>0, else full.",
    )
    args = parser.parse_args()
    num_generations = os.environ.get("SHINKA_NUM_GENERATIONS", "")

    if not args.dataset and not args.datasets:
        raise ValueError("Provide -dataset <path> or -datasets <path1> <path2> ...")

    arr_dataset: List[str] = []
    if args.datasets:
        arr_dataset += args.datasets
    elif args.dataset:
        arr_dataset.append(args.dataset)

    print("trace_profiles = " + str(arr_dataset))
    program_path = Path(args.program_path).resolve()
    train_pct, eval_pct = ev.parse_train_eval_split(args.split_tag)
    resolved_split_section = args.split_section
    if resolved_split_section == "auto":
        resolved_split_section = "eval" if eval_pct > 0 else "full"

    for dataset_path in arr_dataset:
        print("\nTraining on " + str(dataset_path))
        _evaluate_single_dataset(
            program_path=program_path,
            dataset_path=dataset_path,
            output_root=args.output_root,
            split_tag=args.split_tag,
            split_section=resolved_split_section,
            num_generations=num_generations,
        )


if __name__ == "__main__":
    main()
