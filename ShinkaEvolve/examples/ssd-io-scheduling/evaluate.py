#!/usr/bin/env python3
"""
evaluate.py — ShinkaEvolve evaluation script for Heimdall I/O admission heuristic.

ShinkaEvolve calls this as:
    python evaluate.py --program_path <evolved.py> --results_dir <dir>

It uses run_shinka_eval() from shinka.core, which:
  1. Calls get_experiment_kwargs(run_idx) to build kwargs for each run.
  2. Calls program.run_experiment(**kwargs) num_runs times.
  3. Passes all results to aggregate_metrics_fn() to produce the final metrics dict.
  4. Optionally calls validate_fn() on each result before aggregating.
"""

import argparse
import csv
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shinka.core import run_shinka_eval

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
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
]
LABEL_COLUMN = "reject"
DEFAULT_DATASET_PATH = "data.csv"   # default if nothing is provided
NUM_RUNS = 3
NUM_SAMPLES_PER_RUN = 2000
BASE_SEED = 42



# ---------------------------------------------------------------------------
# Dataset loading (cached at module level so it's only read once per worker)
# ---------------------------------------------------------------------------

_cached_dataset: Optional[Tuple[List[Dict[str, float]], List[int]]] = None
_cached_dataset_path: Optional[str] = None


def _resolve_dataset_path(dataset_path: Optional[str]) -> str:
    """
    Resolve dataset_path into an absolute path.

    Priority:
      1) CLI --dataset_path if provided
      2) env var SHINKA_DATASET_PATH if set
      3) default: <dir_of_evaluate.py>/data.csv

    Relative paths are resolved relative to evaluate.py's directory.
    """
    here = Path(__file__).resolve().parent

    if dataset_path is None:
        dataset_path = os.environ.get("SHINKA_DATASET_PATH")

    p = Path(dataset_path) if dataset_path else Path(DEFAULT_DATASET_PATH)
    if not p.is_absolute():
        p = here / p
    return str(p)


def _load_dataset(dataset_path: Optional[str]) -> Tuple[List[Dict[str, float]], List[int]]:
    global _cached_dataset, _cached_dataset_path

    resolved = _resolve_dataset_path(dataset_path)

    if _cached_dataset is not None and _cached_dataset_path == resolved:
        return _cached_dataset

    features: List[Dict[str, float]] = []
    labels: List[int] = []

    with open(resolved, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            features.append({col: float(row[col]) for col in FEATURE_COLUMNS})
            labels.append(int(row[LABEL_COLUMN]))

    _cached_dataset = (features, labels)
    _cached_dataset_path = resolved
    return features, labels



# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    total = len(y_true)

    accuracy  = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp)   if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn)   if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    # False admits (FN) are weighted 2× heavier than false rejects (FP).
    # Admitting a slow I/O causes direct tail-latency spikes at the SSD.
    FA_WEIGHT = 2.0
    FR_WEIGHT = 1.0
    w_prec = tp / (tp + FR_WEIGHT * fp) if (tp + fp) > 0 else 0.0
    w_rec  = tp / (tp + FA_WEIGHT * fn) if (tp + fn) > 0 else 0.0
    weighted_f1 = (2 * w_prec * w_rec / (w_prec + w_rec)
                   if (w_prec + w_rec) > 0 else 0.0)

    false_admit_rate  = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    false_reject_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    combined_score = 0.7 * weighted_f1 + 0.3 * (1.0 - false_admit_rate)

    return {
        "combined_score":    round(combined_score,    6),
        "weighted_f1":       round(weighted_f1,       4),
        "f1":                round(f1,                4),
        "accuracy":          round(accuracy,          4),
        "precision":         round(precision,         4),
        "recall":            round(recall,            4),
        "false_admit_rate":  round(false_admit_rate,  4),
        "false_reject_rate": round(false_reject_rate, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n_samples": total,
    }


# ---------------------------------------------------------------------------
# ShinkaEvolve hooks
# ---------------------------------------------------------------------------

def get_experiment_kwargs(run_idx: int, dataset_path: Optional[str]) -> Dict[str, Any]:
    """
    Called once per run by run_shinka_eval.
    Each run gets a different seed so the heuristic is tested on different subsets.
    """
    features, labels = _load_dataset(dataset_path)
    seed = BASE_SEED + run_idx

    if NUM_SAMPLES_PER_RUN is not None and NUM_SAMPLES_PER_RUN < len(features):
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(features)), NUM_SAMPLES_PER_RUN))
        sampled_features = [features[i] for i in indices]
        sampled_labels = [labels[i] for i in indices]
    else:
        sampled_features = features
        sampled_labels = labels

    # IMPORTANT: this MUST be a dict, not a set
    return {
        "features": sampled_features,
        "labels": sampled_labels,
        "seed": seed,
    }



def validate_fn(result: Any) -> Tuple[bool, Optional[str]]:
    """
    Sanity-check each run's output before aggregation.
    run_experiment() must return a dict with a valid "predictions" list.
    """
    if not isinstance(result, dict):
        return False, "run_experiment must return a dict"
    preds = result.get("predictions")
    if not isinstance(preds, list) or len(preds) == 0:
        return False, "predictions must be a non-empty list"
    if not all(p in (0, 1) for p in preds):
        return False, "predictions must contain only 0/1 values"
    labels = result.get("labels")
    if not isinstance(labels, list):
        return False, "labels must be a list"
    if len(labels) != len(preds):
        return False, "predictions and labels must have the same length"
    return True, None


def aggregate_metrics_fn(results: List[Any]) -> Dict[str, Any]:
    """
    Aggregate metrics across NUM_RUNS.
    Each result is the dict returned by run_experiment().
    The returned dict must contain "combined_score" (maximised by ShinkaEvolve).
    """
    all_metrics = []
    for res in results:
        m = _compute_metrics(res["labels"], res["predictions"])
        all_metrics.append(m)

    scalar_keys = [
        "combined_score", "weighted_f1", "f1", "accuracy",
        "precision", "recall", "false_admit_rate", "false_reject_rate",
    ]
    aggregated = {
        k: round(float(np.mean([m[k] for m in all_metrics])), 6)
        for k in scalar_keys
    }

    # Aggregate confusion matrix counts across runs for easier diagnostics.
    tp_total = int(sum(m["tp"] for m in all_metrics))
    tn_total = int(sum(m["tn"] for m in all_metrics))
    fp_total = int(sum(m["fp"] for m in all_metrics))
    fn_total = int(sum(m["fn"] for m in all_metrics))
    n_total = int(sum(m["n_samples"] for m in all_metrics))

    confusion_matrix = {
        "tp": tp_total,
        "tn": tn_total,
        "fp": fp_total,
        "fn": fn_total,
        "n_samples": n_total,
        # Rows: actual, columns: predicted
        # [[TN, FP],
        #  [FN, TP]]
        "matrix": [[tn_total, fp_total], [fn_total, tp_total]],
    }

    text_feedback = (
        f"Heuristic evaluation over {len(results)} runs:\n"
        f"  combined_score   : {aggregated['combined_score']:.4f}  (target: maximise)\n"
        f"  weighted_f1      : {aggregated['weighted_f1']:.4f}\n"
        f"  accuracy         : {aggregated['accuracy']:.4f}\n"
        f"  false_admit_rate : {aggregated['false_admit_rate']:.4f}"
        f"  ← slow I/Os wrongly admitted (most costly — minimise this)\n"
        f"  false_reject_rate: {aggregated['false_reject_rate']:.4f}"
        f"  ← fast I/Os wrongly rejected (minor cost)\n"
        f"  confusion_matrix  : TN={tn_total}, FP={fp_total}, FN={fn_total}, TP={tp_total}\n"
    )

    return {
        "combined_score": aggregated["combined_score"],  # required by ShinkaEvolve
        "public":         {**aggregated, "confusion_matrix": confusion_matrix},
        "private":        {"per_run_metrics": all_metrics, "confusion_matrix": confusion_matrix},
        "text_feedback":  text_feedback,                 # fed back to LLM mutators
    }


# --- Entry point ---
def main(program_path: str, results_dir: str, dataset_path: Optional[str]) -> None:
    run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_experiment",
        num_runs=NUM_RUNS,
        get_experiment_kwargs=lambda run_idx: get_experiment_kwargs(run_idx, dataset_path),
        aggregate_metrics_fn=aggregate_metrics_fn,
        validate_fn=validate_fn,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", type=str, default="initial.py")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to CSV dataset. If relative, resolved relative to evaluate.py. "
             "If omitted, uses $SHINKA_DATASET_PATH or ./data.csv next to evaluate.py.",
    )
    args = parser.parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    main(args.program_path, args.results_dir, args.dataset_path)
