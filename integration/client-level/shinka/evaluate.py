#!/usr/bin/env python3
"""
Shinka evaluation harness for the client-level Heimdall heuristic search.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

FILE = Path(__file__).resolve()
SHINKA_WORKFLOW_ROOT = FILE.parent
CLIENT_LEVEL_ROOT = SHINKA_WORKFLOW_ROOT.parent
REPO_ROOT = CLIENT_LEVEL_ROOT.parent.parent
SHINKA_REPO_ROOT = REPO_ROOT / "ShinkaEvolve"
if str(SHINKA_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SHINKA_REPO_ROOT))

from shinka.core import run_shinka_eval

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
]
LABEL_COLUMN = "reject"
DEFAULT_DATASET_PATH = "data.csv"
NUM_RUNS = 3
NUM_SAMPLES_PER_RUN = 2000
BASE_SEED = 42
DEFAULT_TRAIN_EVAL_SPLIT = "50_50"
DEFAULT_SPLIT_SECTION = "full"

_cached_dataset: Optional[Tuple[List[Dict[str, float]], List[int]]] = None
_cached_dataset_path: Optional[str] = None


def _resolve_dataset_path(dataset_path: Optional[str]) -> str:
    if dataset_path is None:
        dataset_path = os.environ.get("SHINKA_DATASET_PATH")

    candidate = Path(dataset_path) if dataset_path else Path(DEFAULT_DATASET_PATH)
    if not candidate.is_absolute():
        candidate = SHINKA_WORKFLOW_ROOT / candidate
    return str(candidate.resolve())


def _load_dataset(dataset_path: Optional[str]) -> Tuple[List[Dict[str, float]], List[int]]:
    global _cached_dataset, _cached_dataset_path

    resolved = _resolve_dataset_path(dataset_path)
    if _cached_dataset is not None and _cached_dataset_path == resolved:
        return _cached_dataset

    features: List[Dict[str, float]] = []
    labels: List[int] = []
    with open(resolved, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            features.append({column: float(row[column]) for column in FEATURE_COLUMNS + ["latency"] if column in row})
            labels.append(int(row[LABEL_COLUMN]))

    _cached_dataset = (features, labels)
    _cached_dataset_path = resolved
    return features, labels


def parse_train_eval_split(train_eval_split: str) -> Tuple[int, int]:
    parts = train_eval_split.split("_")
    if len(parts) != 2:
        raise ValueError(f"Invalid split '{train_eval_split}'. Expected format like '80_20'.")
    train_pct = int(parts[0])
    eval_pct = int(parts[1])
    if train_pct < 0 or eval_pct < 0 or (train_pct + eval_pct) != 100:
        raise ValueError(
            f"Invalid split '{train_eval_split}'. Train and eval must be >= 0 and sum to 100."
        )
    return train_pct, eval_pct


def split_dataset(
    features: List[Dict[str, float]],
    labels: List[int],
    train_eval_split: str,
    split_section: str = "full",
    split_seed: int = BASE_SEED,
) -> Tuple[List[Dict[str, float]], List[int]]:
    section = split_section.lower()
    if section not in {"full", "train", "eval"}:
        raise ValueError(f"Invalid split_section '{split_section}'. Expected one of: full, train, eval.")

    train_pct, eval_pct = parse_train_eval_split(train_eval_split)
    if section == "full" or train_pct == 100 or eval_pct == 0:
        return features, labels

    x_train, x_eval, y_train, y_eval = train_test_split(
        features,
        labels,
        test_size=eval_pct / 100.0,
        random_state=split_seed,
        shuffle=True,
    )
    if section == "train":
        return list(x_train), list(y_train)
    return list(x_eval), list(y_eval)


def _compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    truth = np.array(y_true)
    pred = np.array(y_pred)

    tp = int(np.sum((pred == 1) & (truth == 1)))
    tn = int(np.sum((pred == 0) & (truth == 0)))
    fp = int(np.sum((pred == 1) & (truth == 0)))
    fn = int(np.sum((pred == 0) & (truth == 1)))
    total = len(truth)

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)

    false_admit_weight = 2.0
    false_reject_weight = 1.0
    weighted_precision = tp / (tp + false_reject_weight * fp) if (tp + fp) > 0 else 0.0
    weighted_recall = tp / (tp + false_admit_weight * fn) if (tp + fn) > 0 else 0.0
    weighted_f1 = (
        2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall)
        if (weighted_precision + weighted_recall) > 0
        else 0.0
    )

    false_admit_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    false_reject_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    # Explicitly penalise both FAR and FRR so "reject everything" (FAR=0, FRR=1)
    # cannot score high. FAR is weighted more (slow I/Os admitted are costlier).
    combined_score = (
        0.5 * weighted_f1
        + 0.3 * (1.0 - false_admit_rate)
        + 0.2 * (1.0 - false_reject_rate)
    )

    return {
        "combined_score": round(combined_score, 6),
        "weighted_f1": round(weighted_f1, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_admit_rate": round(false_admit_rate, 4),
        "false_reject_rate": round(false_reject_rate, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n_samples": total,
    }


def get_experiment_kwargs(
    run_idx: int,
    dataset_path: Optional[str],
    train_eval_split: str,
    split_section: str,
) -> Dict[str, Any]:
    features, labels = _load_dataset(dataset_path)
    features, labels = split_dataset(
        features=features,
        labels=labels,
        train_eval_split=train_eval_split,
        split_section=split_section,
        split_seed=BASE_SEED,
    )
    seed = BASE_SEED + run_idx
    if NUM_SAMPLES_PER_RUN is not None and NUM_SAMPLES_PER_RUN < len(features):
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(features)), NUM_SAMPLES_PER_RUN))
        sampled_features = [features[index] for index in indices]
        sampled_labels = [labels[index] for index in indices]
    else:
        sampled_features = features
        sampled_labels = labels

    return {
        "features": sampled_features,
        "labels": sampled_labels,
        "seed": seed,
    }


def validate_fn(result: Any) -> Tuple[bool, Optional[str]]:
    if not isinstance(result, dict):
        return False, "run_experiment must return a dict"
    predictions = result.get("predictions")
    if not isinstance(predictions, list) or len(predictions) == 0:
        return False, "predictions must be a non-empty list"
    if not all(value in (0, 1) for value in predictions):
        return False, "predictions must contain only 0/1 values"
    labels = result.get("labels")
    if not isinstance(labels, list):
        return False, "labels must be a list"
    if len(labels) != len(predictions):
        return False, "predictions and labels must have the same length"
    return True, None


def aggregate_metrics_fn(results: List[Any]) -> Dict[str, Any]:
    all_metrics = [_compute_metrics(result["labels"], result["predictions"]) for result in results]
    scalar_keys = [
        "combined_score",
        "weighted_f1",
        "f1",
        "accuracy",
        "precision",
        "recall",
        "false_admit_rate",
        "false_reject_rate",
    ]
    aggregated = {
        key: round(float(np.mean([metric[key] for metric in all_metrics])), 6)
        for key in scalar_keys
    }

    tp_total = int(sum(metric["tp"] for metric in all_metrics))
    tn_total = int(sum(metric["tn"] for metric in all_metrics))
    fp_total = int(sum(metric["fp"] for metric in all_metrics))
    fn_total = int(sum(metric["fn"] for metric in all_metrics))
    n_total = int(sum(metric["n_samples"] for metric in all_metrics))
    confusion_matrix = {
        "tp": tp_total,
        "tn": tn_total,
        "fp": fp_total,
        "fn": fn_total,
        "n_samples": n_total,
        "matrix": [[tn_total, fp_total], [fn_total, tp_total]],
    }

    text_feedback = (
        f"Heuristic evaluation over {len(results)} runs:\n"
        f"  combined_score   : {aggregated['combined_score']:.4f}  (target: maximise)\n"
        f"  Score formula    : 0.5*weighted_f1 + 0.3*(1-FAR) + 0.2*(1-FRR)\n"
        f"  weighted_f1      : {aggregated['weighted_f1']:.4f}\n"
        f"  accuracy         : {aggregated['accuracy']:.4f}\n"
        f"  false_admit_rate : {aggregated['false_admit_rate']:.4f}"
        f"  <- slow I/Os wrongly admitted (most costly; minimise)\n"
        f"  false_reject_rate: {aggregated['false_reject_rate']:.4f}"
        f"  <- fast I/Os wrongly rejected (also penalised; rejecting everything scores ~0.63 max)\n"
        f"  confusion_matrix : TN={tn_total}, FP={fp_total}, FN={fn_total}, TP={tp_total}\n"
    )

    return {
        "combined_score": aggregated["combined_score"],
        "public": {**aggregated, "confusion_matrix": confusion_matrix},
        "private": {"per_run_metrics": all_metrics, "confusion_matrix": confusion_matrix},
        "text_feedback": text_feedback,
    }


def main(program_path: str, results_dir: str, dataset_path: Optional[str]) -> None:
    train_eval_split = os.environ.get("SHINKA_TRAIN_EVAL_SPLIT", DEFAULT_TRAIN_EVAL_SPLIT)
    split_section = os.environ.get("SHINKA_SPLIT_SECTION", DEFAULT_SPLIT_SECTION)
    run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_experiment",
        num_runs=NUM_RUNS,
        get_experiment_kwargs=lambda run_idx: get_experiment_kwargs(
            run_idx,
            dataset_path,
            train_eval_split,
            split_section,
        ),
        aggregate_metrics_fn=aggregate_metrics_fn,
        validate_fn=validate_fn,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", type=str, default="initial.py")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--train_eval_split", type=str, default=None)
    parser.add_argument("--split_section", type=str, default=None, choices=["full", "train", "eval"])
    args = parser.parse_args()

    if args.train_eval_split:
        os.environ["SHINKA_TRAIN_EVAL_SPLIT"] = args.train_eval_split
    if args.split_section:
        os.environ["SHINKA_SPLIT_SECTION"] = args.split_section

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    main(args.program_path, args.results_dir, args.dataset_path)
