#!/usr/bin/env python3
"""
Decision Tree baseline for Heimdall/FlashNet-style labeled datasets.

- Loads a CSV dataset that includes a binary label column: "reject" (0=Fast, 1=Slow)
- Splits into train/test (default 80/20)
- Trains a sklearn DecisionTreeClassifier
- Evaluates with confusion matrix + classification report + ROC-AUC/PR-AUC
- Optionally writes figures + stats into an output directory similar to your previous script

Usage:
  ./train_and_eval_dt.py -dataset /path/to/profile_v1.feat_v6.dataset -train_eval_split 80_20
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    classification_report,
)


def create_output_dir(output_path: str) -> str:
    os.makedirs(output_path, exist_ok=True)
    return output_path


def write_stats(file_path: str, text: str) -> None:
    with open(file_path, "w") as f:
        f.write(text)
    print("===== output file : " + file_path)


def print_confusion_matrix(figure_path: str, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[list[str], float, float]:
    target_names = ["Fast", "Slow"]
    labels_names = [0, 1]
    stats: list[str] = []
    stats.append(
        classification_report(
            y_true,
            y_pred,
            labels=labels_names,
            target_names=target_names,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels_names)
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    # Avoid div-by-zero with tiny epsilon
    eps = 1e-9
    FPR = round(FP / (FP + TN + eps), 3)
    FNR = round(FN / (TP + FN + eps), 3)

    # NOTE: AUC should be computed from probabilities/scores.
    # We'll set these to 0 here and compute proper AUC elsewhere using model scores.
    ROC_AUC = 0.0
    PR_AUC = 0.0

    stats.append(f"FPR = {FPR}  ({round(FPR*100,1)}%)")
    stats.append(f"FNR = {FNR}  ({round(FNR*100,1)}%)")

    fig, ax = plt.subplots(figsize=(4, 3))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format="g")
    ax.set_title(f"FPR = {round(FPR*100,1)}%  and FNR = {round(FNR*100,1)}%")
    plt.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    return stats, ROC_AUC, PR_AUC


def plot_latency_cdf(figure_path: str, df: pd.DataFrame, title: str) -> None:
    """
    df is assumed to already be read-only IOs if you filtered it.
    Uses y_pred == 0 (accepted/fast) to create "FlashNet-powered" curve.
    """
    accepted_lat = df.loc[df["y_pred"] == 0, "latency"].values
    raw_lat = df["latency"].values

    if len(raw_lat) == 0:
        return

    # CDF for accepted
    x1 = np.sort(accepted_lat) if len(accepted_lat) else np.array([0.0])
    y1 = np.arange(len(x1)) / float(len(x1)) if len(x1) else np.array([0.0])

    # CDF for raw
    x2 = np.sort(raw_lat)
    y2 = np.arange(len(x2)) / float(len(x2))

    percent_slow = int((len(raw_lat) - len(accepted_lat)) / len(raw_lat) * 100)

    plt.figure(figsize=(6, 3))
    plt.xlabel("Latency (us)")
    plt.ylabel("CDF")
    plt.title(f"{title}; Slow = {percent_slow}%")

    p70_lat = np.percentile(x2, 70)
    plt.xlim(0, max(p70_lat * 3, 1000))
    plt.ylim(0, 1)

    plt.plot(x2, y2, label="Raw Latency", color="red")
    if len(accepted_lat):
        plt.plot(x1, y1, label="Tree-powered", linestyle="dashdot", color="green")

    plt.legend(loc="lower right")
    plt.savefig(figure_path, bbox_inches="tight")
    plt.close()


def train_model(
    dataset_path: str,
    train_eval_split: str,
    max_depth: int | None,
    min_samples_leaf: int,
    random_state: int,
) -> None:
    ratios = train_eval_split.split("_")
    percent_train = int(ratios[0])
    percent_eval = int(ratios[1])
    assert percent_train + percent_eval == 100

    dataset = pd.read_csv(dataset_path)

    # Put "latency" at the end (like your original script)
    if "latency" in dataset.columns:
        reordered_cols = [c for c in dataset.columns if c != "latency"] + ["latency"]
        dataset = dataset[reordered_cols]

    # Basic checks
    if "reject" not in dataset.columns:
        raise ValueError('Dataset must contain a "reject" column (0=Fast, 1=Slow).')

    # Split
    X = dataset.copy(deep=True).drop(columns=["reject"], axis=1)
    y = dataset["reject"].copy(deep=True).astype(int)

    # Profile rejection rate
    vc = y.value_counts().to_dict()
    p_rejection = (vc.get(1, 0) / len(y)) if len(y) else 0.0

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=percent_eval / 100.0,
        random_state=random_state,
        stratify=y if len(vc) > 1 else None,
    )

    # Keep latency for plotting, but do not use it as feature
    X_train_latency = X_train["latency"] if "latency" in X_train.columns else None
    X_test_latency = X_test["latency"] if "latency" in X_test.columns else None

    if "latency" in X_train.columns:
        X_train = X_train.drop(columns=["latency"], axis=1)
        X_test = X_test.drop(columns=["latency"], axis=1)

    # Normalization (not required for trees, but keeps the interface similar)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_norm = scaler.transform(X_train)
    X_test_norm = scaler.transform(X_test)

    # Train decision tree
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(X_train_norm, y_train)

    # Predict labels
    y_pred = model.predict(X_test_norm).astype(int)

    # Scores for AUC (use predict_proba if available)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test_norm)[:, 1]
    else:
        probs = y_pred.astype(float)

    # Output directory (similar structure)
    dataset_name = str(Path(os.path.basename(dataset_path)).with_suffix(""))
    model_name = "decision_tree"
    parent_dir = Path(dataset_path).parent
    out_dir = os.path.join(
        str(parent_dir),
        dataset_name,
        model_name,
        "split_" + train_eval_split,
        f"depth_{max_depth if max_depth is not None else 'None'}_leaf_{min_samples_leaf}",
    )
    create_output_dir(out_dir)

    # Metrics
    stats: list[str] = []
    stats.append(f"Dataset: {dataset_path}")
    stats.append(f"%Profile rejection : {p_rejection}")
    stats.append(f"%Model rejection   : {float((y_pred == 1).mean())}")

    # Confusion matrix (and FPR/FNR) figure
    fig_cm = os.path.join(out_dir, "conf_matrix.png")
    cm_stats, _, _ = print_confusion_matrix(fig_cm, y_test.to_numpy(), y_pred)
    stats += cm_stats

    # Proper ROC/PR AUC using probs
    try:
        roc = float(roc_auc_score(y_test, probs))
    except ValueError:
        roc = 0.0
    try:
        pr = float(average_precision_score(y_test, probs))
    except ValueError:
        pr = 0.0

    stats.append(f"ROC-AUC = {round(roc,3)}  ({round(roc*100,1)}%)")
    stats.append(f"PR-AUC = {round(pr,3)}  ({round(pr*100,1)}%)")

    # Save eval stats
    write_stats(os.path.join(out_dir, "eval.stats"), "\n".join(stats))

    # Plot CDF (needs latency + y_pred + optional read-only filter)
    X_test_df = X_test.copy(deep=True)
    if X_test_latency is not None:
        X_test_df["latency"] = X_test_latency.values
    else:
        # If latency isn't present, skip CDF
        X_test_df["latency"] = np.nan

    X_test_df["y_test"] = y_test.values
    X_test_df["y_pred"] = y_pred

    # Keep read IO only (same logic you used)
    if "io_type" in X_test_df.columns and "readonly" not in dataset_path:
        X_test_df = X_test_df[X_test_df["io_type"] == 1]

    fig_cdf = os.path.join(out_dir, "tree_cdf.png")
    title = (
        f"Read-IO Latency CDF [ROC-AUC = {round(roc,3)} = {round(roc*100,1)}%]\n"
        f"model = {model_name} ; eval = {percent_eval}%"
    )
    # Only plot if latency is real
    if X_test_df["latency"].notna().all():
        plot_latency_cdf(fig_cdf, X_test_df, title)
    else:
        # If latency has NaNs, avoid confusing plots
        pass

    print("===== output dir : " + out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", required=True, help="Path to the dataset CSV", type=str)
    parser.add_argument("-train_eval_split", default="80_20", help="Train/eval split like 80_20", type=str)
    parser.add_argument("-max_depth", default=None, help="Decision tree max depth (int) or omit", type=int)
    parser.add_argument("-min_samples_leaf", default=1, help="Decision tree min_samples_leaf", type=int)
    parser.add_argument("-seed", default=42, help="Random seed", type=int)
    args = parser.parse_args()

    train_model(
        dataset_path=args.dataset,
        train_eval_split=args.train_eval_split,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
    )