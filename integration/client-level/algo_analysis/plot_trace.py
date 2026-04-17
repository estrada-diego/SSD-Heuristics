#!/usr/bin/env python3
"""
Plot latency analysis for one or more specific trace directories.

Usage:
    python plot_trace.py --trace_dirs <dir1> [<dir2> ...] [--output_dir <dir>] [--format png|eps]

Each <dir> should be the nvme0n1...nvme1n1 directory for a trace, e.g.:
    .../data/alibaba.../alibaba.../original...modified.rerate_4.00.resize_4.00/nvme0n1...nvme1n1
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ALGOS  = ['baseline', 'hedging', 'linnos', 'flashnet', 'shinka']
LABELS = ['Baseline', 'Hedging', 'LinnOS', 'FlashNet', 'Shinka']
COLORS = ['red', 'magenta', 'lime', 'blue', 'darkcyan']
STYLES = ['s-', '|-', 'd-', '*-.', 'o-']

PERCENTILE_COLS = ['avg', 'p50.0', 'p80.0', 'p90.0', 'p95.0', 'p99.0', 'p99.9', 'p99.99']
X_TICKS         = ['avg', 'p50',   'p80',   'p90',   'p95',   'p99',   'p99.9', 'p99.99']


def load_stats(trace_dir: str) -> pd.DataFrame:
    """Read latency_characteristic.stats for every algorithm in trace_dir."""
    rows = []
    for algo in ALGOS:
        stats_path = os.path.join(trace_dir, algo, "latency_characteristic.stats")
        if not os.path.exists(stats_path):
            print(f"  [skip] {algo}: stats file not found")
            continue
        with open(stats_path) as f:
            lines = f.readlines()
        row = {"algo": algo}
        for col in PERCENTILE_COLS:
            for line in lines:
                if line.split(" = ")[0] == col:
                    row[col] = float(line.split(" = ")[1].split(" us")[0])
                    break
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("algo")


def make_trace_label(trace_dir: str) -> str:
    """Build a short human-readable label from the directory path."""
    parts = Path(trace_dir).parts

    def src_name(s: str) -> str:
        if "tencent" in s:
            return "Tencent"
        if "alibaba" in s:
            m = re.search(r"alibaba_([\d.]+)", s)
            return f"Ali {m.group(1)}" if m else "Alibaba"
        if "msr" in s:
            m = re.search(r"(src1|prxy)_[\d.]+", s)
            return f"MSR {m.group(0)}" if m else "MSR"
        return s.split(".")[-1][:12]

    # parts[-1] = nvme0n1...nvme1n1
    # parts[-2] = config
    # parts[-3] = secondary source
    # parts[-4] = primary source
    src1 = src_name(parts[-4]) if len(parts) >= 4 else ""
    src2 = src_name(parts[-3]) if len(parts) >= 3 else ""
    cfg  = parts[-2].split("...")[-1] if len(parts) >= 2 else parts[-2]

    rerate = re.search(r"rerate_([\d.]+)", cfg)
    resize = re.search(r"resize_([\d.]+)",  cfg)
    rw     = "rw" if "rw_ratio" in cfg else ""

    cfg_parts = []
    if rerate: cfg_parts.append(f"rerate×{rerate.group(1)}")
    if resize: cfg_parts.append(f"resize×{resize.group(1)}")
    if rw:     cfg_parts.append(rw)

    cfg_str = ", ".join(cfg_parts) or cfg[:30]
    return f"{src1} × {src2}\n({cfg_str})"


def plot_trace(trace_dir: str, output_dir: str, fmt: str) -> None:
    label = make_trace_label(trace_dir)
    slug  = re.sub(r"[^\w]+", "_", label.replace("\n", "_")).strip("_")

    df = load_stats(trace_dir)
    if df.empty:
        print(f"No data found in {trace_dir}, skipping.")
        return

    # Convert µs → ms
    df_ms = df / 1000

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # ── Left: absolute latency ────────────────────────────────────────────────
    for algo, lbl, color, style in zip(ALGOS, LABELS, COLORS, STYLES):
        if algo not in df_ms.index:
            continue
        vals = [df_ms.loc[algo, col] for col in PERCENTILE_COLS if col in df_ms.columns]
        ax1.plot(range(len(vals)), vals, style, label=lbl, color=color, linewidth=1.8)

    ax1.set_xticks(range(len(X_TICKS)))
    ax1.set_xticklabels(X_TICKS, rotation=35, ha="center")
    ax1.set_ylabel("Read Latency (ms)")
    ax1.set_xlabel("Percentile")
    ax1.set_title(f"Absolute latency\n{label}")

    # ── Right: % change vs baseline ───────────────────────────────────────────
    if "baseline" in df.index:
        plotted_any = False
        for algo, lbl, color, style in zip(ALGOS[1:], LABELS[1:], COLORS[1:], STYLES[1:]):
            if algo not in df.index:
                continue
            pcts = []
            for col in PERCENTILE_COLS:
                base_v = df.loc["baseline", col] if col in df.columns else None
                algo_v = df.loc[algo, col]       if col in df.columns else None
                if base_v and algo_v and base_v != 0:
                    pcts.append((algo_v - base_v) / base_v * 100)
            if pcts:
                ax2.plot(range(len(pcts)), pcts, style, label=lbl, color=color, linewidth=1.8)
                plotted_any = True

        if plotted_any:
            ax2.axhline(0, color="red", linestyle="--", linewidth=1, label="Baseline")
            ax2.set_xticks(range(len(X_TICKS)))
            ax2.set_xticklabels(X_TICKS, rotation=35, ha="center")
            ax2.set_ylabel("% change vs baseline")
            ax2.set_xlabel("Percentile")
            ax2.set_title(f"Relative improvement\n{label}")
    else:
        ax2.set_visible(False)

    handles, labels_leg = ax1.get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="lower center", ncol=len(ALGOS),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.08))
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"trace_{slug}.{fmt}")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", format=fmt)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_comparison_bar(trace_dirs: list[str], output_dir: str, fmt: str) -> None:
    """One subplot per trace, bars per algorithm with vertical labels on the bars."""
    font_colors = ['black', 'black', 'black', 'white', 'white']

    plt.rcParams.update({'font.size': 16})

    n = len(trace_dirs)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 5.0), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, trace_dir in zip(axes, trace_dirs):
        df = load_stats(trace_dir)
        if df.empty:
            ax.set_visible(False)
            continue

        vals = [df.loc[algo, 'avg'] / 1000 if algo in df.index else 0.0 for algo in ALGOS]
        trace_label = make_trace_label(trace_dir)

        ax.bar(ALGOS, vals, color=COLORS)
        ax.set_ylabel('Average Latency (ms)')
        ax.set_title(trace_label, fontsize=11, pad=14, wrap=True)

        # Algorithm names vertically inside the bars
        ax.set_xticks(range(len(ALGOS)))
        ax.set_xticklabels(LABELS, rotation=90, fontsize=11)
        ax.tick_params(axis='x', which='major', pad=-10)
        for ticklabel, fc in zip(ax.get_xticklabels(), font_colors):
            ticklabel.set_color(fc)
            ticklabel.set_horizontalalignment('center')
            ticklabel.set_verticalalignment('bottom')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = os.path.join(output_dir, f"comparison_bar.{fmt}")
    plt.savefig(out_path, dpi=200, bbox_inches='tight', format=fmt)
    plt.close()
    print(f"  Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--trace_dirs", nargs="+", required=True,
        help="One or more nvme0n1...nvme1n1 directories to analyse."
    )
    parser.add_argument(
        "--output_dir", default=".",
        help="Directory to write output images (default: current directory)."
    )
    parser.add_argument(
        "--format", default="png", choices=["png", "eps", "pdf", "svg"],
        help="Output image format (default: png)."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for trace_dir in args.trace_dirs:
        trace_dir = os.path.abspath(trace_dir)
        print(f"\nProcessing: {trace_dir}")
        plot_trace(trace_dir, args.output_dir, args.format)

    if len(args.trace_dirs) > 1:
        print("\nGenerating comparison bar chart...")
        plot_comparison_bar(
            [os.path.abspath(d) for d in args.trace_dirs],
            args.output_dir, args.format
        )


if __name__ == "__main__":
    main()
