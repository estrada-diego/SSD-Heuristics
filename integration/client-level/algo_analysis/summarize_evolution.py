#!/usr/bin/env python3
"""
Summarize Shinka evolution results across all traces.

Usage:
    python summarize_evolution.py --data_dir <path-to-data>
    python summarize_evolution.py --data_dir ../data
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path


def find_evolution_dirs(data_dir: Path):
    """Yield (trace_label, device_idx, evolution_dir) for every evolution run."""
    for evo_dir in sorted(data_dir.rglob("evolution_device_*")):
        if not evo_dir.is_dir():
            continue
        # path: .../nvme.../shinka/training_results/evolution_device_X
        try:
            dev_idx = int(evo_dir.name.split("_")[-1])
        except ValueError:
            continue
        parts = evo_dir.parts
        # build a short label from the path
        try:
            nvme_part  = next(p for p in parts if "nvme" in p and "..." in p)
            nvme_idx   = parts.index(nvme_part)
            cfg_part   = parts[nvme_idx - 1]
            src2_part  = parts[nvme_idx - 2]
            src1_part  = parts[nvme_idx - 3]
        except (StopIteration, IndexError, ValueError):
            src1_part = src2_part = cfg_part = "unknown"

        def src_name(s):
            if "alibaba" in s:
                m = re.search(r"alibaba_([\d.]+)", s)
                return f"Ali {m.group(1)}" if m else "Alibaba"
            if "msr" in s:
                m = re.search(r"(src1|prxy)_[\d.]+", s)
                return f"MSR {m.group(0)}" if m else "MSR"
            if "tencent" in s:
                return "Tencent"
            return s.split(".")[-1][:14]

        rerate = re.search(r"rerate_([\d.]+)", cfg_part)
        resize = re.search(r"resize_([\d.]+)", cfg_part)
        cfg_str = ""
        if rerate: cfg_str += f"rerate×{rerate.group(1)}"
        if resize: cfg_str += f" resize×{resize.group(1)}"
        label = f"{src_name(src1_part)} × {src_name(src2_part)} ({cfg_str.strip()})"

        yield label, dev_idx, evo_dir


def load_gen_metrics(evo_dir: Path) -> list[dict]:
    """Return list of {gen, score, f1, far, frr} sorted by generation."""
    results = []
    for gen_dir in sorted(evo_dir.glob("gen_*"), key=lambda p: int(p.name.split("_")[1])):
        metrics_path = gen_dir / "results" / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        pub = m.get("public", m)
        results.append({
            "gen":   int(gen_dir.name.split("_")[1]),
            "score": pub.get("combined_score"),
            "f1":    pub.get("f1"),
            "far":   pub.get("false_admit_rate"),
            "frr":   pub.get("false_reject_rate"),
            "exec_time_mean_us": m.get("execution_time_mean", 0) * 1e6,
        })
    return results


def load_attempt_costs(evo_dir: Path) -> dict:
    """Sum up LLM costs and count attempts across all generations."""
    total_cost = 0.0
    total_attempts = 0
    successful_attempts = 0
    models_used: set[str] = set()
    first_ts = None
    last_ts = None

    for meta_path in evo_dir.rglob("metadata.json"):
        try:
            with open(meta_path) as f:
                m = json.load(f)
        except Exception:
            continue
        total_attempts += 1
        cost = m.get("llm_cost", 0) or 0
        total_cost += cost
        if m.get("success"):
            successful_attempts += 1
        model = m.get("llm_model")
        if model:
            models_used.add(model)
        ts_str = m.get("timestamp")
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str)
                if first_ts is None or ts < first_ts:
                    first_ts = ts
                if last_ts is None or ts > last_ts:
                    last_ts = ts
            except ValueError:
                pass

    duration_mins = None
    if first_ts and last_ts:
        duration_mins = (last_ts - first_ts).total_seconds() / 60

    return {
        "total_cost_usd": total_cost,
        "total_attempts": total_attempts,
        "successful_attempts": successful_attempts,
        "models_used": sorted(models_used),
        "first_ts": first_ts.isoformat() if first_ts else None,
        "last_ts":  last_ts.isoformat()  if last_ts  else None,
        "duration_mins": duration_mins,
    }


def load_best_metrics(evo_dir: Path) -> dict | None:
    """Load metrics for the best program."""
    best_metrics = evo_dir / "best" / "results" / "metrics.json"
    if not best_metrics.exists():
        # fall back to highest-scoring gen
        gen_metrics = load_gen_metrics(evo_dir)
        if not gen_metrics:
            return None
        return max(gen_metrics, key=lambda x: x["score"] or 0)
    with open(best_metrics) as f:
        m = json.load(f)
    pub = m.get("public", m)
    return {
        "gen":   "best",
        "score": pub.get("combined_score"),
        "f1":    pub.get("f1"),
        "far":   pub.get("false_admit_rate"),
        "frr":   pub.get("false_reject_rate"),
        "exec_time_mean_us": m.get("execution_time_mean", 0) * 1e6,
    }


def sqlite_program_count(evo_dir: Path) -> int | None:
    db = evo_dir.parent / "programs.sqlite"
    if not db.exists():
        return None
    try:
        con = sqlite3.connect(db)
        cur = con.execute("SELECT COUNT(*) FROM programs")
        count = cur.fetchone()[0]
        con.close()
        return count
    except Exception:
        return None


def summarize(data_dir: Path) -> None:
    all_runs = list(find_evolution_dirs(data_dir))
    if not all_runs:
        print("No evolution directories found.")
        return

    grand_cost = 0.0
    grand_duration = 0.0
    grand_attempts = 0

    print("=" * 80)
    print("SHINKA EVOLUTION SUMMARY")
    print("=" * 80)

    for label, dev_idx, evo_dir in all_runs:
        print(f"\n{'─' * 80}")
        print(f"Trace : {label}")
        print(f"Device: {dev_idx}  ({evo_dir})")

        # --- generation-by-generation progress ---
        gen_metrics = load_gen_metrics(evo_dir)
        if gen_metrics:
            print(f"\n  Generation progress ({len(gen_metrics)} gens recorded):")
            print(f"  {'Gen':>4}  {'Score':>7}  {'F1':>7}  {'FAR':>7}  {'FRR':>7}  {'ExecTime(µs)':>13}")
            def fmt(v, w=7): return f"{v:{w}.4f}" if v is not None else f"{'N/A':>{w}}"
        for g in gen_metrics:
                print(f"  {g['gen']:>4}  {fmt(g['score'])}  {fmt(g['f1'])}  "
                      f"{fmt(g['far'])}  {fmt(g['frr'])}  {g['exec_time_mean_us']:>13.2f}")

        # --- best program ---
        best = load_best_metrics(evo_dir)
        if best:
            def pct(v): return f"{v*100:.1f}%" if v is not None else "N/A"
            print(f"\n  Best heuristic:")
            print(f"    combined_score   : {best['score']:.4f}" if best['score'] is not None else "    combined_score   : N/A")
            print(f"    F1               : {best['f1']:.4f}" if best['f1'] is not None else "    F1               : N/A")
            print(f"    false_admit_rate : {pct(best['far'])}  ← slow I/Os wrongly admitted")
            print(f"    false_reject_rate: {pct(best['frr'])}  ← fast I/Os wrongly rejected")
            print(f"    exec_time_mean   : {best['exec_time_mean_us']:.2f} µs per decision")

        # --- cost & timing ---
        costs = load_attempt_costs(evo_dir)
        print(f"\n  LLM cost & timing:")
        print(f"    total LLM cost   : ${costs['total_cost_usd']:.4f}")
        print(f"    total attempts   : {costs['total_attempts']}  ({costs['successful_attempts']} successful)")
        if costs['duration_mins'] is not None:
            h, m = divmod(costs['duration_mins'], 60)
            print(f"    wall-clock time  : {int(h)}h {m:.0f}m  ({costs['duration_mins']:.1f} mins)")
        if costs['first_ts']:
            print(f"    started          : {costs['first_ts']}")
            print(f"    finished         : {costs['last_ts']}")
        if costs['models_used']:
            print(f"    models used      : {', '.join(costs['models_used'])}")

        # --- program database ---
        n_programs = sqlite_program_count(evo_dir)
        if n_programs is not None:
            print(f"    programs in DB   : {n_programs}")

        grand_cost += costs['total_cost_usd']
        grand_attempts += costs['total_attempts']
        if costs['duration_mins']:
            grand_duration += costs['duration_mins']

    # --- grand totals ---
    print(f"\n{'=' * 80}")
    print("TOTALS ACROSS ALL RUNS")
    print(f"  Total LLM cost : ${grand_cost:.4f}")
    print(f"  Total attempts : {grand_attempts}")
    h, m = divmod(grand_duration, 60)
    print(f"  Total wall time: {int(h)}h {m:.0f}m  (sum across all devices/traces)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True,
                        help="Path to the client-level data directory")
    args = parser.parse_args()
    summarize(Path(args.data_dir).resolve())


if __name__ == "__main__":
    main()
