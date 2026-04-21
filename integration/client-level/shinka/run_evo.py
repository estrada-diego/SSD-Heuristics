#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

FILE = Path(__file__).resolve()
SHINKA_WORKFLOW_ROOT = FILE.parent
CLIENT_LEVEL_ROOT = SHINKA_WORKFLOW_ROOT.parent
REPO_ROOT = CLIENT_LEVEL_ROOT.parent.parent
SHINKA_REPO_ROOT = REPO_ROOT / "ShinkaEvolve"
if str(SHINKA_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SHINKA_REPO_ROOT))


def _parse_model_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_llm_models(cli_value: str | None) -> List[str]:
    raw = cli_value or os.getenv("SHINKA_LLM_MODELS")
    if raw:
        return _parse_model_list(raw)

    models: List[str] = []
    if os.getenv("OPENAI_API_KEY"):
        models.extend(["gpt-5-mini", "gpt-5-nano"])
    if os.getenv("GEMINI_API_KEY"):
        models.append("gemini-2.5-flash")
    if os.getenv("ANTHROPIC_API_KEY"):
        models.append("claude-sonnet-4-6")
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        models.append("us.anthropic.claude-sonnet-4-6-v1:0")

    deduped: List[str] = []
    for model in models:
        if model not in deduped:
            deduped.append(model)
    if deduped:
        return deduped

    raise RuntimeError(
        "No Shinka LLM models could be inferred. Set SHINKA_LLM_MODELS "
        "(comma-separated) or configure an LLM API key."
    )


def _resolve_embedding_model(cli_value: str | None) -> str:
    if cli_value is not None:
        return cli_value

    env_value = os.getenv("SHINKA_EMBEDDING_MODEL")
    if env_value is not None:
        return env_value

    if os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY"):
        return "text-embedding-3-small"
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        return "amazon.titan-embed-text-v2:0"
    return ""


def _compute_dataset_stats(dataset_path: Path) -> str:
    import csv
    import numpy as np
    feature_cols = [
        "size", "queue_len",
        "prev_queue_len_1", "prev_queue_len_2", "prev_queue_len_3",
        "prev_latency_1", "prev_latency_2", "prev_latency_3",
        "prev_throughput_1", "prev_throughput_2", "prev_throughput_3",
    ]
    data: dict = {c: [] for c in feature_cols}
    labels = []
    try:
        with open(dataset_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for col in feature_cols:
                    if col in row:
                        data[col].append(float(row[col]))
                if "reject" in row:
                    labels.append(int(row["reject"]))
    except Exception:
        return ""
    lines = ["Dataset feature statistics (use these to calibrate thresholds):"]
    for col in feature_cols:
        vals = np.array(data[col])
        if len(vals) > 0:
            lines.append(
                f"  {col:<24} mean={np.mean(vals):>9.1f}  "
                f"p50={np.percentile(vals,50):>9.1f}  "
                f"p90={np.percentile(vals,90):>9.1f}  "
                f"p99={np.percentile(vals,99):>9.1f}"
            )
    if labels:
        r = np.mean(labels)
        lines.append(f"  {'slow_io_rate':<24} {r:.3f}  ({r*100:.1f}% of I/Os are labelled slow/reject)")
    return "\n".join(lines)


def _task_prompt(dataset_stats: str = "") -> str:
    stats_section = f"\n{dataset_stats}\n" if dataset_stats else ""
    return f"""
You are a world-class systems programming expert specializing in storage I/O
optimization and latency-sensitive SSD scheduling.

You must optimize a heuristic function for client-level I/O admission control.
For each incoming read request, the heuristic decides whether to REJECT it
(return 1, meaning redirect/hedge away from the original device) or KEEP it
(return 0, meaning leave it on the original device).

Problem setting:
- SSD tail latency spikes under load.
- The heuristic is called on the fast path and must stay extremely cheap.
- The evolved logic will be exported into C and compiled into the replay-time
  client-level experiment.

Input features available in predict(features):
- size
- queue_len
- prev_queue_len_1 / prev_queue_len_2 / prev_queue_len_3
- prev_latency_1 / prev_latency_2 / prev_latency_3
- prev_throughput_1 / prev_throughput_2 / prev_throughput_3

Output:
- 1 -> REJECT  (predicted slow read; redirect it)
- 0 -> KEEP    (predicted normal read; keep it local)

Fitness to maximize:
  combined_score = 0.5 * MCC + 0.3 * (1 - false_admit_rate) + 0.2 * (1 - false_reject_rate)

  MCC (Matthews Correlation Coefficient) = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
  MCC = 0 for any degenerate solution (reject-all or keep-all), regardless of class balance.
  MCC = 1 only for perfect classification.

Key asymmetry:
- False admit  (predict KEEP for a truly slow I/O) is the worst mistake.
- False reject (predict REJECT for a truly fast I/O) is also penalised.
- Rejecting everything always scores 0.3 max — you must correctly identify BOTH fast and slow I/Os.

Hard deployment constraints:
- predict() must return only 0 or 1.
- predict() must be deterministic, stateless, and microsecond-scale.
- Use only C-translatable Python in the EVOLVE block:
  scalar assignments, arithmetic, comparisons, boolean operators,
  if/elif/else, and return statements.
- Avoid loops, comprehensions, helper functions, recursion, exceptions,
  containers, imports inside predict(), and file/network access.
- Builtins max(), min(), and abs() are acceptable. max()/min() support 2 or more arguments.
- CRITICAL: Never include git conflict markers (=======, <<<<<<<, >>>>>>>) anywhere in your output.
  These are not valid Python and will cause the program to fail immediately with score 0.

Good ideas to explore:
- latency history thresholds
- queue-pressure thresholds
- trend or momentum signals from recent history
- size-aware thresholds
- simple composite risk scores
{stats_section}"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Shinka evolution on a client-level dataset.")
    parser.add_argument("--dataset_path", type=str, required=True, help="CSV dataset for one device.")
    parser.add_argument("--results_dir", type=str, default=None, help="Directory for Shinka outputs.")
    parser.add_argument("--llm_models", type=str, default=None, help="Comma-separated model list.")
    parser.add_argument("--embedding_model", type=str, default=None, help="Embedding model or empty string to disable embeddings.")
    parser.add_argument("--num_generations", type=int, default=int(os.getenv("SHINKA_NUM_GENERATIONS", "40")))
    parser.add_argument("--train_eval_split", type=str, default=os.getenv("SHINKA_TRAIN_EVAL_SPLIT", "50_50"))
    parser.add_argument("--split_section", type=str, choices=["full", "train", "eval"], default=os.getenv("SHINKA_SPLIT_SECTION", "train"))
    parser.add_argument("--max_parallel_jobs", type=int, default=int(os.getenv("SHINKA_MAX_PARALLEL_JOBS", "4")))
    parser.add_argument("--num_islands", type=int, default=int(os.getenv("SHINKA_NUM_ISLANDS", "2")))
    parser.add_argument("--archive_size", type=int, default=int(os.getenv("SHINKA_ARCHIVE_SIZE", "40")))
    parser.add_argument("--disable_meta", action="store_true", help="Disable meta recommendations and novelty judges.")
    parser.add_argument("--init_program_path", type=str, default=None, help="Seed program path; defaults to initial.py.")
    args = parser.parse_args()

    from shinka.core import EvolutionConfig, EvolutionRunner
    from shinka.database import DatabaseConfig
    from shinka.launch import LocalJobConfig

    # Load ShinkaEvolve/.env so API keys are available even when invoked via sudo
    _env_file = SHINKA_REPO_ROOT / ".env"
    if _env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=_env_file, override=False)

    dataset_path = Path(args.dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    dataset_stats = _compute_dataset_stats(dataset_path)

    results_dir = Path(args.results_dir).resolve() if args.results_dir else (SHINKA_WORKFLOW_ROOT / "results" / dataset_path.stem).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SHINKA_PYTHON_EXECUTABLE", sys.executable)

    llm_models = _resolve_llm_models(args.llm_models)
    embedding_model = _resolve_embedding_model(args.embedding_model)

    job_config = LocalJobConfig(
        eval_program_path=str(SHINKA_WORKFLOW_ROOT / "evaluate.py"),
        extra_cmd_args={
            "dataset_path": str(dataset_path),
            "train_eval_split": args.train_eval_split,
            "split_section": args.split_section,
        },
    )

    db_config = DatabaseConfig(
        db_path=str(results_dir / "programs.sqlite"),
        num_islands=args.num_islands,
        archive_size=args.archive_size,
        elite_selection_ratio=0.3,
        num_archive_inspirations=4,
        num_top_k_inspirations=2,
        migration_interval=10,
        migration_rate=0.1,
        island_elitism=True,
        parent_selection_strategy="power_law",
        exploitation_alpha=1.0,
        exploitation_ratio=0.2,
    )

    primary_model = llm_models[0]
    secondary_model = llm_models[-1]
    evo_config = EvolutionConfig(
        task_sys_msg=_task_prompt(dataset_stats),
        patch_types=["diff", "full", "cross"],
        patch_type_probs=[0.6, 0.3, 0.1],
        num_generations=args.num_generations,
        max_parallel_jobs=args.max_parallel_jobs,
        max_patch_resamples=3,
        max_patch_attempts=3,
        job_type="local",
        language="python",
        llm_models=llm_models,
        llm_kwargs={
            "temperatures": [0.0, 0.4, 0.8],
            "reasoning_efforts": ["auto"],
            "max_tokens": 32768,
        },
        meta_rec_interval=None if args.disable_meta else 10,
        meta_llm_models=None if args.disable_meta else [primary_model],
        meta_llm_kwargs=None if args.disable_meta else {"temperatures": [0.0], "max_tokens": 16384},
        novelty_llm_models=None if args.disable_meta else [secondary_model],
        novelty_llm_kwargs=None if args.disable_meta else {"temperatures": [0.0], "max_tokens": 16384},
        embedding_model=embedding_model,
        code_embed_sim_threshold=0.995,
        llm_dynamic_selection="ucb1" if len(llm_models) > 1 else None,
        llm_dynamic_selection_kwargs={"exploration_coef": 1.0} if len(llm_models) > 1 else {},
        use_text_feedback=True,
        init_program_path=args.init_program_path or str(SHINKA_WORKFLOW_ROOT / "initial.py"),
        results_dir=str(results_dir),
    )

    print(f"Dataset      : {dataset_path}")
    print(f"Results dir  : {results_dir}")
    print(f"LLM models   : {llm_models}")
    print(f"Embeddings   : {embedding_model or '(disabled)'}")

    runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        verbose=True,
    )
    runner.run()


if __name__ == "__main__":
    main()
