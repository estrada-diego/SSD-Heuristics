#!/usr/bin/env python3
from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

job_config = LocalJobConfig(eval_program_path="evaluate.py")

db_config = DatabaseConfig(
    db_path="evolution_db.sqlite",
    num_islands=2,
    archive_size=40,
    # Inspiration
    elite_selection_ratio=0.3,
    num_archive_inspirations=4,
    num_top_k_inspirations=2,
    # Island migration
    migration_interval=10,
    migration_rate=0.1,
    island_elitism=True,
    # Parent selection
    parent_selection_strategy="power_law",   # "power_law" | "weighted" | "beam_search"
    exploitation_alpha=1.0,                  # sharpness of power-law (0=uniform)
    exploitation_ratio=0.2,                  # chance to pick parent from archive
)

search_task_sys_msg = """
You are a world-class systems programming expert specializing in storage I/O
optimization and latency-sensitive systems.

You must optimize a heuristic function for SSD I/O admission control, inspired
by the Heimdall system (EuroSys '25). The heuristic decides, for each incoming
I/O request, whether to REJECT it (return 1) or KEEP it (return 0).

## Problem
SSDs suffer from severe tail latency under high load. An admission control
heuristic acts as a gatekeeper: it observes lightweight features of each
incoming I/O and decides in microseconds whether to let it through or reject
it (e.g. hedge at a higher storage layer). A good heuristic dramatically
reduces p99/p999 latency.

## Input Features (dict keys passed to predict(features))
- size                  : I/O request size in bytes (e.g. 4096, 65536)
- queue_len             : current observed queue depth at decision time
- prev_queue_len_1/2/3  : queue depth at the previous 3 time steps
- prev_latency_1/2/3    : observed latency (µs) at the previous 3 time steps
- prev_throughput_1/2/3 : observed throughput at the previous 3 time steps
- latency               : current observed latency (µs) at decision time

## Output
- 1 → REJECT  (predicted slow / high-latency — block or hedge this request)
- 0 → KEEP    (predicted fast / normal — let it through)

## Fitness function (combined_score, maximise, range [0, 1])
    combined_score = 0.7 × weighted_f1  +  0.3 × (1 − false_admit_rate)

- weighted_f1       : F1 score with false admits penalised 2× vs false rejects
- false_admit_rate  : fraction of truly slow I/Os the heuristic admitted (FNR)

## Key asymmetry — false admits >> false rejects in cost
- False admit  (predict KEEP, truth REJECT): slow I/O reaches SSD → tail spike.
- False reject (predict REJECT, truth KEEP): fast I/O hedged → minor throughput loss.
Aggressively reducing false admits is the primary lever for improving combined_score.

## Hard constraints on predict()
- Must return only 0 or 1.
- Must be stateless and extremely fast (target < 1 µs per call in Python).
- No file I/O, no model loading, no heavy imports inside predict().
- Allowed inside predict(): math, builtins, simple arithmetic. No numpy/sklearn.

## Approaches to explore
- Threshold rules on latency, queue_len, size.
- Weighted linear scores compared to a threshold.
- Trend / slope detection over prev_* time series.
- Ratios or products of features (e.g. latency × queue_len).
- Piecewise / decision-tree-style logic.
- Exponential moving averages computed inline.

Think outside the box. Diverse strategies are encouraged.
"""

evo_config = EvolutionConfig(
    task_sys_msg=search_task_sys_msg,
    patch_types=["diff", "full", "cross"],
    patch_type_probs=[0.6, 0.3, 0.1],
    num_generations=100,
    max_parallel_jobs=5,
    max_patch_resamples=3,
    max_patch_attempts=3,
    job_type="local",
    language="python",
    llm_models=[
        "gemini-2.5-flash",
        "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "o4-mini",
        "gpt-5-mini",
        "gpt-5-nano",
    ],
    llm_kwargs=dict(
        temperatures=[0.0, 0.5, 1.0],
        reasoning_efforts=["auto"],
        max_tokens=32768,
    ),
    meta_rec_interval=10,
    meta_llm_models=["us.anthropic.claude-sonnet-4-20250514-v1:0"],
    meta_llm_kwargs=dict(temperatures=[0.0], max_tokens=16384),
    embedding_model="amazon.titan-embed-text-v2:0",
    code_embed_sim_threshold=0.995,
    novelty_llm_models=["us.anthropic.claude-sonnet-4-20250514-v1:0"],
    novelty_llm_kwargs=dict(temperatures=[0.0], max_tokens=16384),
    llm_dynamic_selection="ucb1",
    llm_dynamic_selection_kwargs=dict(exploration_coef=1.0),
    use_text_feedback=True,   # passes text_feedback from aggregate_metrics_fn to LLMs
    init_program_path="initial.py",
    results_dir="results_heimdall",
)


def main():
    evo_runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        verbose=True,
    )
    evo_runner.run()


if __name__ == "__main__":
    main()