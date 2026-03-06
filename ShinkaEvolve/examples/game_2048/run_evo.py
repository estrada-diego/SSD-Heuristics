#!/usr/bin/env python3
from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results_heimdall"
RESULTS.mkdir(exist_ok=True)

job_config = LocalJobConfig(eval_program_path=str(HERE / "evaluate.py"))

parent_config = dict(
    parent_selection_strategy="weighted",
    parent_selection_lambda=10.0,
)

db_config = DatabaseConfig(
    db_path=str(RESULTS / "programs.sqlite"),
    num_islands=2,
    archive_size=40,
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


search_task_sys_msg = """
You are a world-class optimization expert and algorithm engineer in game design and AI.
You have to optimize the AI used in the famous game 2048.

## Game variant
- The goal of this variant is to reach the 2,048 value with the *least amount of actions possible*.


## Problem Constraints
- The game only lasts for *2,000 steps (actions)*.
- Each `get_best_move` function call must complete *within 100ms*. Make sure that your solution is highly efficient and suitable for implementation in Python.


Your goal is to improve the performance of the program and maximize the `combined_score` by suggesting improvements.

Typical approaches involve utilizing shallow search and heuristic for evaluating how "good" the board is.

Try diverse approaches to solve the problem. Think outside the box.
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
    llm_models=[...],
    llm_kwargs=...,
    meta_rec_interval=10,
    meta_llm_models=[...],
    meta_llm_kwargs=...,
    embedding_model="amazon.titan-embed-text-v2:0",
    code_embed_sim_threshold=0.995,
    novelty_llm_models=[...],
    novelty_llm_kwargs=...,
    llm_dynamic_selection="ucb1",
    llm_dynamic_selection_kwargs=dict(exploration_coef=1.0),
    use_text_feedback=True,
    init_program_path=str(HERE / "initial.py"),
    results_dir=str(RESULTS),
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
    results_data = main()
#!/usr/bin/env python3
from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

job_config = LocalJobConfig(eval_program_path="evaluate.py")


parent_config = dict(
    parent_selection_strategy="weighted",
    parent_selection_lambda=10.0,
)


db_config = DatabaseConfig(
    db_path="evolution_db.sqlite",
    num_islands=2,
    archive_size=40,
    # Inspiration parameters
    elite_selection_ratio=0.3,
    num_archive_inspirations=4,
    num_top_k_inspirations=2,
    # Island migration parameters
    migration_interval=10,
    migration_rate=0.1,
    island_elitism=True,
    **parent_config,
)


search_task_sys_msg = """
You are a world-class systems programming expert and algorithm engineer specializing in
storage I/O optimization and latency-sensitive systems.

You have to optimize a heuristic function for SSD I/O admission control. 
The goal is to decide, for each incoming I/O request,
whether to REJECT it (return 1) or KEEP it (return 0) before it reaches the SSD.

## Problem
SSDs suffer from severe tail latency under high load. An admission control heuristic
acts as a gatekeeper: it observes lightweight features of each incoming I/O request and
decides in microseconds whether to let it through or reject it (e.g., via hedging or
queueing at a higher level). A good heuristic dramatically reduces p99/p999 latency.

## Input Features (dict keys passed to predict(features))
- size              : I/O request size in bytes (e.g. 4096, 65536)
- queue_len         : current observed queue depth at decision time
- prev_queue_len_1/2/3 : queue depth at the previous 3 time steps
- prev_latency_1/2/3   : observed latency (µs) at the previous 3 time steps
- prev_throughput_1/2/3: observed throughput at the previous 3 time steps
- latency           : current observed latency (µs) at decision time

## Output
- 1 → REJECT (predicted slow / high-latency I/O — block or hedge this request)
- 0 → KEEP   (predicted fast / normal I/O — let it through)

## Evaluation (combined_score, higher is better, max=1.0)
    combined_score = 0.7 × weighted_f1  +  0.3 × (1 − false_admit_rate)

Where:
- weighted_f1        balances precision and recall, penalising false admits (FA) 2× more
                     than false rejects (FR), because admitting a slow I/O causes direct
                     tail-latency spikes at the SSD.
- false_admit_rate   is the fraction of truly slow I/Os that the heuristic let through
                     (false negatives). Minimising this is the primary objective.

## Key Insight
False admits are far more costly than false rejects:
- False admit  (predict KEEP, truth REJECT): a slow I/O hits the SSD → tail latency spike.
- False reject (predict REJECT, truth KEEP): a fast I/O is unnecessarily hedged → minor
  throughput loss but no latency catastrophe.

## Constraints
- Each `predict(features)` call must be extremely fast (target: < 1 µs in Python).
  No model loading, no file I/O, no heavy imports inside predict().
- The heuristic must be stateless per-call (no hidden global mutable state that drifts).
- Allowed: math, statistics, simple Python builtins. Avoid numpy/sklearn inside predict().

## Typical Approaches
- Threshold rules on individual features (latency, queue_len, size).
- Weighted linear combinations of features compared to a threshold.
- Piecewise / decision-tree-style logic.
- Trend detection over the prev_* time series (e.g. slope, acceleration).
- Non-linear combinations (ratios, products) of features.

Think outside the box. Explore diverse strategies. The goal is to maximise combined_score
by correctly identifying slow I/Os before they reach the SSD.
"""


evo_config = EvolutionConfig(
    task_sys_msg=search_task_sys_msg,
    patch_types=["diff", "full", "cross"],
    patch_type_probs=[0.6, 0.3, 0.1],
    num_generations=10,
    max_parallel_jobs=5,
    max_patch_resamples=3,
    max_patch_attempts=3,
    job_type="local",
    language="python",
    llm_models=[
        "us.anthropic.claude-sonnet-4-20250514-v1:0",
    ],
    llm_kwargs=dict(
        temperatures=[0.0, 0.5, 1.0],
        reasoning_efforts=["auto"],
        max_tokens=32768,
    ),
    meta_rec_interval=10,
    meta_llm_models=["gpt-5-nano"],
    meta_llm_kwargs=dict(temperatures=[0.0], max_tokens=16384),
    embedding_model="text-embedding-3-small",
    code_embed_sim_threshold=0.995,
    novelty_llm_models=["gpt-5-nano"],
    novelty_llm_kwargs=dict(temperatures=[0.0], max_tokens=16384),
    llm_dynamic_selection="ucb1",
    llm_dynamic_selection_kwargs=dict(exploration_coef=1.0),
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
    results_data = main()