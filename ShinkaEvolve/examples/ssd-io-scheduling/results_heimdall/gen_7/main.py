from typing import Dict, List


# ---------------------------------------------------------------------------
# Feature columns — must match evaluate.py's FEATURE_COLUMNS.
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


# EVOLVE-BLOCK-START
def predict(features: Dict[str, float]) -> int:
    """
    Heuristic for SSD I/O admission control.

    Args:
        features: dict mapping feature name -> float value.
                  Keys: size, queue_len, prev_queue_len_1/2/3,
                        prev_latency_1/2/3, prev_throughput_1/2/3, latency.

    Returns:
        1  →  REJECT  (predicted slow/high-latency I/O — block or hedge)
        0  →  KEEP    (predicted fast/normal I/O — let through)

    Fitness being maximised:
        combined_score = 0.7 * weighted_f1 + 0.3 * (1 - false_admit_rate)

    False admits (predicting KEEP for a truly slow I/O) are the most costly
    error because they cause direct tail-latency spikes at the SSD.
    False rejects (predicting REJECT for a fast I/O) are less costly.
    """
    latency   = features["latency"]
    queue_len = features["queue_len"]
    size      = features["size"]

    prev_latency_avg = (
        features["prev_latency_1"]
        + features["prev_latency_2"]
        + features["prev_latency_3"]
    ) / 3.0

    prev_queue_avg = (
        features["prev_queue_len_1"]
        + features["prev_queue_len_2"]
        + features["prev_queue_len_3"]
    ) / 3.0

    # Calculate throughput trend (declining throughput indicates system stress)
    prev_throughput_avg = (
        features["prev_throughput_1"]
        + features["prev_throughput_2"]
        + features["prev_throughput_3"]
    ) / 3.0

    # Throughput slope (negative = declining performance)
    if features["prev_throughput_2"] > 0:
        throughput_slope = (features["prev_throughput_1"] - features["prev_throughput_2"]) / features["prev_throughput_2"]
    else:
        throughput_slope = 0.0

    # Adaptive latency threshold based on recent history and throughput trends
    base_threshold = max(120.0, prev_latency_avg * 1.2)

    # Lower threshold when throughput is declining (early warning)
    if throughput_slope < -0.15:
        adaptive_threshold = base_threshold * 0.7
    elif throughput_slope < -0.05:
        adaptive_threshold = base_threshold * 0.85
    else:
        adaptive_threshold = base_threshold

    # Rule 1: adaptive latency spike detection
    if latency > adaptive_threshold:
        return 1

    # Rule 1b: absolute high latency ceiling
    if latency > 220.0:
        return 1
=======

    # Rule 2: queue congestion
    if queue_len > 8:
        return 1

    # Rule 3: sustained high latency trend
    if prev_latency_avg > 150.0:
        return 1

    # Rule 4: large request into a non-empty queue
    if size >= 65536 and queue_len > 2:
        return 1

    # Rule 5: compounding pressure
    if prev_queue_avg > 6 and prev_latency_avg > 100.0:
        return 1

    # Rule 6: throughput degradation with moderate latency
    if throughput_slope < -0.2 and latency > 80.0:
        return 1

    # Rule 7: sustained throughput decline with queue pressure
    if prev_throughput_avg > 0 and throughput_slope < -0.1 and queue_len > 4:
        return 1

    return 0
# EVOLVE-BLOCK-END


# ---------------------------------------------------------------------------
# run_experiment — called by evaluate.py via run_shinka_eval.
# Must accept the kwargs produced by get_experiment_kwargs().
# Must return a dict with at least "predictions" and "labels".
# ---------------------------------------------------------------------------

def run_experiment(
    features: List[Dict[str, float]],
    labels: List[int],
    seed: int = 42,
) -> Dict:
    """
    Run the heuristic over the provided feature list and return predictions.
    The evaluator compares predictions against labels to compute metrics.
    """
    predictions = [predict(f) for f in features]
    return {
        "predictions": predictions,
        "labels":      labels,
    }