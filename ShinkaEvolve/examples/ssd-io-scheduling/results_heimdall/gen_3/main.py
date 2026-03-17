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

    # Compute latency trend (recent vs older)
    recent_latency = (features["prev_latency_1"] + features["prev_latency_2"]) / 2.0
    older_latency = features["prev_latency_3"]
    latency_trend = recent_latency - older_latency if older_latency > 0 else 0

    # Weighted risk scoring system
    risk_score = 0.0

    # Current conditions (high weight)
    if latency > 150.0:
        risk_score += 3.0
    elif latency > 100.0:
        risk_score += 1.5
    elif latency > 80.0:
        risk_score += 0.8

    if queue_len > 6:
        risk_score += 2.5
    elif queue_len > 4:
        risk_score += 1.2
    elif queue_len > 2:
        risk_score += 0.6

    # Historical pressure (medium weight)
    if prev_latency_avg > 120.0:
        risk_score += 2.0
    elif prev_latency_avg > 80.0:
        risk_score += 1.0

    if prev_queue_avg > 5:
        risk_score += 1.5
    elif prev_queue_avg > 3:
        risk_score += 0.8

    # Trend analysis (medium weight)
    if latency_trend > 50.0:
        risk_score += 1.8
    elif latency_trend > 20.0:
        risk_score += 0.9

    # Request size factor (lower weight but multiplicative for large requests)
    size_factor = 1.0
    if size >= 65536:
        size_factor = 1.4
        risk_score += 0.8
    elif size >= 32768:
        size_factor = 1.2
        risk_score += 0.4

    # Apply size multiplier
    risk_score *= size_factor

    # Compounding effects
    if queue_len > 3 and prev_latency_avg > 60.0:
        risk_score += 1.0

    if latency > 80.0 and queue_len > 2:
        risk_score += 1.2

    # More aggressive threshold to catch more slow I/Os
    return 1 if risk_score >= 3.5 else 0
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