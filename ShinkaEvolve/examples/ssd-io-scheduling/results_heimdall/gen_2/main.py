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

    # Compute risk score (higher = more likely to be slow)
    risk_score = 0.0

    # Current latency factor (heavily weighted)
    risk_score += max(0, (latency - 50.0) / 100.0) * 3.0

    # Queue depth factor (exponential penalty for high queue)
    if queue_len > 0:
        risk_score += (queue_len / 10.0) ** 1.5 * 2.0

    # Historical latency trend
    risk_score += max(0, (prev_latency_avg - 80.0) / 120.0) * 1.5

    # Size penalty (large requests are riskier)
    if size >= 32768:
        risk_score += (size / 65536.0) * 1.0

    # Queue growth trend (current vs recent average)
    queue_trend = queue_len - prev_queue_avg
    if queue_trend > 0:
        risk_score += queue_trend / 5.0 * 1.2

    # Interaction: large requests into growing queues
    if size >= 32768 and queue_trend > 1:
        risk_score += 1.0

    # Interaction: sustained pressure (high queue + high latency)
    if prev_queue_avg > 4 and prev_latency_avg > 100:
        risk_score += 1.5

    # Be aggressive: reject if risk score exceeds threshold
    return 1 if risk_score > 2.5 else 0
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