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

    # Core rules from current program (proven effective)
    # Rule 1: current latency spike
    if latency > 200.0:
        return 1

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

    # Enhanced momentum detection from crossover inspiration
    latency_trend = latency - prev_latency_avg
    queue_trend = queue_len - prev_queue_avg

    # Early warning: latency momentum with moderate current latency
    if latency_trend > 40.0 and latency > 120.0:
        return 1

    # Early warning: queue momentum detection
    if queue_trend > 2.0 and queue_len > 4:
        return 1

    # Enhanced size-aware rejection (more aggressive than current)
    if size >= 32768:
        if queue_len > 1 and prev_latency_avg > 80.0:
            return 1
        if queue_len > 3:  # Any significant queue with large request
            return 1

    # Composite risk assessment for borderline cases
    risk_score = 0.0

    # Latency pressure component
    if latency > 100.0:
        risk_score += (latency - 100.0) / 100.0

    # Queue pressure component
    if queue_len > 3:
        risk_score += (queue_len - 3) * 0.25

    # Historical stress component
    if prev_latency_avg > 80.0:
        risk_score += (prev_latency_avg - 80.0) / 120.0

    # Size under load component
    if size >= 16384 and (queue_len > 0 or prev_queue_avg > 1.0):
        risk_score += size / 120000.0

    # Trend momentum component
    if latency_trend > 15.0:
        risk_score += latency_trend / 100.0

    # Reject if composite risk exceeds threshold
    if risk_score > 1.1:  # Slightly more conservative than crossover
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