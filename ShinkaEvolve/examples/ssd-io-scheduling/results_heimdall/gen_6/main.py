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
    Aggressive early rejection heuristic for SSD I/O admission control.

    Prioritizes minimizing false admits (slow I/Os wrongly admitted) over
    false rejects, using cascading rejection rules and conservative thresholds.

    Args:
        features: dict mapping feature name -> float value.

    Returns:
        1  →  REJECT  (predicted slow/high-latency I/O — block or hedge)
        0  →  KEEP    (predicted fast/normal I/O — let through)
    """
    # Extract core features
    latency = features["latency"]
    queue_len = features["queue_len"]
    size = features["size"]

    # Historical data
    prev_latencies = [
        features["prev_latency_1"],
        features["prev_latency_2"],
        features["prev_latency_3"]
    ]
    prev_queues = [
        features["prev_queue_len_1"],
        features["prev_queue_len_2"],
        features["prev_queue_len_3"]
    ]

    # LAYER 1: Immediate danger signals - reject without further analysis
    if latency > 150:  # More aggressive latency threshold
        return 1
    if queue_len > 6:  # More aggressive queue threshold
        return 1
    if size >= 65536 and queue_len > 1:  # Large requests with any queue
        return 1

    # LAYER 2: Historical pressure analysis
    recent_latency_avg = (prev_latencies[0] + prev_latencies[1]) / 2.0
    all_latency_avg = sum(prev_latencies) / 3.0
    recent_queue_avg = (prev_queues[0] + prev_queues[1]) / 2.0

    # Sustained high latency
    if all_latency_avg > 80:  # More conservative threshold
        return 1

    # Recent latency spike pattern
    if recent_latency_avg > 100 and latency > 80:
        return 1

    # LAYER 3: Cascading moderate signals (multiple weak signals = reject)
    danger_signals = 0

    # Signal 1: Moderate current latency
    if latency > 75:
        danger_signals += 1

    # Signal 2: Moderate queue depth
    if queue_len > 3:
        danger_signals += 1

    # Signal 3: Large request size
    if size >= 32768:
        danger_signals += 1

    # Signal 4: Rising latency trend
    if prev_latencies[0] > prev_latencies[2] and prev_latencies[0] > 60:
        danger_signals += 1

    # Signal 5: Rising queue trend
    if prev_queues[0] > prev_queues[2] and recent_queue_avg > 2:
        danger_signals += 1

    # Signal 6: Recent queue congestion
    if recent_queue_avg > 4:
        danger_signals += 1

    # Reject if multiple moderate signals present
    if danger_signals >= 2:
        return 1

    # LAYER 4: Size-based progressive rejection
    if size >= 49152:  # 48KB+
        if queue_len > 2 or latency > 60:
            return 1
    elif size >= 32768:  # 32KB+
        if queue_len > 3 or latency > 80:
            return 1
    elif size >= 16384:  # 16KB+
        if queue_len > 4 and latency > 60:
            return 1

    # LAYER 5: Compound pressure conditions
    pressure_product = (latency / 100.0) * (queue_len / 8.0) * (size / 65536.0)
    if pressure_product > 0.25:  # Conservative compound threshold
        return 1

    # LAYER 6: Historical pattern protection
    if recent_latency_avg > 70 and recent_queue_avg > 3:
        return 1

    # If we get here, the request appears safe
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