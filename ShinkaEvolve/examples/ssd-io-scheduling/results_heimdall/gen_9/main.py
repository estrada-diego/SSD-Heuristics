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
    Aggressive multi-signal detection heuristic for SSD I/O admission control.
    
    Uses multiplicative pressure scoring and cascading rejection rules to
    aggressively minimize false admits (slow I/Os wrongly admitted).

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
    
    # LAYER 1: Immediate danger - reject without further analysis
    if latency > 120:  # Much more aggressive than 300
        return 1
    if queue_len > 5:  # Much more aggressive than 12
        return 1
    if size >= 65536 and queue_len > 0:  # Any queue with large request
        return 1
    
    # LAYER 2: Historical pressure analysis
    recent_latency_avg = (prev_latencies[0] + prev_latencies[1]) / 2.0
    all_latency_avg = sum(prev_latencies) / 3.0
    recent_queue_avg = (prev_queues[0] + prev_queues[1]) / 2.0
    all_queue_avg = sum(prev_queues) / 3.0
    
    # Sustained pressure patterns
    if all_latency_avg > 60:  # Much lower than previous 30
        return 1
    if recent_latency_avg > 80 and latency > 50:
        return 1
    if all_queue_avg > 3 and queue_len > 2:
        return 1
    
    # LAYER 3: Trend detection (deteriorating conditions)
    latency_rising = prev_latencies[0] > prev_latencies[2] and prev_latencies[0] > 40
    queue_rising = prev_queues[0] > prev_queues[2] and prev_queues[0] > 2
    
    if latency_rising and queue_rising:
        return 1
    if latency_rising and latency > 70:
        return 1
    if queue_rising and queue_len > 3:
        return 1
    
    # LAYER 4: Size-progressive rejection (stricter for larger requests)
    if size >= 49152:  # 48KB+
        if queue_len > 1 or latency > 40 or recent_latency_avg > 50:
            return 1
    elif size >= 32768:  # 32KB+
        if queue_len > 2 or latency > 60 or recent_latency_avg > 60:
            return 1
    elif size >= 16384:  # 16KB+
        if queue_len > 3 and latency > 50:
            return 1
    
    # LAYER 5: Multi-signal cascade (weak signals combine)
    danger_signals = 0
    
    if latency > 50:
        danger_signals += 1
    if queue_len > 2:
        danger_signals += 1
    if size >= 24576:  # 24KB+
        danger_signals += 1
    if recent_latency_avg > 50:
        danger_signals += 1
    if recent_queue_avg > 2.5:
        danger_signals += 1
    if prev_latencies[0] > 60:
        danger_signals += 1
    
    # Reject if multiple moderate signals present
    if danger_signals >= 3:
        return 1
    if danger_signals >= 2 and (latency > 40 or queue_len > 1):
        return 1
    
    # LAYER 6: Multiplicative pressure scoring
    # Normalize features and multiply (amplifies combined pressure)
    latency_factor = max(0, (latency - 20) / 100.0)  # 0 at 20μs, 1 at 120μs
    queue_factor = max(0, queue_len / 8.0)  # 0 at 0, 1 at 8
    size_factor = max(0, (size - 4096) / 61440.0)  # 0 at 4KB, 1 at 64KB
    hist_latency_factor = max(0, (all_latency_avg - 20) / 80.0)
    hist_queue_factor = max(0, all_queue_avg / 6.0)
    
    pressure_product = latency_factor * queue_factor * (1 + size_factor) * (1 + hist_latency_factor) * (1 + hist_queue_factor)
    
    if pressure_product > 0.15:  # Much lower threshold for multiplicative score
        return 1
    
    # LAYER 7: Emergency conditions
    if latency > 90 and queue_len > 1:
        return 1
    if queue_len > 4 and size >= 16384:
        return 1
    if recent_latency_avg > 70 and recent_queue_avg > 2:
        return 1
    
    # LAYER 8: Conservative compound conditions
    compound_score = (latency / 150.0) + (queue_len / 10.0) + (size / 100000.0) + (all_latency_avg / 120.0)
    if compound_score > 0.6:
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