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
    Cascading filter heuristic for SSD I/O admission control.
    
    Uses a multi-stage cascade where each filter specializes in detecting
    different types of problematic I/O patterns. Any filter can trigger rejection.

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
    # Extract features
    latency = features["latency"]
    queue_len = features["queue_len"]
    size = features["size"]
    
    prev_latencies = [features["prev_latency_1"], features["prev_latency_2"], features["prev_latency_3"]]
    prev_queues = [features["prev_queue_len_1"], features["prev_queue_len_2"], features["prev_queue_len_3"]]
    prev_throughputs = [features["prev_throughput_1"], features["prev_throughput_2"], features["prev_throughput_3"]]
    
    # === FILTER 1: Immediate Danger Detection ===
    # Catch obviously problematic requests immediately
    if latency > 200.0:
        return 1
    if queue_len > 8:
        return 1
    if latency > 120.0 and queue_len > 3:
        return 1
    if size >= 65536 and queue_len > 4:
        return 1
    
    # === FILTER 2: System Saturation Detection ===
    # Detect when system is approaching saturation using exponential weighting
    ewa_latency = 0.6 * prev_latencies[0] + 0.3 * prev_latencies[1] + 0.1 * prev_latencies[2]
    ewa_queue = 0.6 * prev_queues[0] + 0.3 * prev_queues[1] + 0.1 * prev_queues[2]
    ewa_throughput = 0.6 * prev_throughputs[0] + 0.3 * prev_throughputs[1] + 0.1 * prev_throughputs[2]
    
    # Saturation indicators
    latency_pressure = latency / max(ewa_latency, 30.0)  # Avoid division by zero
    queue_pressure = queue_len / max(ewa_queue, 1.0)
    throughput_decline = max(0, (ewa_throughput - prev_throughputs[0]) / max(ewa_throughput, 1000.0))
    
    saturation_score = latency_pressure + queue_pressure + throughput_decline * 2.0
    if saturation_score > 3.2:
        return 1
    
    # === FILTER 3: Trend-Based Prediction ===
    # Detect deteriorating trends that predict future problems
    
    # Latency acceleration (is latency trend accelerating?)
    lat_velocity = prev_latencies[0] - prev_latencies[2]  # Recent change
    lat_acceleration = (prev_latencies[0] - prev_latencies[1]) - (prev_latencies[1] - prev_latencies[2])
    
    # Queue momentum
    queue_velocity = prev_queues[0] - prev_queues[2]
    queue_acceleration = (prev_queues[0] - prev_queues[1]) - (prev_queues[1] - prev_queues[2])
    
    # Throughput momentum (negative = declining)
    throughput_velocity = prev_throughputs[0] - prev_throughputs[2]
    
    # Trend-based rejection
    if lat_velocity > 40.0 and lat_acceleration > 10.0:  # Accelerating latency increase
        return 1
    if queue_velocity > 2.0 and latency > 80.0:  # Growing queue + elevated latency
        return 1
    if throughput_velocity < -2000.0 and queue_len > 2:  # Throughput collapse + queue buildup
        return 1
    
    # === FILTER 4: Compound Risk Detection ===
    # Detect combinations of moderate risks that compound into high risk
    
    # Multi-dimensional risk space
    normalized_latency = min(latency / 150.0, 2.0)  # Cap at 2x
    normalized_queue = min(queue_len / 6.0, 2.0)
    normalized_size = min(size / 32768.0, 2.0)
    normalized_ewa_lat = min(ewa_latency / 100.0, 2.0)
    normalized_ewa_queue = min(ewa_queue / 4.0, 2.0)
    
    # Compound risk using geometric mean (emphasizes when multiple factors are elevated)
    compound_risk = (normalized_latency * normalized_queue * normalized_size * 
                    normalized_ewa_lat * normalized_ewa_queue) ** 0.2
    
    if compound_risk > 1.15:
        return 1
    
    # === FILTER 5: Adaptive Threshold Based on System State ===
    # Lower thresholds when system is already under stress
    
    system_stress = (ewa_latency / 100.0) + (ewa_queue / 5.0) + max(0, -throughput_velocity / 3000.0)
    
    # Adaptive thresholds that get more aggressive under stress
    stress_multiplier = 1.0 + system_stress * 0.3
    
    if latency > (90.0 / stress_multiplier):
        return 1
    if queue_len > (5.0 / stress_multiplier):
        return 1
    if size >= 32768 and (latency > (70.0 / stress_multiplier) or queue_len > (3.0 / stress_multiplier)):
        return 1
    
    # === FILTER 6: Pattern-Based Detection ===
    # Detect specific problematic patterns
    
    # Pattern: Sustained moderate pressure
    if (ewa_latency > 60.0 and ewa_queue > 2.5 and 
        latency > 50.0 and queue_len > 1):
        return 1
    
    # Pattern: Large request into unstable system
    if (size >= 32768 and 
        (lat_velocity > 15.0 or queue_velocity > 1.0 or throughput_velocity < -1000.0)):
        return 1
    
    # Pattern: Oscillating system (high variance in recent measurements)
    lat_variance = ((prev_latencies[0] - ewa_latency) ** 2 + 
                   (prev_latencies[1] - ewa_latency) ** 2 + 
                   (prev_latencies[2] - ewa_latency) ** 2) / 3.0
    if lat_variance > 1000.0 and latency > 70.0:
        return 1
    
    # If no filter triggered, keep the request
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