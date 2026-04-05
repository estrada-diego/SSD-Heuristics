from typing import Dict, List


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
]


# EVOLVE-BLOCK-START
def predict(features: Dict[str, float]) -> int:
    """
    Heuristic for SSD I/O admission control.

    Returns:
        1 -> REJECT (redirect / hedge this read)
        0 -> KEEP   (leave the read on the original device)
    """
    size = features["size"]
    queue_len = features["queue_len"]
    prev_latency_1 = features["prev_latency_1"]

    prev_latency_avg = (
        prev_latency_1
        + features["prev_latency_2"]
        + features["prev_latency_3"]
    ) / 3.0
    prev_queue_avg = (
        features["prev_queue_len_1"]
        + features["prev_queue_len_2"]
        + features["prev_queue_len_3"]
    ) / 3.0

    if prev_latency_1 > 200.0:
        return 1

    if queue_len > 8:
        return 1

    if prev_latency_avg > 150.0:
        return 1

    if size >= 65536 and queue_len > 2:
        return 1

    if prev_queue_avg > 6.0 and prev_latency_avg > 100.0:
        return 1

    return 0
# EVOLVE-BLOCK-END


def run_experiment(
    features: List[Dict[str, float]],
    labels: List[int],
    seed: int = 42,
) -> Dict:
    predictions = [predict(feature_dict) for feature_dict in features]
    return {
        "predictions": predictions,
        "labels": labels,
    }
