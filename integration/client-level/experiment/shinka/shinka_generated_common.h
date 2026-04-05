#ifndef SHINKA_GENERATED_COMMON_H
#define SHINKA_GENERATED_COMMON_H

typedef struct {
    long size;
    long queue_len;
    long prev_queue_len_1;
    long prev_queue_len_2;
    long prev_queue_len_3;
    long prev_latency_1;
    long prev_latency_2;
    long prev_latency_3;
    long prev_throughput_1;
    long prev_throughput_2;
    long prev_throughput_3;
} ShinkaFeatures;

static inline double shinka_max_double(double left, double right) {
    return left > right ? left : right;
}

static inline double shinka_min_double(double left, double right) {
    return left < right ? left : right;
}

static inline double shinka_abs_double(double value) {
    return value < 0.0 ? -value : value;
}

static inline int shinka_default_predict(const ShinkaFeatures *features) {
    double size = (double)features->size;
    double queue_len = (double)features->queue_len;
    double prev_latency_avg =
        ((double)features->prev_latency_1 +
         (double)features->prev_latency_2 +
         (double)features->prev_latency_3) / 3.0;
    double prev_queue_avg =
        ((double)features->prev_queue_len_1 +
         (double)features->prev_queue_len_2 +
         (double)features->prev_queue_len_3) / 3.0;

    if ((double)features->prev_latency_1 > 200.0) {
        return 1;
    }
    if (queue_len > 8.0) {
        return 1;
    }
    if (prev_latency_avg > 150.0) {
        return 1;
    }
    if (size >= 65536.0 && queue_len > 2.0) {
        return 1;
    }
    if (prev_queue_avg > 6.0 && prev_latency_avg > 100.0) {
        return 1;
    }
    return 0;
}

#endif
