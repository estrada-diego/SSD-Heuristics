#ifndef SHINKA_ALGO_H
#define SHINKA_ALGO_H

#include <stdint.h>

#include "shinka_generated_common.h"

#define DEVICE_NUM 2

#define N_HIST 3
#define VERBOSE 0

long add_fetch_cur_queue_len();
void inc_queue_len();
void dec_queue_len();
void set_shinka(int total_io_num);
int shinka_inference(long io_type, long size, uint32_t device, long cur_queue_len);
void update_shinka(long io_queue_len, long io_latency, long io_throughput);
void free_shinka();

#endif
