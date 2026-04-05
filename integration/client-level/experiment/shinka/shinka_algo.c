#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <pthread.h>
#include <stdlib.h>
#include "2ssds_heuristics_header/shinka_dev_0.h"
#include "2ssds_heuristics_header/shinka_dev_1.h"
#include "atomic.h"
#include "shinka_algo.h"


long queue_len;   // queue len is updated by multiple thread and should be protected by atomic operation.

/* The following four variables are not updated by multiple threads any more.  */
long hist_index;
long * prev_queue_len;     // An array<long> to store historical queue len
long * prev_latency;       // An array<long> to store historical latency 
long * prev_throughput;  // An array<long> to store historical throughput


long add_fetch_cur_queue_len() {
	/*
		Return the current IO queue len.
	*/
	return atomic_inc_fetch(&queue_len);   // the queue len will also count the current IO itself.
}


void inc_queue_len() {
	/*
		Increase the current queue length by 1.
	*/
	atomic_inc(&queue_len);
}


void dec_queue_len() {
	/*
		Decrease the current queue length by 1.
	*/
	atomic_dec(&queue_len);
}

void set_shinka(int total_io_num) {
	// 1. init the global params
	queue_len = 0;
	hist_index = 0;
	
	// 2. init previous queue len
	prev_queue_len = malloc(total_io_num * sizeof(long));
	for (long i = 0; i < total_io_num; i++) {
		prev_queue_len[i] = -1;    // init to `-1` to indicate invalid
	}

	// 3. init previous latency
	prev_latency = malloc(total_io_num * sizeof(long));
	for (long i = 0; i < total_io_num; i++) {
		prev_latency[i] = -1;
	}

	// 4. init previous throughput
	prev_throughput = malloc(total_io_num * sizeof(long));
	for (long i = 0; i < total_io_num; i++) {
		prev_throughput[i] = -1;
	}
}


int shinka_inference(long io_type, long size, uint32_t device, long cur_queue_len) {
	ShinkaFeatures features = {0};
	long cur_hist_index = hist_index;
	int prediction;

	(void)io_type;
	features.size = size;
	features.queue_len = cur_queue_len;

	if (cur_hist_index - 1 >= 0) {
		features.prev_queue_len_1 = prev_queue_len[cur_hist_index - 1];
		features.prev_latency_1 = prev_latency[cur_hist_index - 1];
		features.prev_throughput_1 = prev_throughput[cur_hist_index - 1];
	}
	if (cur_hist_index - 2 >= 0) {
		features.prev_queue_len_2 = prev_queue_len[cur_hist_index - 2];
		features.prev_latency_2 = prev_latency[cur_hist_index - 2];
		features.prev_throughput_2 = prev_throughput[cur_hist_index - 2];
	}
	if (cur_hist_index - 3 >= 0) {
		features.prev_queue_len_3 = prev_queue_len[cur_hist_index - 3];
		features.prev_latency_3 = prev_latency[cur_hist_index - 3];
		features.prev_throughput_3 = prev_throughput[cur_hist_index - 3];
	}

	if (device == 0) {
		prediction = shinka_predict_dev_0(&features);
	} else {
		prediction = shinka_predict_dev_1(&features);
	}

	if (VERBOSE) {
		printf(
			"[dev %d] - [cur_hist_index=%ld] [pred=%d] features = {%ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld}\n",
			device,
			cur_hist_index,
			prediction,
			features.size,
			features.queue_len,
			features.prev_queue_len_1,
			features.prev_queue_len_2,
			features.prev_queue_len_3,
			features.prev_latency_1,
			features.prev_latency_2,
			features.prev_latency_3,
			features.prev_throughput_1,
			features.prev_throughput_2,
			features.prev_throughput_3
		);
	}

	return prediction;
}


void update_shinka(long io_queue_len, long io_latency, long io_throughput){
    /*
		Called by single update_thread in io_replayer.c, to insert latest completed IO's queue len
		when submitted, IO latency, and IO throughput into the historical pool.
        The historical pools to be Updated:
            prev_queue_len,
            prev_latency,
            prev_throughput,
    */

	// update previous queue len
	prev_queue_len[hist_index] = io_queue_len;

	// update previous latency
	prev_latency[hist_index] = io_latency;

	// update previous throughout
	prev_throughput[hist_index] = io_throughput;

	hist_index += 1;
}


void free_shinka() {
	/*
		Free the memory that shinka allocated
	*/
	
	free(prev_queue_len);
	free(prev_latency);
	free(prev_throughput);
}
