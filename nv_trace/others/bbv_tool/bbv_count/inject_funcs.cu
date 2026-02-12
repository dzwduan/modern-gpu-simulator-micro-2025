#include <stdint.h>
#include <stdio.h>

#include "utils/utils.h"

extern "C" __device__ __noinline__ void count_instrs(int num_instrs,
                                                     int count_warp_level,
                                                     int bb,
                                                     uint64_t pbbv,
                                                     uint32_t num_basic_blocks) {
    int *bbv_local = (int *)pbbv;
    if (bbv_local == nullptr) return;
    int global_wid = get_global_warp_id();
    const int active_mask = __ballot_sync(__activemask(), 1);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;
    const int num_threads = __popc(active_mask);
    if (first_laneid == laneid) {
        atomicAdd(&bbv_local[global_wid * num_basic_blocks + bb], num_threads);
    }
}
