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
    const int active_mask = __ballot_sync(__activemask(), 1);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;
    const int num_threads = __popc(active_mask);
    if (first_laneid == laneid) {
        int global_tb = blockIdx.y * gridDim.x + blockIdx.x;
        atomicAdd(&bbv_local[global_tb * num_basic_blocks + bb], num_threads);
    }
}
