#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>

#include "nvbit_tool.h"
#include "nvbit.h"
#include "utils/utils.h"

#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <tuple>
#include <string>

std::unordered_map<void *, std::tuple<int, size_t>> tracking_map;

int callback_tracker = 0;
int free_count = 0;
int alloc_count = 0;
int snapshot_number = 0;

uint32_t kernel_id = 0;
uint64_t tot_app_instrs = 0;
__managed__ uint64_t counter = 0;

uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
uint32_t ker_begin_interval = 0;
uint32_t ker_end_interval = UINT32_MAX;
int verbose = 0;
int count_warp_level = 1;
int exclude_pred_off = 0;

pthread_mutex_t mutex;
bool skip_callback_flag = false;
std::unordered_set<CUfunction> already_instrumented;

extern "C" __device__ __noinline__ void count_instrs(int predicate,
                                                     int count_warp_level) {
    const int active_mask = __ballot_sync(__activemask(), 1);
    const int predicate_mask = __ballot_sync(__activemask(), predicate);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;
    const int num_threads = __popc(predicate_mask);
    if (first_laneid == laneid) {
        if (count_warp_level) {
            if (num_threads > 0) atomicAdd((unsigned long long *)&counter, 1);
        } else {
            atomicAdd((unsigned long long *)&counter, num_threads);
        }
    }
}

void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    GET_VAR_INT(instr_begin_interval, "INSTR_BEGIN", 0,
                "Beginning of the instruction interval where to apply instrumentation");
    GET_VAR_INT(instr_end_interval, "INSTR_END", UINT32_MAX,
                "End of the instruction interval where to apply instrumentation");
    GET_VAR_INT(ker_begin_interval, "KERNEL_BEGIN", 0,
                "Beginning of the kernel launch interval where to apply instrumentation");
    GET_VAR_INT(ker_end_interval, "KERNEL_END", UINT32_MAX,
                "End of the kernel launch interval where to apply instrumentation");
    GET_VAR_INT(count_warp_level, "COUNT_WARP_LEVEL", 1,
                "Count warp level or thread level instructions");
    GET_VAR_INT(exclude_pred_off, "EXCLUDE_PRED_OFF", 0,
                "Exclude predicated off instruction from count");
    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");

    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&mutex, &attr);

    std::string pad(100, '-');
    printf("%s\n", pad.c_str());
}

void nvbit_at_term() {}

void nvbit_at_ctx_init(CUcontext ctx) {}

void nvbit_at_ctx_term(CUcontext ctx) {}

void nvbit_tool_init(CUcontext ctx) {}

void nvbit_at_graph_node_launch(CUcontext ctx, CUfunction func,
                                CUstream stream, uint64_t launch_handle) {}

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    std::vector<CUfunction> related_functions =
        nvbit_get_related_functions(ctx, func);
    related_functions.push_back(func);

    for (auto f : related_functions) {
        if (!already_instrumented.insert(f).second) {
            continue;
        }

        const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
        if (verbose) {
            printf("inspecting %s - num instrs %ld\n",
                   nvbit_get_func_name(ctx, f), instrs.size());
        }

        for (auto i : instrs) {
            if (i->getIdx() >= instr_begin_interval &&
                i->getIdx() < instr_end_interval) {
                nvbit_insert_call(i, "count_instrs", IPOINT_BEFORE);
                if (exclude_pred_off) {
                    nvbit_add_call_arg_guard_pred_val(i);
                } else {
                    nvbit_add_call_arg_const_val32(i, 1);
                }
                nvbit_add_call_arg_const_val32(i, count_warp_level);
            }
        }
    }
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    if (skip_callback_flag) return;
    skip_callback_flag = true;

    if (cbid == API_CUDA_cuMemAlloc_v2) {
        callback_tracker++;
        if (callback_tracker % 2 == 0) {
            cuMemAlloc_v2_params *p = (cuMemAlloc_v2_params *)params;
            printf("Found an allocation!\n");
            tracking_map.insert({(void *)*p->dptr, std::make_tuple(alloc_count++, p->bytesize)});
            printf("%p, %zu\n", (void *)*p->dptr, p->bytesize);
        }
    }

    if (cbid == API_CUDA_cuMemFree_v2) {
        callback_tracker++;
        if (callback_tracker % 2 == 0) {
            printf("Found a free!\n");
            cuMemFree_v2_params *p = (cuMemFree_v2_params *)params;
            tracking_map.erase((void *)p->dptr);
            printf("%p\n", (void *)p->dptr);
            for (const auto &pair : tracking_map) {
                printf("Address: %p, Number: %d, Size: %zu\n", pair.first,
                       std::get<0>(pair.second), std::get<1>(pair.second));
            }
        }
    }

    if (cbid == API_CUDA_cuMemcpyHtoD_v2) {
        printf("Encountered a memcpy HtoD!\n");
        cuMemcpyHtoD_v2_params *p = (cuMemcpyHtoD_v2_params *)params;
        printf("%p, %p, %zu\n", (void *)p->dstDevice, p->srcHost, p->ByteCount);
    }

    if (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchGrid || cbid == API_CUDA_cuLaunchGridAsync ||
        cbid == API_CUDA_cuLaunchKernel ||
        cbid == API_CUDA_cuLaunchKernelEx ||
        cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
        cbid == API_CUDA_cuLaunchCooperativeKernel ||
        cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz) {
        cuLaunch_params *p = (cuLaunch_params *)params;

        if (!is_exit) {
            pthread_mutex_lock(&mutex);
            instrument_function_if_needed(ctx, p->f);
        } else {
            CUDA_SAFECALL(cudaDeviceSynchronize());

            if (1) {
                for (const auto &pair : tracking_map) {
                    size_t bytes = std::get<1>(pair.second);
                    int alloc_number = std::get<0>(pair.second);
                    void *tmp = pair.first;

                    std::string fname = std::to_string(kernel_id) + "_" +
                                        std::to_string(alloc_number) + ".txt";
                    FILE *f = fopen(fname.c_str(), "w");

                    std::vector<uint8_t> buffer;
                    buffer.resize(bytes);
                    cudaMemcpy(buffer.data(), tmp, bytes, cudaMemcpyDeviceToHost);

                    for (auto i : buffer) {
                        fprintf(f, "%hhu ", i);
                    }
                    fclose(f);
                }
            }
            kernel_id++;
            pthread_mutex_unlock(&mutex);
        }
    }

    skip_callback_flag = false;
}
