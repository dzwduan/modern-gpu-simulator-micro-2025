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
int alloc_count = 0;
uint32_t kernel_id = 0;

int verbose = 0;
pthread_mutex_t mutex;
bool skip_callback_flag = false;

void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
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

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    if (skip_callback_flag) return;
    skip_callback_flag = true;

    if (cbid == API_CUDA_cuMemAlloc_v2) {
        callback_tracker++;
        if (callback_tracker % 2 == 0) {
            cuMemAlloc_v2_params *p = (cuMemAlloc_v2_params *)params;
            if (verbose) printf("Found an allocation: %p, %zu\n", (void *)*p->dptr, p->bytesize);
            tracking_map.insert({(void *)*p->dptr, std::make_tuple(alloc_count++, p->bytesize)});
        }
    }

    if (cbid == API_CUDA_cuMemFree_v2) {
        callback_tracker++;
        if (callback_tracker % 2 == 0) {
            cuMemFree_v2_params *p = (cuMemFree_v2_params *)params;
            if (verbose) printf("Found a free: %p\n", (void *)p->dptr);
            tracking_map.erase((void *)p->dptr);
        }
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
        } else {
            CUDA_SAFECALL(cudaDeviceSynchronize());

            printf("Checkpoint: kernel %d - %s - dumping %zu allocations\n",
                   kernel_id, nvbit_get_func_name(ctx, p->f), tracking_map.size());

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
            kernel_id++;
            pthread_mutex_unlock(&mutex);
        }
    }

    skip_callback_flag = false;
}
