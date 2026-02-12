#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>

#include "nvbit_tool.h"
#include "nvbit.h"
#include "utils/utils.h"

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
int num_sms = 0;

pthread_mutex_t mutex;
bool skip_flag = false;
bool alternate = false;

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

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    num_sms = prop.multiProcessorCount;

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
    if (skip_flag) return;

    if (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchGrid || cbid == API_CUDA_cuLaunchGridAsync ||
        cbid == API_CUDA_cuLaunchKernel ||
        cbid == API_CUDA_cuLaunchKernelEx ||
        cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
        cbid == API_CUDA_cuLaunchCooperativeKernel ||
        cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz) {
        cuLaunch_params *p = (cuLaunch_params *)params;
        cuLaunchKernel_params_st *p_2 = (cuLaunchKernel_params_st *)params;

        if (alternate) {
            alternate = false;
            int blocks;
            int threads = p_2->blockDimX * p_2->blockDimY * p_2->blockDimZ;
            CUDA_SAFECALL(cuOccupancyMaxActiveBlocksPerMultiprocessor(
                &blocks, p->f, threads, p_2->sharedMemBytes));
            printf("Kernel = %s\n", nvbit_get_func_name(ctx, p->f));
            printf("Max Blocks / GPU = %d\n", blocks * num_sms);
            skip_flag = false;
        } else {
            alternate = true;
        }

        if (!is_exit) {
            pthread_mutex_lock(&mutex);
        } else {
            CUDA_SAFECALL(cudaDeviceSynchronize());
            kernel_id++;
            pthread_mutex_unlock(&mutex);
        }
    }
}
