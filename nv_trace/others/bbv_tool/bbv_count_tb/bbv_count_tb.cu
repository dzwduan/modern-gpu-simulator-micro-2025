#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <map>
#include <string>
#include <vector>
#include <unordered_set>

#include "nvbit_tool.h"
#include "nvbit.h"
#include "utils/utils.h"

uint32_t kernel_id = 0;
int *bbv = nullptr;
unsigned int tot_blocks = 0;
unsigned int kid = 0;
bool first = true;
std::string fname = "bb_log_";

unsigned int num_basic_blocks = 0;
std::map<std::string, int> kbb_map;
std::map<std::string, std::vector<int>> kbb_insns;

uint32_t ker_begin_interval = 0;
uint32_t ker_end_interval = UINT32_MAX;
int verbose = 0;
int count_warp_level = 1;

pthread_mutex_t mutex;
bool skip_callback_flag = false;
std::unordered_set<CUfunction> already_instrumented;

void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    GET_VAR_INT(ker_begin_interval, "KERNEL_BEGIN", 0,
                "Beginning of the kernel launch interval where to apply instrumentation");
    GET_VAR_INT(ker_end_interval, "KERNEL_END", UINT32_MAX,
                "End of the kernel launch interval where to apply instrumentation");
    GET_VAR_INT(count_warp_level, "COUNT_WARP_LEVEL", 1,
                "Count warp level or thread level instructions");
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
        if (!already_instrumented.insert(f).second) continue;

        const CFG_t &cfg = nvbit_get_CFG(ctx, f);
        if (cfg.is_degenerate) continue;

        int local_bb = 0;
        for (auto &bb : cfg.bbs) {
            Instr *i = bb->instrs[0];
            nvbit_insert_call(i, "count_instrs", IPOINT_BEFORE);
            nvbit_add_call_arg_const_val32(i, bb->instrs.size());
            nvbit_add_call_arg_const_val32(i, count_warp_level);
            nvbit_add_call_arg_const_val32(i, local_bb++);
            nvbit_add_call_arg_launch_val64(i, 0);
            nvbit_add_call_arg_const_val32(i, cfg.bbs.size());
        }

        kbb_map.insert({nvbit_get_func_name(ctx, f), (int)cfg.bbs.size()});

        std::vector<int> i_counts;
        for (auto &bb : cfg.bbs) {
            i_counts.push_back(bb->instrs.size());
        }
        kbb_insns.insert({nvbit_get_func_name(ctx, f), i_counts});
    }
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    if (skip_callback_flag) return;
    skip_callback_flag = true;

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
            cudaDeviceSynchronize();

            cuLaunchKernel_params_st *p_test = (cuLaunchKernel_params_st *)params;
            tot_blocks = p_test->gridDimX * p_test->gridDimY * p_test->gridDimZ;

            if (first) {
                first = false;
            } else if (bbv) {
                cudaFree(bbv);
                bbv = nullptr;
            }

            instrument_function_if_needed(ctx, p->f);

            auto it = kbb_map.find(nvbit_get_func_name(ctx, p->f));
            if (it != kbb_map.end()) {
                num_basic_blocks = it->second;
                size_t alloc_size = (size_t)tot_blocks * num_basic_blocks * sizeof(int);
                if (alloc_size > 0) {
                    cudaMallocManaged(&bbv, alloc_size);
                    cudaMemset(bbv, 0, alloc_size);
                }
            }

            nvbit_set_at_launch(ctx, p->f, (uint64_t)bbv);

            if (kernel_id >= ker_begin_interval && kernel_id < ker_end_interval) {
                nvbit_enable_instrumented(ctx, p->f, true);
            } else {
                nvbit_enable_instrumented(ctx, p->f, false);
            }
        } else {
            CUDA_SAFECALL(cudaDeviceSynchronize());

            auto it = kbb_insns.find(nvbit_get_func_name(ctx, p->f));
            std::vector<int> test = it->second;
            FILE *f = fopen((fname + std::to_string(kid) + ".txt").c_str(), "w+");
            kid++;
            fprintf(f, "%s\n", nvbit_get_func_name(ctx, p->f));
            if (bbv) {
                for (unsigned int i = 0; i < tot_blocks; i++) {
                    for (unsigned int j = 0; j < num_basic_blocks; j++) {
                        fprintf(f, "%d ", bbv[i * num_basic_blocks + j] * test[j]);
                    }
                    fprintf(f, "\n");
                }
            }
            fclose(f);
            kernel_id++;
            pthread_mutex_unlock(&mutex);
        }
    }

    skip_callback_flag = false;
}
