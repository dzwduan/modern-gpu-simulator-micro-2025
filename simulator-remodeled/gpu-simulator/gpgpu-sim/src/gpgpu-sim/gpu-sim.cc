// Copyright (c) 2023-2025, Rodrigo Huerta, Mojtaba Abaie Shoushtary, Josep-Llorenç Cruz, Antonio González
// Universitat Politecnica de Catalunya
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The Universitat Politecnica de Catalunya nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

// Copyright (c) 2009-2021, Tor M. Aamodt, Wilson W.L. Fung, George L. Yuan,
// Ali Bakhoda, Andrew Turner, Ivan Sham, Vijay Kandiah, Nikos Hardavellas, 
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern 
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "gpu-sim.h"
#include "kernel-scheduler.h"
#include "icnt-handler.h"

#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include "zlib.h"

#include "dram.h"
#include "mem_fetch.h"
#include "shader.h"
#include "shader_trace.h"

#include <time.h>
#include "addrdec.h"
#include "delayqueue.h"
#include "dram.h"
#include "gpu-cache.h"
#include "gpu-misc.h"
#include "icnt_wrapper.h"
#include "l2cache.h"
#include "shader.h"
#include "stat-tool.h"

#include "../../libcuda/gpgpu_context.h"
#include "../abstract_hardware_model.h"
#include "../cuda-sim/cuda-sim.h"
#include "../cuda-sim/cuda_device_runtime.h"
#include "../cuda-sim/ptx-stats.h"
#include "../cuda-sim/ptx_ir.h"
#include "../debug.h"
#include "../gpgpusim_entrypoint.h"
#include "../statwrapper.h"
#include "../trace.h"
#include "mem_latency_stat.h"
#include "stats.h"
#include "visualizer.h"

#include "../constants.h" // MOD. Added to do not duplicate some constant declarations in many files

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <string>

bool g_interactive_debugger_enabled = false;

tr1_hash_map<new_addr_type, unsigned> address_random_interleaving;

/* Clock Domains */

#define CORE 0x01
#define L2 0x02
#define DRAM 0x04
#define ICNT 0x08

#define MEM_LATENCY_STAT_IMPL

#include "mem_latency_stat.h"

// MOD. Begin. Improved tracer
void gpgpu_sim::parse_extra_trace_info(std::string filepath, bool is_extra_trace_enabled) {
  if(is_extra_trace_enabled) {
    m_extra_trace_info.DeserializeFromFile(filepath.c_str());
  }
}
// MOD. End. Improved tracer

/////////////////////////////////////////////////////////////////////////////

void increment_x_then_y_then_z(dim3 &i, const dim3 &bound) {
  i.x++;
  if (i.x >= bound.x) {
    i.x = 0;
    i.y++;
    if (i.y >= bound.y) {
      i.y = 0;
      if (i.z < bound.z) i.z++;
    }
  }
}

void gpgpu_sim::launch(kernel_info_t *kinfo) {
  m_kernel_scheduler->launch(kinfo);
}

bool gpgpu_sim::can_start_kernel() {
  return m_kernel_scheduler->can_start_kernel();
}

bool gpgpu_sim::hit_max_cta_count() const {
  if (m_config.gpu_max_cta_opt != 0) {
    if ((gpu_tot_issued_cta + m_total_cta_launched) >= m_config.gpu_max_cta_opt)
      return true;
  }
  return false;
}

bool gpgpu_sim::kernel_more_cta_left(kernel_info_t *kernel) const {
  if (hit_max_cta_count()) return false;

  if (kernel && !kernel->no_more_ctas_to_run()) return true;

  return false;
}

bool gpgpu_sim::get_more_cta_left() const {
  if (hit_max_cta_count()) return false;

  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    if (m_running_kernels[n] && !m_running_kernels[n]->no_more_ctas_to_run())
      return true;
  }
  return false;
}

void gpgpu_sim::decrement_kernel_latency() {
  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    if (m_running_kernels[n] && m_running_kernels[n]->m_kernel_TB_latency)
      m_running_kernels[n]->m_kernel_TB_latency--;
  }
}

kernel_info_t *gpgpu_sim::select_kernel() {
  return m_kernel_scheduler->select_kernel();
}

unsigned gpgpu_sim::finished_kernel() {
  if (m_finished_kernel.empty()) return 0;
  unsigned result = m_finished_kernel.front();
  m_finished_kernel.pop_front();
  return result;
}

void gpgpu_sim::set_kernel_done(kernel_info_t *kernel) {
  m_kernel_scheduler->set_kernel_done(kernel);
}

void gpgpu_sim::stop_all_running_kernels() {
  std::vector<kernel_info_t *>::iterator k;
  for (k = m_running_kernels.begin(); k != m_running_kernels.end(); ++k) {
    if (*k != NULL) {       // If a kernel is active
      set_kernel_done(*k);  // Stop the kernel
      assert(*k == NULL);
    }
  }
}

gpgpu_sim::gpgpu_sim(const gpgpu_sim_config &config, gpgpu_context *ctx)
    : gpgpu_t(config, ctx), m_gpu_per_sm_stats("GPU_per_SM_stats"),
    m_coalescing_stats_across_sms_l1d("l1d", _memory_space_t::global_space),
    m_coalescing_stats_across_sms_const("const", _memory_space_t::const_space),
    m_coalescing_stats_across_sms_sharedmem("sharedMem", _memory_space_t::shared_space),
    m_config(config)  {
  gpgpu_ctx = ctx;
  m_shader_config = &m_config.m_shader_config;
  m_memory_config = &m_config.m_memory_config;
  ctx->ptx_parser->set_ptx_warp_size(m_shader_config);
  ptx_file_line_stats_create_exposed_latency_tracker(m_config.num_shader());

  m_shader_stats = new shader_core_stats(m_shader_config, this);
  m_kernel_scheduler = std::make_unique<kernel_scheduler>(*this);
  m_icnt_handler = std::make_unique<icnt_handler>(*this);

  m_memory_stats = new memory_stats_t(m_config.num_shader(), m_shader_config,
                                      m_memory_config, this);
  average_pipeline_duty_cycle = (float *)malloc(sizeof(float));
  active_sms = (float *)malloc(sizeof(float));
  total_sms_accumulated_across_cycles = 0;

  gpu_sim_insn = 0;
  gpu_tot_sim_insn = 0;
  gpu_tot_issued_cta = 0;
  gpu_completed_cta = 0;
  m_total_cta_launched = 0;
  gpu_deadlock = false;

  gpu_stall_dramfull = 0;
  gpu_stall_icnt2sh = 0;
  partiton_reqs_in_parallel = 0;
  partiton_reqs_in_parallel_total = 0;
  partiton_reqs_in_parallel_util = 0;
  partiton_reqs_in_parallel_util_total = 0;
  gpu_sim_cycle_parition_util = 0;
  gpu_tot_sim_cycle_parition_util = 0;
  partiton_replys_in_parallel = 0;
  partiton_replys_in_parallel_total = 0;

  m_memory_partition_unit =
      new memory_partition_unit *[m_memory_config->m_n_mem];
  m_memory_sub_partition =
      new memory_sub_partition *[m_memory_config->m_n_mem_sub_partition];
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
    memory_stats_t *subpartition_mem_stats = new memory_stats_t(m_config.num_shader(), m_shader_config,
                                      m_memory_config, this);
    m_memory_partition_unit[i] =
        new memory_partition_unit(i, m_memory_config, subpartition_mem_stats, this);
        // new memory_partition_unit(i, m_memory_config, m_memory_stats, this);
    for (unsigned p = 0;
         p < m_memory_config->m_n_sub_partition_per_memory_channel; p++) {
      unsigned submpid =
          i * m_memory_config->m_n_sub_partition_per_memory_channel + p;
      m_memory_sub_partition[submpid] =
          m_memory_partition_unit[i]->get_sub_partition(p);
    }
  }

  icnt_wrapper_init();
  icnt_create(m_shader_config->n_simt_clusters,
              m_memory_config->m_n_mem_sub_partition, 0);

  time_vector_create(NUM_MEM_REQ_STAT);
  fprintf(stdout,
          "GPGPU-Sim uArch: performance model initialization complete.\n");

  m_running_kernels.resize(config.max_concurrent_kernel, NULL);
  m_last_issued_kernel = 0;
  m_last_cluster_issue = m_shader_config->n_simt_clusters -
                         1;  // this causes first launch to use simt cluster 0
  *average_pipeline_duty_cycle = 0;
  *active_sms = 0;

  last_liveness_message_time = 0;

  // Jin: functional simulation for CDP
  m_functional_sim = false;
  m_functional_sim_kernel = NULL;

  create_gpu_per_sm_stats();
}

gpgpu_sim::~gpgpu_sim() {
  delete m_shader_stats;
  delete m_memory_stats;
  for(unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
    delete m_memory_partition_unit[i];
  }
  delete[] m_memory_partition_unit;
  delete[] m_memory_sub_partition;
  for(unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    delete m_cluster[i];
  }
  delete[] m_cluster;
  free(average_pipeline_duty_cycle);
  free(active_sms);
  icnt_delete(0);
  shader_CTA_count_destroy();
  time_vector_destroy();
  ptx_file_line_stats_destroy_exposed_latency_tracker();
}

void gpgpu_sim::create_gpu_per_sm_stats() {
  using remodel::AllowedTypesStats;
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpu_sim_insn", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, true, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_mem_read_local", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_mem_write_local", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_mem_read_global", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_mem_write_global", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_mem_texture", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_mem_const", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_mem_read_inst", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_mem_l2_writeback", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_mem_l1_write_allocate", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_mem_l2_write_allocate", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_mem_grid_barrier", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_mem_tlb_miss_data", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_mem_tlb_miss_inst", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_load_insn", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, " = ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_store_insn", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_shmem_insn", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_sstarr_insn", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_tex_insn", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_const_mem_insn", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_param_mem_insn", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_shmem_bkconflict", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_shmem_port_conflict", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_stall_dispatch_to_subpipeline_mem", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_l1cache_bkconflict", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_l1cache_coalescing_conflicts", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_directly_to_l2_coalescing_conflicts", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_cmem_portconflict", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_cmem_coalescing_conflicts", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_warp_instructions", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("Total_effective_incomplete_warps", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_accesses_l1d_instructions", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_l1d_instructions", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_conflicts_shared_instructions", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_shared_instructions", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_shared_mem_accesses", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_ldst_unit_instructions", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_dp_instructions", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("ctas_completed", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);

  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_times_wb_port_conflict", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_times_wb_evaluated", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_cycles_issue_stage_issuing", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_evals_rf", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_evals_rf_with_conflict", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_cycles_issue_stage_stall_next_stage_not_available", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_cycles_issue_stage_stall_issue_port_busy", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_cycles_issue_stage_stall_no_valid_instruction", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_cycles_issue_stage_stall_no_warps_ready", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_cycles_issue_stage_evaluated", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_register_file_cache_hits", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_register_file_cache_allocations", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_regular_regfile_reads", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_regular_regfile_writes", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_constant_cache_reads", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_uniform_predicate_regfile_writes", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_uniform_predicate_regfile_reads", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_uniform_regfile_writes", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_uniform_regfile_reads", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_predicate_regfile_writes", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_predicate_regfile_reads", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_accesses", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_accesses_coalesced", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_accesses_not_coalesced", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("remodeled_dispatch_half_to_sp", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("remodeled_dispatch_half_to_int", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("remodeled_dispatch_sp_to_sp", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("remodeled_dispatch_sp_to_int", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("remodeled_dispatch_int_to_sp", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("remodeled_dispatch_int_to_int", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("remodeled_dispatch_dp_to_dp", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("remodeled_dispatch_mem_to_mem", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("remodeled_shared_throttle_dp_events", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("remodeled_shared_throttle_dp_cycles", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("remodeled_shared_throttle_mem_events", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("remodeled_shared_throttle_mem_cycles", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  
  for(unsigned int i = 0; i < N_MEM_STAGE_ACCESS_TYPE; i++) {
    for(unsigned int j = 0; j < N_MEM_STAGE_STALL_TYPE; j++) {
      std::string stat_name = "gpgpu_stall_shd_mem[" + mem_stage_access_type_to_string(static_cast<mem_stage_access_type>(i)) + "][" + mem_stage_stall_type_to_string(static_cast<mem_stage_stall_type>(j)) + "]";
      m_gpu_per_sm_stats.add_unsigned_long_long_stat(stat_name, AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
    }
  }

  for(unsigned int i = 0; i < m_shader_config->warp_size + 1; i++) {
    m_gpu_per_sm_stats.add_unsigned_long_long_stat("warp_occ_dist" + std::to_string(i), AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, true, false);
  }

  
}

void gpgpu_sim::gather_gpu_per_sm_stats() {
  for(unsigned int i = 0; i < m_shader_config->n_simt_clusters; i++) {
    for(unsigned int j = 0; j < m_shader_config->n_simt_cores_per_cluster; j++) {
      m_cluster[i]->gather_stats(m_gpu_per_sm_stats, m_coalescing_stats_across_sms_l1d, m_coalescing_stats_across_sms_const, m_coalescing_stats_across_sms_sharedmem);
      m_shader_stats->m_incoming_traffic_stats->join_stats(m_cluster[i]->get_incomming_traffic_stats());
      m_shader_stats->m_outgoing_traffic_stats->join_stats(m_cluster[i]->get_outgoing_traffic_stats());
    }
  }
}

void gpgpu_sim::reset_cycless_access_history() {
  for(unsigned int i = 0; i < m_shader_config->n_simt_clusters; i++) {
    for(unsigned int j = 0; j < m_shader_config->n_simt_cores_per_cluster; j++) {
      m_cluster[i]->reset_cycless_access_history();
    }
  }
}

void gpgpu_sim::gather_gpu_per_sm_single_stat(std::string stat_name) {
  for(unsigned int i = 0; i < m_shader_config->n_simt_clusters; i++) {
    for(unsigned int j = 0; j < m_shader_config->n_simt_cores_per_cluster; j++) {
      m_cluster[i]->gather_single_stat(m_gpu_per_sm_stats, stat_name);
    }
  }
}

void gpgpu_sim::reset_gpu_per_sm_stats() {
  for(auto stat_name : m_gpu_per_sm_stats.m_stats_name) {
    m_gpu_per_sm_stats.reset_stat(stat_name);
  }
}

int gpgpu_sim::shared_mem_size() const {
  return m_shader_config->gpgpu_shmem_size;
}

int gpgpu_sim::shared_mem_per_block() const {
  return m_shader_config->gpgpu_shmem_per_block;
}

int gpgpu_sim::num_registers_per_core() const {
  // MOD. Begin. VPREG
  if(m_shader_config->is_vpreg_enabled) {
    return m_shader_config->vpreg_num_physical_regs_per_sm * 32; // Translate from warp registers to thread registers
  }else {
    return m_shader_config->gpgpu_shader_registers;
  }
  // MOD. End. VPREG
}

int gpgpu_sim::num_registers_per_block() const {
  return m_shader_config->gpgpu_registers_per_block;
}

int gpgpu_sim::wrp_size() const { return m_shader_config->warp_size; }

int gpgpu_sim::shader_clock() const { return m_config.core_freq / 1000; }

int gpgpu_sim::max_cta_per_core() const {
  return m_shader_config->max_cta_per_core;
}

int gpgpu_sim::get_max_cta(const kernel_info_t &k) const {
  return m_shader_config->max_cta(k);
}

void gpgpu_sim::set_prop(cudaDeviceProp *prop) { m_cuda_properties = prop; }

int gpgpu_sim::compute_capability_major() const {
  return m_config.gpgpu_compute_capability_major;
}

int gpgpu_sim::compute_capability_minor() const {
  return m_config.gpgpu_compute_capability_minor;
}

const struct cudaDeviceProp *gpgpu_sim::get_prop() const {
  return m_cuda_properties;
}

enum divergence_support_t gpgpu_sim::simd_model() const {
  return m_shader_config->model;
}

void gpgpu_sim_config::init_clock_domains(void) {
  sscanf(gpgpu_clock_domains, "%lf:%lf:%lf:%lf", &core_freq, &icnt_freq,
         &l2_freq, &dram_freq);
  core_freq = core_freq MhZ;
  icnt_freq = icnt_freq MhZ;
  l2_freq = l2_freq MhZ;
  dram_freq = dram_freq MhZ;
  core_period = 1 / core_freq;
  icnt_period = 1 / icnt_freq;
  dram_period = 1 / dram_freq;
  l2_period = 1 / l2_freq;
  printf("GPGPU-Sim uArch: clock freqs: %lf:%lf:%lf:%lf\n", core_freq,
         icnt_freq, l2_freq, dram_freq);
  printf("GPGPU-Sim uArch: clock periods: %.20lf:%.20lf:%.20lf:%.20lf\n",
         core_period, icnt_period, l2_period, dram_period);
}

void gpgpu_sim::reinit_clock_domains(void) {
  core_time = 0;
  dram_time = 0;
  icnt_time = 0;
  l2_time = 0;
}

bool gpgpu_sim::active() {
  if (m_config.gpu_max_cycle_opt &&
      (gpu_tot_sim_cycle + gpu_sim_cycle) >= m_config.gpu_max_cycle_opt)
    return false;
  if (m_config.gpu_max_insn_opt &&
      (gpu_tot_sim_insn + gpu_sim_insn) >= m_config.gpu_max_insn_opt)
    return false;
  if (m_config.gpu_max_cta_opt &&
      (gpu_tot_issued_cta >= m_config.gpu_max_cta_opt))
    return false;
  if (m_config.gpu_max_completed_cta_opt &&
      (gpu_completed_cta >= m_config.gpu_max_completed_cta_opt))
    return false;
  if (m_config.gpu_deadlock_detect && gpu_deadlock) return false;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    if (m_cluster[i]->get_not_completed() > 0) return true;
  ;
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
    if (m_memory_partition_unit[i]->busy() > 0) return true;
  ;
  if (icnt_busy(0)) return true;
  if (get_more_cta_left()) return true;
  return false;
}

void gpgpu_sim::init() {
  // run a CUDA grid on the GPU microarchitecture simulator
  gpu_sim_cycle = 0;
  dram_sim_cycle = 0;
  gpu_sim_insn = 0;
  last_gpu_sim_insn = 0;
  m_total_cta_launched = 0;
  gpu_completed_cta = 0;
  partiton_reqs_in_parallel = 0;
  partiton_replys_in_parallel = 0;
  partiton_reqs_in_parallel_util = 0;
  gpu_sim_cycle_parition_util = 0;

  reinit_clock_domains();
  gpgpu_ctx->func_sim->set_param_gpgpu_num_shaders(m_config.num_shader());
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i]->reinit();
  m_shader_stats->new_grid();
  // initialize the control-flow, memory access, memory latency logger
  if (m_config.g_visualizer_enabled) {
    create_thread_CFlogger(gpgpu_ctx, m_config.num_shader(),
                           m_shader_config->n_thread_per_shader, 0,
                           m_config.gpgpu_cflog_interval);
  }
  shader_CTA_count_create(m_config.num_shader(), m_config.gpgpu_cflog_interval);
  if (m_config.gpgpu_cflog_interval != 0) {
    insn_warp_occ_create(m_config.num_shader(), m_shader_config->warp_size);
    shader_warp_occ_create(m_config.num_shader(), m_shader_config->warp_size,
                           m_config.gpgpu_cflog_interval);
    shader_mem_acc_create(m_config.num_shader(), m_memory_config->m_n_mem, 4,
                          m_config.gpgpu_cflog_interval);
    shader_mem_lat_create(m_config.num_shader(), m_config.gpgpu_cflog_interval);
    shader_cache_access_create(m_config.num_shader(), 3,
                               m_config.gpgpu_cflog_interval);
    set_spill_interval(m_config.gpgpu_cflog_interval * 40);
  }

  icnt_init(0);
}

void gpgpu_sim::update_stats() {
  m_memory_stats->memlatstat_lat_pw();
  gpu_tot_sim_cycle += gpu_sim_cycle;
  gpu_tot_sim_insn += gpu_sim_insn;
  gpu_tot_issued_cta += m_total_cta_launched;
  partiton_reqs_in_parallel_total += partiton_reqs_in_parallel;
  partiton_replys_in_parallel_total += partiton_replys_in_parallel;
  partiton_reqs_in_parallel_util_total += partiton_reqs_in_parallel_util;
  gpu_tot_sim_cycle_parition_util += gpu_sim_cycle_parition_util;
  gpu_tot_occupancy += gpu_occupancy;

  gpu_sim_cycle = 0;
  dram_sim_cycle = 0;
  partiton_reqs_in_parallel = 0;
  partiton_replys_in_parallel = 0;
  partiton_reqs_in_parallel_util = 0;
  gpu_sim_cycle_parition_util = 0;
  gpu_sim_insn = 0;
  m_total_cta_launched = 0;
  gpu_completed_cta = 0;
  gpu_occupancy = occupancy_stats();
}

PowerscalingCoefficients *gpgpu_sim::get_scaling_coeffs()
{
  // AccelWattch power model removed; scaling coefficients are unavailable.
  return NULL;
}

void gpgpu_sim::print_stats() {
  m_shader_stats->gpu_cycles_per_kernel[m_shader_stats->m_current_kernel_pos]=gpu_sim_cycle; // MOD. Custom Stats

  gpgpu_ctx->stats->ptx_file_line_stats_write_file();
  gpu_print_stat();

  printf(
      "----------------------------Interconnect-DETAILS----------------------"
      "----------\n");
  icnt_display_stats(0);
  icnt_display_overall_stats(0);
  printf(
      "----------------------------END-of-Interconnect-DETAILS---------------"
      "----------\n");
}

void gpgpu_sim::deadlock_check() {
  if (m_config.gpu_deadlock_detect && gpu_deadlock) {
    fflush(stdout);
    printf(
        "\n\nGPGPU-Sim uArch: ERROR ** deadlock detected: last writeback core "
        "%u @ gpu_sim_cycle %u (+ gpu_tot_sim_cycle %u) (%u cycles ago)\n",
        gpu_sim_insn_last_update_sid, (unsigned)gpu_sim_insn_last_update,
        (unsigned)(gpu_tot_sim_cycle - gpu_sim_cycle),
        (unsigned)(gpu_sim_cycle - gpu_sim_insn_last_update));
    unsigned num_cores = 0;
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      unsigned not_completed = m_cluster[i]->get_not_completed();
      if (not_completed) {
        if (!num_cores) {
          printf(
              "GPGPU-Sim uArch: DEADLOCK  shader cores no longer committing "
              "instructions [core(# threads)]:\n");
          printf("GPGPU-Sim uArch: DEADLOCK  ");
          m_cluster[i]->print_not_completed(stdout);
        } else if (num_cores < 8) {
          m_cluster[i]->print_not_completed(stdout);
        } else if (num_cores >= 8) {
          printf(" + others ... ");
        }
        num_cores += m_shader_config->n_simt_cores_per_cluster;
      }
    }
    printf("\n");
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      bool busy = m_memory_partition_unit[i]->busy();
      if (busy)
        printf("GPGPU-Sim uArch DEADLOCK:  memory partition %u busy\n", i);
    }
    if (icnt_busy(0)) {
      printf("GPGPU-Sim uArch DEADLOCK:  iterconnect contains traffic\n");
      icnt_display_state(stdout, 0);
    }
    printf(
        "\nRe-run the simulator in gdb and use debug routines in .gdbinit to "
        "debug this\n");
    fflush(stdout);
    abort();
  }
}

/// printing the names and uids of a set of executed kernels (usually there is
/// only one)
std::string gpgpu_sim::executed_kernel_info_string() {
  std::stringstream statout;

  statout << "kernel_name = ";
  for (unsigned int k = 0; k < m_executed_kernel_names.size(); k++) {
    statout << m_executed_kernel_names[k] << " ";
  }
  statout << std::endl;
  statout << "kernel_launch_uid = ";
  for (unsigned int k = 0; k < m_executed_kernel_uids.size(); k++) {
    statout << m_executed_kernel_uids[k] << " ";
  }
  statout << std::endl;

  return statout.str();
}

std::string gpgpu_sim::executed_kernel_name() {
  std::stringstream statout;  
  if( m_executed_kernel_names.size() == 1)
     statout << m_executed_kernel_names[0];
  else{
    for (unsigned int k = 0; k < m_executed_kernel_names.size(); k++) {
      statout << m_executed_kernel_names[k] << " ";
    }
  }
  return statout.str();
}
void gpgpu_sim::set_cache_config(std::string kernel_name,
                                 FuncCache cacheConfig) {
  m_special_cache_config[kernel_name] = cacheConfig;
}

FuncCache gpgpu_sim::get_cache_config(std::string kernel_name) {
  for (std::map<std::string, FuncCache>::iterator iter =
           m_special_cache_config.begin();
       iter != m_special_cache_config.end(); iter++) {
    std::string kernel = iter->first;
    if (kernel_name.compare(kernel) == 0) {
      return iter->second;
    }
  }
  return (FuncCache)0;
}

bool gpgpu_sim::has_special_cache_config(std::string kernel_name) {
  for (std::map<std::string, FuncCache>::iterator iter =
           m_special_cache_config.begin();
       iter != m_special_cache_config.end(); iter++) {
    std::string kernel = iter->first;
    if (kernel_name.compare(kernel) == 0) {
      return true;
    }
  }
  return false;
}

void gpgpu_sim::set_cache_config(std::string kernel_name) {
  if (has_special_cache_config(kernel_name)) {
    change_cache_config(get_cache_config(kernel_name));
  } else {
    change_cache_config(FuncCachePreferNone);
  }
}

void gpgpu_sim::change_cache_config(FuncCache cache_config) {
  if (cache_config != m_shader_config->m_L1D_config.get_cache_status()) {
    printf("FLUSH L1 Cache at configuration change between kernels\n");
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      m_cluster[i]->cache_invalidate();
    }
  }

  switch (cache_config) {
    case FuncCachePreferNone:
      m_shader_config->m_L1D_config.init(
          m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
      m_shader_config->gpgpu_shmem_size =
          m_shader_config->gpgpu_shmem_sizeDefault;
      break;
    case FuncCachePreferL1:
      if ((m_shader_config->m_L1D_config.m_config_stringPrefL1 == NULL) ||
          (m_shader_config->gpgpu_shmem_sizePrefL1 == (unsigned)-1)) {
        printf("WARNING: missing Preferred L1 configuration\n");
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizeDefault;

      } else {
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_stringPrefL1,
            FuncCachePreferL1);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizePrefL1;
      }
      break;
    case FuncCachePreferShared:
      if ((m_shader_config->m_L1D_config.m_config_stringPrefShared == NULL) ||
          (m_shader_config->gpgpu_shmem_sizePrefShared == (unsigned)-1)) {
        printf("WARNING: missing Preferred L1 configuration\n");
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizeDefault;
      } else {
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_stringPrefShared,
            FuncCachePreferShared);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizePrefShared;
      }
      break;
    default:
      break;
  }
}

void gpgpu_sim::clear_executed_kernel_info() {
  m_executed_kernel_names.clear();
  m_executed_kernel_uids.clear();
}
void gpgpu_sim::gpu_print_stat() {
  FILE *statfout = stdout;

  std::string kernel_info_str = executed_kernel_info_string();
  gather_gpu_per_sm_stats();
  reset_cycless_access_history();
  
  gpu_sim_insn = m_gpu_per_sm_stats.m_stats_map["gpu_sim_insn"]->get_value();
  
  fprintf(statfout, "%s", kernel_info_str.c_str());

  printf("gpu_sim_cycle = %lld\n", gpu_sim_cycle);
  m_gpu_per_sm_stats.m_stats_map["gpu_sim_insn"]->print(statfout);
  printf("gpu_ipc = %12.4f\n", (float)gpu_sim_insn / gpu_sim_cycle);
  printf("gpu_tot_sim_cycle = %lld\n", gpu_tot_sim_cycle + gpu_sim_cycle);
  printf("gpu_tot_sim_insn = %lld\n", gpu_tot_sim_insn + gpu_sim_insn);
  printf("gpu_tot_ipc = %12.4f\n", (float)(gpu_tot_sim_insn + gpu_sim_insn) /
                                       (gpu_tot_sim_cycle + gpu_sim_cycle));
  const char *remodeled_semantic_stats[] = {
      "remodeled_dispatch_half_to_sp",
      "remodeled_dispatch_half_to_int",
      "remodeled_dispatch_sp_to_sp",
      "remodeled_dispatch_sp_to_int",
      "remodeled_dispatch_int_to_sp",
      "remodeled_dispatch_int_to_int",
      "remodeled_dispatch_dp_to_dp",
      "remodeled_dispatch_mem_to_mem",
      "remodeled_shared_throttle_dp_events",
      "remodeled_shared_throttle_dp_cycles",
      "remodeled_shared_throttle_mem_events",
      "remodeled_shared_throttle_mem_cycles",
  };
  for (const char *stat_name : remodeled_semantic_stats) {
    m_gpu_per_sm_stats.m_stats_map[stat_name]->print(statfout);
  }
  printf("gpu_tot_issued_cta = %lld\n",
         gpu_tot_issued_cta + m_total_cta_launched);
  printf("gpu_occupancy = %.4f%% \n", gpu_occupancy.get_occ_fraction() * 100);
  printf("gpu_tot_occupancy = %.4f%% \n",
         (gpu_occupancy + gpu_tot_occupancy).get_occ_fraction() * 100);
  printf("gpu_tot_sms_occupancy = %.4f%%\n", ((*active_sms )/ total_sms_accumulated_across_cycles)*100);
  fprintf(statfout, "max_total_param_size = %llu\n",
          gpgpu_ctx->device_runtime->g_max_total_param_size);

  // performance counter for stalls due to congestion.
  printf("gpu_stall_dramfull = %d\n", gpu_stall_dramfull);
  printf("gpu_stall_icnt2sh    = %d\n", gpu_stall_icnt2sh);

  // printf("partiton_reqs_in_parallel = %lld\n", partiton_reqs_in_parallel);
  // printf("partiton_reqs_in_parallel_total    = %lld\n",
  // partiton_reqs_in_parallel_total );
  printf("partiton_level_parallism = %12.4f\n",
         (float)partiton_reqs_in_parallel / gpu_sim_cycle);
  printf("partiton_level_parallism_total  = %12.4f\n",
         (float)(partiton_reqs_in_parallel + partiton_reqs_in_parallel_total) /
             (gpu_tot_sim_cycle + gpu_sim_cycle));
  // printf("partiton_reqs_in_parallel_util = %lld\n",
  // partiton_reqs_in_parallel_util);
  // printf("partiton_reqs_in_parallel_util_total    = %lld\n",
  // partiton_reqs_in_parallel_util_total ); printf("gpu_sim_cycle_parition_util
  // = %lld\n", gpu_sim_cycle_parition_util);
  // printf("gpu_tot_sim_cycle_parition_util    = %lld\n",
  // gpu_tot_sim_cycle_parition_util );
  printf("partiton_level_parallism_util = %12.4f\n",
         (float)partiton_reqs_in_parallel_util / gpu_sim_cycle_parition_util);
  printf("partiton_level_parallism_util_total  = %12.4f\n",
         (float)(partiton_reqs_in_parallel_util +
                 partiton_reqs_in_parallel_util_total) /
             (gpu_sim_cycle_parition_util + gpu_tot_sim_cycle_parition_util));
  // printf("partiton_replys_in_parallel = %lld\n",
  // partiton_replys_in_parallel); printf("partiton_replys_in_parallel_total =
  // %lld\n", partiton_replys_in_parallel_total );
  printf("L2_BW  = %12.4f GB/Sec\n",
         ((float)(partiton_replys_in_parallel * 32) /
          (gpu_sim_cycle * m_config.icnt_period)) /
             1000000000);
  printf("L2_BW_total  = %12.4f GB/Sec\n",
         ((float)((partiton_replys_in_parallel +
                   partiton_replys_in_parallel_total) *
                  32) /
          ((gpu_tot_sim_cycle + gpu_sim_cycle) * m_config.icnt_period)) /
             1000000000);

  time_t curr_time;
  time(&curr_time);
  unsigned long long elapsed_time =
      MAX(curr_time - gpgpu_ctx->the_gpgpusim->g_simulation_starttime, 1);
  printf("gpu_total_sim_rate=%u\n",
         (unsigned)((gpu_tot_sim_insn + gpu_sim_insn) / elapsed_time));

  // shader_print_l1_miss_stat( stdout );
  shader_print_cache_stats(stdout);

  cache_stats core_cache_stats;
  core_cache_stats.clear();
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    m_cluster[i]->get_cache_stats(core_cache_stats);
  }
  core_cache_stats.compute_total_write_and_read_accesses();
  double total_cache_accesses = core_cache_stats.get_total_write_and_read_accesses();
  double l1d_bw = ((total_cache_accesses * 32) / ((gpu_tot_sim_cycle + gpu_sim_cycle) * m_config.core_period)) / 1000000000;
  printf("L1D_BW_total  = %12.4lf GB/Sec\n", l1d_bw);
  printf("\nTotal_core_cache_stats:\n");
  core_cache_stats.print_stats(stdout, "Total_core_cache_stats_breakdown");
  printf("\nTotal_core_cache_fail_stats:\n");
  core_cache_stats.print_fail_stats(stdout,
                                    "Total_core_cache_fail_stats_breakdown");
  shader_print_scheduler_stat(stdout, false);

  m_shader_stats->compute_derived_custom_stats(); // MOD. Custom Stats
  m_shader_stats->compute_ibuffer_ooo_stats(); // MOD. IBuffer_ooo custom Stats
  m_shader_stats->print(stdout);

  dram_tot_sim_cycle += dram_sim_cycle;
  // performance counter that are not local to one shader
  for(unsigned int i = 0; i < m_memory_config->m_n_mem; i++) {
    m_memory_stats->add(m_memory_partition_unit[i]->get_memory_partition_stats());
    m_memory_partition_unit[i]->get_memory_partition_stats()->reset();
  }
  m_memory_stats->memlatstat_print(m_memory_config->m_n_mem,
                                   m_memory_config->nbk);
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
    m_memory_partition_unit[i]->print(stdout);

  // L2 cache stats
  if (!m_memory_config->m_L2_config.disabled()) {
    cache_stats l2_stats;
    struct cache_sub_stats l2_css;
    struct cache_sub_stats total_l2_css;
    l2_stats.clear();
    l2_css.clear();
    total_l2_css.clear();

    printf("\n========= L2 cache stats =========\n");
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      m_memory_sub_partition[i]->accumulate_L2cache_stats(l2_stats);
      m_memory_sub_partition[i]->get_L2cache_sub_stats(l2_css);

      fprintf(stdout,
              "L2_cache_bank[%d]: Access = %llu, Miss = %llu, Miss_rate = "
              "%.3lf, Pending_hits = %llu, Reservation_fails = %llu\n",
              i, l2_css.accesses, l2_css.misses,
              (double)l2_css.misses / (double)l2_css.accesses,
              l2_css.pending_hits, l2_css.res_fails);

      total_l2_css += l2_css;
    }
    if (!m_memory_config->m_L2_config.disabled() &&
        m_memory_config->m_L2_config.get_num_lines()) {
      // L2c_print_cache_stat();
      printf("L2_total_cache_accesses = %llu\n", total_l2_css.accesses);
      printf("L2_total_cache_misses = %llu\n", total_l2_css.misses);
      if (total_l2_css.accesses > 0)
        printf("L2_total_cache_miss_rate = %.4lf\n",
               (double)total_l2_css.misses / (double)total_l2_css.accesses);
      printf("L2_total_cache_pending_hits = %llu\n", total_l2_css.pending_hits);
      printf("L2_total_cache_reservation_fails = %llu\n",
             total_l2_css.res_fails);
      printf("L2_total_cache_breakdown:\n");
      l2_stats.print_stats(stdout, "L2_cache_stats_breakdown");
      printf("L2_total_cache_reservation_fail_breakdown:\n");
      l2_stats.print_fail_stats(stdout, "L2_cache_stats_fail_breakdown");
      total_l2_css.print_port_stats(stdout, "L2_cache");
    }
  }

  if (m_config.gpgpu_cflog_interval != 0) {
    spill_log_to_file(stdout, 1, gpu_sim_cycle);
    insn_warp_occ_print(stdout);
  }
  if (gpgpu_ctx->func_sim->gpgpu_ptx_instruction_classification) {
    StatDisp(gpgpu_ctx->func_sim->g_inst_classification_stat
                 [gpgpu_ctx->func_sim->g_ptx_kernel_count]);
    StatDisp(gpgpu_ctx->func_sim->g_inst_op_classification_stat
                 [gpgpu_ctx->func_sim->g_ptx_kernel_count]);
  }

  // Interconnect power stat print
  long total_simt_to_mem = 0;
  long total_mem_to_simt = 0;
  long temp_stm = 0;
  long temp_mts = 0;
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    m_cluster[i]->get_icnt_stats(temp_stm, temp_mts);
    total_simt_to_mem += temp_stm;
    total_mem_to_simt += temp_mts;
  }
  printf("\nicnt_total_pkts_mem_to_simt=%ld\n", total_mem_to_simt);
  printf("icnt_total_pkts_simt_to_mem=%ld\n", total_simt_to_mem);

  time_vector_print();
  fflush(stdout);

  clear_executed_kernel_info();
  reset_gpu_per_sm_stats();
}

// performance counter that are not local to one shader
unsigned gpgpu_sim::threads_per_core() const {
  return m_shader_config->n_thread_per_shader;
}

///////////////////////////////////////////////////////////////////////////////////////////

void dram_t::dram_log(int task) {
  if (task == SAMPLELOG) {
    StatAddSample(mrqq_Dist, que_length());
  } else if (task == DUMPLOG) {
    printf("Queue Length DRAM[%d] ", id);
    StatDisp(mrqq_Dist);
  }
}

// Find next clock domain and increment its time
int gpgpu_sim::next_clock_domain(void) {
  double smallest = min3(core_time, icnt_time, dram_time);
  int mask = 0x00;
  if (l2_time <= smallest) {
    smallest = l2_time;
    mask |= L2;
    l2_time += m_config.l2_period;
  }
  if (icnt_time <= smallest) {
    mask |= ICNT;
    icnt_time += m_config.icnt_period;
  }
  if (dram_time <= smallest) {
    mask |= DRAM;
    dram_time += m_config.dram_period;
  }
  if (core_time <= smallest) {
    mask |= CORE;
    core_time += m_config.core_period;
  }
  return mask;
}

void gpgpu_sim::issue_block2core() {
  unsigned last_issued = m_last_cluster_issue;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    unsigned idx = (i + last_issued + 1) % m_shader_config->n_simt_clusters;
    unsigned num = m_cluster[idx]->issue_block2core();
    if (num) {
      m_last_cluster_issue = idx;
      m_total_cta_launched += num;
    }
  }
}

unsigned long long g_single_step =
    0;  // set this in gdb to single step the pipeline



std::unique_ptr<grid_barrier_notify_info> gpgpu_sim::register_grid_barrier_arrivement(mem_fetch *mf) {
  return m_icnt_handler->register_grid_barrier_arrivement(mf);
}

void gpgpu_sim::increase_num_threads_kernel(unsigned kernel_id, unsigned num_threads) {
  assert(m_grid_barrier_status.find(kernel_id) != m_grid_barrier_status.end());
  m_grid_barrier_status[kernel_id].num_threads_kernel += num_threads;
}

void gpgpu_sim::decrease_num_threads_kernel(unsigned kernel_id, unsigned num_threads) {
  assert(m_grid_barrier_status.find(kernel_id) != m_grid_barrier_status.end());
  assert(m_grid_barrier_status[kernel_id].num_threads_kernel >= num_threads);
  m_grid_barrier_status[kernel_id].num_threads_kernel -= num_threads;
}

void gpgpu_sim::core_front_tick() {
  // shader core loading (pop from ICNT into core) follows CORE clock
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    m_cluster[i]->icnt_cycle();
  }
}

void gpgpu_sim::icnt_tick(unsigned &partiton_replys_in_parallel_per_cycle) {
  m_icnt_handler->icnt_tick(partiton_replys_in_parallel_per_cycle);
}

void gpgpu_sim::dram_tick() {
#pragma omp parallel for
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
    if (m_memory_config->simple_dram_model)
      m_memory_partition_unit[i]->simple_dram_model_cycle();
    else
      m_memory_partition_unit[i]
          ->dram_cycle();  // Issue the dram command (scheduler + delay model)
  }
  dram_sim_cycle++;
}

void gpgpu_sim::l2_tick(unsigned &partiton_reqs_in_parallel_per_cycle) {
  for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
    // move memory request from interconnect into memory partition (if not
    // backed up) Note:This needs to be called in DRAM clock domain if there
    // is no L2 cache in the system In the worst case, we may need to push
    // SECTOR_CHUNCK_SIZE requests, so ensure you have enough buffer for them
    if (m_memory_sub_partition[i]->full(SECTOR_CHUNCK_SIZE)) {
      gpu_stall_dramfull++;
    } else {
      mem_fetch *mf = (mem_fetch *)icnt_pop(m_shader_config->mem2device(i), 0);
      if (mf) {
        if (mf->get_inst().op == GRID_BARRIER_OP) {
          std::unique_ptr<grid_barrier_notify_info> notifcation_res =
              register_grid_barrier_arrivement(mf);
          mf = nullptr;
          if (notifcation_res) {
            m_grid_barrier_notify_queue.push(std::move(notifcation_res));
          }
        } else {
          partiton_reqs_in_parallel_per_cycle++;
        }
      }
      m_memory_sub_partition[i]->push(mf,
                                      gpu_sim_cycle + gpu_tot_sim_cycle);
    }
    m_memory_sub_partition[i]->cache_cycle(gpu_sim_cycle + gpu_tot_sim_cycle);
  }
}

void gpgpu_sim::icnt_transfer_tick() {
  m_icnt_handler->icnt_transfer_tick();
}

void gpgpu_sim::core_tick() {
  // L1 cache + shader core pipeline stages
#pragma omp parallel for schedule(runtime) reduction(+:m_active_sms_this_cycle)
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    if (m_cluster[i]->get_not_completed() || get_more_cta_left()) {
      m_cluster[i]->core_cycle();
    }
    m_active_sms_this_cycle += m_cluster[i]->get_n_active_sms();
  }
  float temp = 0;
  float previous_active_sms = *active_sms;
  for (unsigned i = 0; i < m_shader_config->num_shader(); i++) {
    m_cluster[i]->get_current_occupancy(
        gpu_occupancy.aggregate_warp_slot_filled,
        gpu_occupancy.aggregate_theoretical_warp_slots);
    temp += m_shader_stats->m_pipeline_duty_cycle[i];
    // active_sms_this_cycle += m_cluster[i]->get_n_active_sms();
  }
  *active_sms += m_active_sms_this_cycle;
  if (previous_active_sms != *active_sms) {
    total_sms_accumulated_across_cycles += m_shader_config->num_shader();
  }

  if (m_shader_config->is_custom_omp_scheduler_enabled) {
    omp_sched_t next_omp_scheduler = omp_sched_static;
    bool is_gpu_action = (m_active_sms_this_cycle > 0);
    float ratio_active =
        m_active_sms_this_cycle / m_shader_config->num_shader();
    if ((ratio_active < m_shader_config->custom_omp_scheduler_ratio_to_dynamic) &&
        is_gpu_action) {
      next_omp_scheduler = omp_sched_dynamic;
    }
    if ((m_current_omp_scheduler != next_omp_scheduler) && is_gpu_action) {
      m_current_omp_scheduler = next_omp_scheduler;
      omp_set_schedule(next_omp_scheduler, 1);
    }
  }

  temp = temp / m_shader_config->num_shader();
  *average_pipeline_duty_cycle = ((*average_pipeline_duty_cycle) + temp);
  // cout<<"Average pipeline duty cycle:
  // "<<*average_pipeline_duty_cycle<<endl;

  if (g_single_step && ((gpu_sim_cycle + gpu_tot_sim_cycle) >= g_single_step)) {
    raise(SIGTRAP);  // Debug breakpoint
  }
  gpu_sim_cycle++;

  if (g_interactive_debugger_enabled) {
    gpgpu_debug();
  }

  issue_block2core();
  decrement_kernel_latency();

  // Depending on configuration, invalidate the caches once all of threads are
  // completed.
  int all_threads_complete = 1;
  if (m_config.gpgpu_flush_l1_cache) {
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      if (m_cluster[i]->get_not_completed() == 0)
        m_cluster[i]->cache_invalidate();
      else
        all_threads_complete = 0;
    }
  }

  if (m_config.gpgpu_flush_l2_cache) {
    if (!m_config.gpgpu_flush_l1_cache) {
      for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
        if (m_cluster[i]->get_not_completed() != 0) {
          all_threads_complete = 0;
          break;
        }
      }
    }

    if (all_threads_complete && !m_memory_config->m_L2_config.disabled()) {
      // printf("Flushed L2 caches...\n");
      if (m_memory_config->m_L2_config.get_num_lines()) {
        int dlc = 0;
        for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
          // dlc = m_memory_sub_partition[i]->flushL2();
          dlc = m_memory_sub_partition[i]->invalidateL2();
          assert(dlc ==
                 0);  // TODO: need to model actual writes to DRAM here
          // printf("Dirty lines flushed from L2 %d is %d\n", i, dlc);
        }
      }
    }
  }

  if (!(gpu_sim_cycle % m_config.gpu_stat_sample_freq)) {
    time_t days, hrs, minutes, sec;
    time_t curr_time;
    time(&curr_time);
    unsigned long long elapsed_time =
        MAX(curr_time - gpgpu_ctx->the_gpgpusim->g_simulation_starttime, 1);
    if ((elapsed_time - last_liveness_message_time) >=
            m_config.liveness_message_freq &&
        DTRACE(LIVENESS)) {
      days = elapsed_time / (3600 * 24);
      hrs = elapsed_time / 3600 - 24 * days;
      minutes = elapsed_time / 60 - 60 * (hrs + 24 * days);
      sec = elapsed_time - 60 * (minutes + 60 * (hrs + 24 * days));

      unsigned long long active = 0, total = 0;
      for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
        m_cluster[i]->get_current_occupancy(active, total);
      }
      DPRINTFG(LIVENESS,
               "uArch: inst.: %lld (ipc=%4.1f, occ=%0.4f%% [%llu / %llu]) "
               "sim_rate=%u (inst/sec) elapsed = %u:%u:%02u:%02u / %s",
               gpu_tot_sim_insn + gpu_sim_insn,
               (double)gpu_sim_insn / (double)gpu_sim_cycle,
               float(active) / float(total) * 100, active, total,
               (unsigned)((gpu_tot_sim_insn + gpu_sim_insn) / elapsed_time),
               (unsigned)days, (unsigned)hrs, (unsigned)minutes,
               (unsigned)sec, ctime(&curr_time));
      fflush(stdout);
      last_liveness_message_time = elapsed_time;
    }
    visualizer_printstat();
    m_memory_stats->memlatstat_lat_pw();
    if (m_config.gpgpu_runtime_stat && (m_config.gpu_runtime_stat_flag != 0)) {
      if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_BW_STAT) {
        for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
          m_memory_partition_unit[i]->print_stat(stdout);
        printf("maxmrqlatency = %d \n", m_memory_stats->max_mrq_latency);
        printf("maxmflatency = %d \n", m_memory_stats->max_mf_latency);
      }
      if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SHD_INFO)
        shader_print_runtime_stat(stdout);
      if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_L1MISS)
        shader_print_l1_miss_stat(stdout);
      if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SCHED)
        shader_print_scheduler_stat(stdout, false);
    }
  }

  if (!(gpu_sim_cycle % 100000)) {
    // deadlock detection
    gather_gpu_per_sm_single_stat("gpu_sim_insn");
    gpu_sim_insn = m_gpu_per_sm_stats.m_stats_map["gpu_sim_insn"]->get_value();
    if (m_config.gpu_deadlock_detect &&
        ((gpu_tot_sim_insn + gpu_sim_insn) == last_gpu_sim_insn)) {
      gpu_deadlock = true;
    } else {
      last_gpu_sim_insn =
          gpu_sim_insn + gpu_tot_sim_insn;  // MOD. More accurate deadlock detection
    }
  }
  try_snap_shot(gpu_sim_cycle);
  spill_log_to_file(stdout, 0, gpu_sim_cycle);

#if (CUDART_VERSION >= 5000)
  // launch device kernel
  gpgpu_ctx->device_runtime->launch_one_device_kernel();
#endif
}

void gpgpu_sim::cycle() {
  m_active_sms_this_cycle = 0;
  m_current_cycle_clock_mask = next_clock_domain();
  if (m_current_cycle_clock_mask & CORE) {
    core_front_tick();
  }
  unsigned partiton_replys_in_parallel_per_cycle = 0;
  if (m_current_cycle_clock_mask & ICNT) {
    icnt_tick(partiton_replys_in_parallel_per_cycle);
  }
  partiton_replys_in_parallel += partiton_replys_in_parallel_per_cycle;

  if (m_current_cycle_clock_mask & DRAM) {
    dram_tick();
  }

  // L2 operations follow L2 clock domain
  unsigned partiton_reqs_in_parallel_per_cycle = 0;
  if (m_current_cycle_clock_mask & L2) {
    l2_tick(partiton_reqs_in_parallel_per_cycle);
  }
  partiton_reqs_in_parallel += partiton_reqs_in_parallel_per_cycle;
  if (partiton_reqs_in_parallel_per_cycle > 0) {
    partiton_reqs_in_parallel_util += partiton_reqs_in_parallel_per_cycle;
    gpu_sim_cycle_parition_util++;
  }

  if (m_current_cycle_clock_mask & ICNT) {
    icnt_transfer_tick();
  }

  if (m_current_cycle_clock_mask & CORE) {
    core_tick();
  }
}


void gpgpu_sim::perf_memcpy_to_gpu(size_t dst_start_addr, size_t count) {
  if (m_memory_config->m_perf_sim_memcpy) {
    // if(!m_config.trace_driven_mode)    //in trace-driven mode, CUDA runtime
    // can start nre data structure at any position 	assert (dst_start_addr %
    // 32
    //== 0);

    for (unsigned counter = 0; counter < count; counter += 32) {
      const unsigned wr_addr = dst_start_addr + counter;
      addrdec_t raw_addr;
      mem_access_sector_mask_t mask;
      mask.set(wr_addr % 128 / 32);
      m_memory_config->m_address_mapping.addrdec_tlx(wr_addr, &raw_addr);
      const unsigned partition_id =
          raw_addr.sub_partition /
          m_memory_config->m_n_sub_partition_per_memory_channel;
      m_memory_partition_unit[partition_id]->handle_memcpy_to_gpu(
          wr_addr, raw_addr.sub_partition, mask);
    }
  }
}

void gpgpu_sim::dump_pipeline(int mask, int s, int m) const {
  /*
     You may want to use this function while running GPGPU-Sim in gdb.
     One way to do that is add the following to your .gdbinit file:

        define dp
           call g_the_gpu.dump_pipeline_impl((0x40|0x4|0x1),$arg0,0)
        end

     Then, typing "dp 3" will show the contents of the pipeline for shader
     core 3.
  */

  printf("Dumping pipeline state...\n");
  if (!mask) mask = 0xFFFFFFFF;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    if (s != -1) {
      i = s;
    }
    if (mask & 1)
      m_cluster[m_shader_config->sid_to_cluster(i)]->display_pipeline(
          i, stdout, 1, mask & 0x2E);
    if (s != -1) {
      break;
    }
  }
  if (mask & 0x10000) {
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      if (m != -1) {
        i = m;
      }
      printf("DRAM / memory controller %u:\n", i);
      if (mask & 0x100000) m_memory_partition_unit[i]->print_stat(stdout);
      if (mask & 0x1000000) m_memory_partition_unit[i]->visualize();
      if (mask & 0x10000000) m_memory_partition_unit[i]->print(stdout);
      if (m != -1) {
        break;
      }
    }
  }
  fflush(stdout);
}

const shader_core_config *gpgpu_sim::getShaderCoreConfig() {
  return m_shader_config;
}

const memory_config *gpgpu_sim::getMemoryConfig() { return m_memory_config; }

simt_core_cluster *gpgpu_sim::getSIMTCluster() { return *m_cluster; }
