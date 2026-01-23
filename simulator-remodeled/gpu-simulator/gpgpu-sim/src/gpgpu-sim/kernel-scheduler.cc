#include "kernel-scheduler.h"

#include "gpu-sim.h"

#include <algorithm>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

kernel_scheduler::kernel_scheduler(gpgpu_sim &gpu) : m_gpu(gpu) {}

void kernel_scheduler::launch(kernel_info_t *kinfo) {
  unsigned cta_size = kinfo->threads_per_cta();
  if (cta_size > m_gpu.m_shader_config->n_thread_per_shader) {
    printf(
        "Execution error: Shader kernel CTA (block) size is too large for "
        "microarch config.\n");
    printf("                 CTA size (x*y*z) = %u, max supported = %u\n",
           cta_size, m_gpu.m_shader_config->n_thread_per_shader);
    printf(
        "                 => either change -gpgpu_shader argument in "
        "gpgpusim.config file or\n");
    printf(
        "                 modify the CUDA source to decrease the kernel block "
        "size.\n");
    abort();
  }
  unsigned n = 0;
  for (n = 0; n < m_gpu.m_running_kernels.size(); n++) {
    if ((NULL == m_gpu.m_running_kernels[n]) ||
        m_gpu.m_running_kernels[n]->done()) {
      m_gpu.m_running_kernels[n] = kinfo;
      break;
    }
  }
  m_gpu.m_shader_stats->allocate_for_a_new_kernel(); // MOD. Custom Stats
  m_gpu.m_shader_stats->num_kernel_not_in_binary +=
      !kinfo->is_captured_from_binary;
  m_gpu.m_grid_barrier_status[kinfo->get_uid()] =
      grid_barrier_status(kinfo->get_uid(), 0);
  assert(n < m_gpu.m_running_kernels.size());
}

bool kernel_scheduler::can_start_kernel() const {
  for (unsigned n = 0; n < m_gpu.m_running_kernels.size(); n++) {
    if ((NULL == m_gpu.m_running_kernels[n]) ||
        m_gpu.m_running_kernels[n]->done())
      return true;
  }
  return false;
}

kernel_info_t *kernel_scheduler::select_kernel() {
  if (m_gpu.m_running_kernels[m_gpu.m_last_issued_kernel] &&
      !m_gpu.m_running_kernels[m_gpu.m_last_issued_kernel]
           ->no_more_ctas_to_run() &&
      !m_gpu.m_running_kernels[m_gpu.m_last_issued_kernel]
           ->m_kernel_TB_latency) {
    unsigned launch_uid =
        m_gpu.m_running_kernels[m_gpu.m_last_issued_kernel]->get_uid();
    if (std::find(m_gpu.m_executed_kernel_uids.begin(),
                  m_gpu.m_executed_kernel_uids.end(),
                  launch_uid) == m_gpu.m_executed_kernel_uids.end()) {
      m_gpu.m_running_kernels[m_gpu.m_last_issued_kernel]->start_cycle =
          m_gpu.gpu_sim_cycle + m_gpu.gpu_tot_sim_cycle;
      m_gpu.m_executed_kernel_uids.push_back(launch_uid);
      m_gpu.m_executed_kernel_names.push_back(
          m_gpu.m_running_kernels[m_gpu.m_last_issued_kernel]->name());
    }
    return m_gpu.m_running_kernels[m_gpu.m_last_issued_kernel];
  }

  for (unsigned n = 0; n < m_gpu.m_running_kernels.size(); n++) {
    unsigned idx = (n + m_gpu.m_last_issued_kernel + 1) %
                   m_gpu.m_config.get_max_concurrent_kernel();
    if (m_gpu.kernel_more_cta_left(m_gpu.m_running_kernels[idx]) &&
        !m_gpu.m_running_kernels[idx]->m_kernel_TB_latency) {
      m_gpu.m_last_issued_kernel = idx;
      m_gpu.m_running_kernels[idx]->start_cycle =
          m_gpu.gpu_sim_cycle + m_gpu.gpu_tot_sim_cycle;
      // record this kernel for stat print if it is the first time this kernel
      // is selected for execution
      unsigned launch_uid = m_gpu.m_running_kernels[idx]->get_uid();
      assert(std::find(m_gpu.m_executed_kernel_uids.begin(),
                       m_gpu.m_executed_kernel_uids.end(),
                       launch_uid) == m_gpu.m_executed_kernel_uids.end());
      m_gpu.m_executed_kernel_uids.push_back(launch_uid);
      m_gpu.m_executed_kernel_names.push_back(
          m_gpu.m_running_kernels[idx]->name());

      return m_gpu.m_running_kernels[idx];
    }
  }
  return NULL;
}

void kernel_scheduler::set_kernel_done(kernel_info_t *kernel) {
  unsigned uid = kernel->get_uid();
  m_gpu.m_grid_barrier_status.erase(uid);
  m_gpu.m_finished_kernel.push_back(uid);
  std::vector<kernel_info_t *>::iterator k;
  for (k = m_gpu.m_running_kernels.begin();
       k != m_gpu.m_running_kernels.end(); k++) {
    if (*k == kernel) {
      kernel->end_cycle = m_gpu.gpu_sim_cycle + m_gpu.gpu_tot_sim_cycle;
      *k = NULL;
      break;
    }
  }
  assert(k != m_gpu.m_running_kernels.end());
}
