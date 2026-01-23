#include "icnt-handler.h"

#include "gpu-sim.h"
#include "icnt_wrapper.h"
#include "mem_fetch.h"

#include <assert.h>

icnt_handler::icnt_handler(gpgpu_sim &gpu) : m_gpu(gpu) {}

std::unique_ptr<grid_barrier_notify_info>
icnt_handler::register_grid_barrier_arrivement(mem_fetch *mf) {
  std::unique_ptr<grid_barrier_notify_info> notifcation_res = nullptr;
  unsigned int kernel_id = mf->get_kernel_id();
  assert(m_gpu.m_grid_barrier_status.find(kernel_id) !=
         m_gpu.m_grid_barrier_status.end());
  if (!m_gpu.m_grid_barrier_status[kernel_id].active) {
    m_gpu.m_grid_barrier_status[kernel_id].active = true;
  }
  m_gpu.m_grid_barrier_status[kernel_id].sm_ids_to_notify.insert(mf->get_sid());
  m_gpu.m_grid_barrier_status[kernel_id].num_threads_arrived +=
      mf->get_inst().active_count();
  if (m_gpu.m_grid_barrier_status[kernel_id].barrier_completed()) {
    m_gpu.m_grid_barrier_status[kernel_id].active = false;
    m_gpu.m_grid_barrier_status[kernel_id].num_threads_arrived = 0;
    notifcation_res = std::make_unique<grid_barrier_notify_info>(
        kernel_id, m_gpu.m_grid_barrier_status[kernel_id].sm_ids_to_notify);
    m_gpu.m_grid_barrier_status[kernel_id].sm_ids_to_notify.clear();
  }
  delete mf;
  return notifcation_res;
}

void icnt_handler::icnt_tick(unsigned &partiton_replys_in_parallel_per_cycle) {
  // pop from grid barrier notify queue
  if (!m_gpu.m_grid_barrier_notify_queue.empty()) {
    auto it_sm_ids_to_notify =
        m_gpu.m_grid_barrier_notify_queue.front()->sm_ids_to_notify.begin();
    while (it_sm_ids_to_notify !=
           m_gpu.m_grid_barrier_notify_queue.front()->sm_ids_to_notify.end()) {
      auto sm_id = *it_sm_ids_to_notify;
      mem_access_t res_acc(m_gpu.gpgpu_ctx);
      res_acc.set_space(miscellaneous_space);
      res_acc.set_write(false);
      res_acc.set_last_access(true);
      res_acc.set_size(32);
      res_acc.set_type(GRID_BARRIER_ACC);
      mem_fetch *mf = new mem_fetch(res_acc, nullptr, 0, 0, sm_id, sm_id,
                                    m_gpu.m_memory_config,
                                    m_gpu.gpu_sim_cycle +
                                        m_gpu.gpu_tot_sim_cycle);
      mf->set_reply();
      mf->set_kernel_id(m_gpu.m_grid_barrier_notify_queue.front()->kernel_id);
      if (::icnt_has_buffer(m_gpu.m_shader_config->mem2device(0), mf->size(),
                            0)) {
        mf->set_return_timestamp(m_gpu.gpu_sim_cycle +
                                 m_gpu.gpu_tot_sim_cycle);
        mf->set_status(IN_ICNT_TO_SHADER,
                       m_gpu.gpu_sim_cycle + m_gpu.gpu_tot_sim_cycle);
        ::icnt_push(m_gpu.m_shader_config->mem2device(0), mf->get_tpc(), mf,
                    mf->size(), 0);
        m_gpu.m_memory_sub_partition[0]->pop();
        it_sm_ids_to_notify =
            m_gpu.m_grid_barrier_notify_queue.front()->sm_ids_to_notify.erase(
                it_sm_ids_to_notify);
      } else {
        m_gpu.gpu_stall_icnt2sh++;
        ++it_sm_ids_to_notify;
      }
    }
    if (m_gpu.m_grid_barrier_notify_queue.front()->sm_ids_to_notify.empty()) {
      m_gpu.m_grid_barrier_notify_queue.pop();
    }
  }
  for (unsigned i = 0; i < m_gpu.m_memory_config->m_n_mem_sub_partition; i++) {
    mem_fetch *mf = m_gpu.m_memory_sub_partition[i]->top();
    if (mf) {
      if (m_gpu.m_shader_config
              ->is_const_cache_accessed_blocks_tracking_enabled &&
          (mf->get_access_type() == CONST_ACC_R)) {
        m_gpu.m_shader_stats->all_const_cache_accessed_blocks.insert(
            mf->get_addr());
      }
      if (m_gpu.m_shader_config->is_num_virtual_pages_tracking_enabled) {
        unsigned int id_page =
            mf->get_addr() /
            m_gpu.m_shader_config->virtual_page_size_in_bytes;
        m_gpu.m_shader_stats->all_virtual_pages_accessed.insert(id_page);
      }
      unsigned response_size =
          mf->get_is_write() ? mf->get_ctrl_size() : mf->size();
      if (::icnt_has_buffer(m_gpu.m_shader_config->mem2device(i),
                            response_size, 0)) {
        mf->set_return_timestamp(m_gpu.gpu_sim_cycle +
                                 m_gpu.gpu_tot_sim_cycle);
        mf->set_status(IN_ICNT_TO_SHADER,
                       m_gpu.gpu_sim_cycle + m_gpu.gpu_tot_sim_cycle);
        ::icnt_push(m_gpu.m_shader_config->mem2device(i), mf->get_tpc(), mf,
                    response_size, 0);
        m_gpu.m_memory_sub_partition[i]->pop();
        partiton_replys_in_parallel_per_cycle++;
      } else {
        m_gpu.gpu_stall_icnt2sh++;
      }
    } else {
      m_gpu.m_memory_sub_partition[i]->pop();
    }
  }
}

void icnt_handler::icnt_transfer_tick() {
  icnt_transfer(0);
}
