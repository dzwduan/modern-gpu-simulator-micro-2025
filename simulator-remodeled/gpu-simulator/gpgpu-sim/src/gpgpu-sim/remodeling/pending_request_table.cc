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

#include "pending_request_table.h"

#include <iostream>
#include <limits>

#include "../gpu-sim.h"
#include "ldst_unit_sm.h"
#include "sm.h"

namespace remodel {

PendingRequestTableEntry::PendingRequestTableEntry() : m_inst(nullptr), m_id(0), m_num_pending_accesses_to_solve(0), m_is_free(true), m_assignation_cycle(0) {

}

void PendingRequestTableEntry::set_id(unsigned id) {
  m_id = id;
}

void PendingRequestTableEntry::assign_entry(std::shared_ptr<warp_inst_t> &inst) {
  m_inst = inst;
  m_is_free = false;
  m_num_pending_accesses_to_solve = 0;
  m_total_num_accesses_to_do = inst->accessq_count(); // Only useful for global memory accesses with active threads
}

void PendingRequestTableEntry::release() {
  m_inst = nullptr;
  m_is_free = true;
  m_num_pending_accesses_to_solve = 0;
  m_assignation_cycle = 0;
  m_total_num_accesses_to_do = 0;
}

void PendingRequestTableEntry::decrement_num_pending_accesses_to_solve() {
  assert(m_num_pending_accesses_to_solve > 0);
  m_num_pending_accesses_to_solve--;
}

void PendingRequestTableEntry::increment_num_pending_accesses_to_solve() {
  m_num_pending_accesses_to_solve++;
}

unsigned int PendingRequestTableEntry::get_total_num_accesses_to_do() const { 
  return m_total_num_accesses_to_do; 
}


std::shared_ptr<warp_inst_t> &PendingRequestTableEntry::get_inst() { return m_inst; }
unsigned int PendingRequestTableEntry::get_id() const { return m_id; }
unsigned int PendingRequestTableEntry::get_num_pending_accesses_to_solve() const { return m_num_pending_accesses_to_solve; }
unsigned long long PendingRequestTableEntry::get_assignation_cycle() const { return m_assignation_cycle; }
void PendingRequestTableEntry::set_assignation_cycle(unsigned long long cycle) { m_assignation_cycle = cycle; }
bool PendingRequestTableEntry::is_free() const { return m_is_free; }
bool PendingRequestTableEntry::is_pending_to_receive_requests() const { return m_num_pending_accesses_to_solve > 0; }

void PendingRequestTableEntry::print(FILE *fout) const {
  fprintf(fout, "PRT Entry[%u]: ", m_id);
  
  if (m_is_free) {
      fprintf(fout, "FREE\n");
      return;
  }else {

    if (m_inst == nullptr) {
        fprintf(fout, "ERROR - Not free but null instruction. Assignation cycle: %llu\n", m_assignation_cycle);
        return;
    }

    // Print instruction details
    fprintf(fout, "warp %u PC=0x%llx ", 
            m_inst->warp_id(), 
            (unsigned long long)m_inst->pc);

    // Print operation type
    fprintf(fout, "op=");
    switch (m_inst->op) {
        case LOAD_OP:           fprintf(fout, "LOAD"); break;
        case STORE_OP:          fprintf(fout, "STORE"); break;
        case MEMORY_BARRIER_OP: fprintf(fout, "BARRIER"); break;
        case GRID_BARRIER_OP: fprintf(fout, "GRID_BARRIER"); break;
        case TENSOR_CORE_LOAD_OP:  fprintf(fout, "TENSOR_LD"); break;
        case TENSOR_CORE_STORE_OP: fprintf(fout, "TENSOR_ST"); break;
        default:               fprintf(fout, "OTHER(%d)", m_inst->op);
    }

    // Print memory space
    fprintf(fout, " space=");
    switch (m_inst->space.get_type()) {
        case global_space:      fprintf(fout, "global"); break;
        case shared_space:      fprintf(fout, "shared"); break;
        case const_space:       fprintf(fout, "const"); break;
        case local_space:       fprintf(fout, "local"); break;
        case tex_space:         fprintf(fout, "texture"); break;
        case param_space_local: fprintf(fout, "param"); break;
        case undefined_space:   fprintf(fout, "undefined"); break;
        default:               fprintf(fout, "other(%d)", m_inst->space.get_type());
    }
    fprintf(fout, ". Assignation cycle: %llu\n", m_assignation_cycle);
  }
}

PendingRequestTable::PendingRequestTable(unsigned int num_entries, ldst_unit_sm *ldst_unit_sm) : m_max_num_entries(num_entries), m_ldst_unit_sm(ldst_unit_sm) {
  m_entries.resize(num_entries);
  for(unsigned int i = 0; i < num_entries; i++) {
    m_entries[i].set_id(i);
    m_entries_id_free_list.push(i);
  }
  m_entries_id_pending_list_to_free.resize(m_ldst_unit_sm->get_SM()->get_num_subcores() + 1);
  m_selection_policy = ldst_unit_sm->get_SM()->get_config()->prt_selection_policy;
  m_max_num_entries_to_process_concurrently = ldst_unit_sm->get_SM()->get_config()->number_of_coalescers;
  m_last_warp_id = std::numeric_limits<unsigned int>::max();
  m_last_pc = std::numeric_limits<address_type>::max();
}

bool PendingRequestTable::is_full() {
  return m_entries_id_free_list.empty();
}

bool PendingRequestTable::is_empty() {
  return m_entries_id_free_list.size() == m_max_num_entries;
}

bool PendingRequestTable::are_entries_to_pop_icnt_id(unsigned int icnt_id) {
  return !m_entries_id_pending_list_to_free[icnt_id].empty();
}

void PendingRequestTable::reactivate_entry(std::shared_ptr<warp_inst_t> &inst) {
  unsigned int id = inst->m_prt_id;
  m_entries_id_pending_list_to_process.push_back(id);
  m_entries[id].assign_entry(inst);
  m_entries[id].set_assignation_cycle(m_ldst_unit_sm->get_SM()->get_current_gpu_cycle());
}

void PendingRequestTable::assign_entry(std::shared_ptr<warp_inst_t> &inst) {
  assert(!is_full());
  unsigned int id = m_entries_id_free_list.front();
  m_entries_id_free_list.pop();
  inst->m_prt_assigned = true;
  inst->m_prt_id = id;
  m_entries[id].assign_entry(inst);
  m_entries_id_pending_list_to_process.push_back(id);
  m_entries[id].set_assignation_cycle(m_ldst_unit_sm->get_SM()->get_current_gpu_cycle());
}

void PendingRequestTable::solve_access(unsigned int id) {
  assert(id < m_max_num_entries);
  assert(!m_entries[id].is_free());
  assert(m_entries[id].is_pending_to_receive_requests());

  m_entries[id].decrement_num_pending_accesses_to_solve();
  if(!m_entries[id].is_pending_to_receive_requests() && m_entries[id].get_inst()->accessq_empty()) {
    unsigned int subid = m_entries[id].get_inst()->get_subcore_id();
    if(m_entries[id].get_inst()->m_is_ldgsts && (m_entries[id].get_inst()->m_ldgsts_state == LOAD_STAGE)) {
      subid = m_ldst_unit_sm->get_reserved_idx_icnt_to_shmem();
    }
    m_entries_id_pending_list_to_free[subid].push(id);
  }
}

std::shared_ptr<warp_inst_t> PendingRequestTable::pop_entry(unsigned int icnt_id) {
  assert(!m_entries_id_pending_list_to_free[icnt_id].empty());
  unsigned int id = m_entries_id_pending_list_to_free[icnt_id].front();
  std::shared_ptr<warp_inst_t> res = std::move(m_entries[id].get_inst());
  bool is_ldgsts = res->m_is_ldgsts;
  bool is_ldgsts_store = is_ldgsts && (res->m_ldgsts_state == STORE_STAGE);
  m_entries_id_pending_list_to_free[icnt_id].pop();
  bool skip_wb = res->skip_wb;
  if(is_ldgsts && !is_ldgsts_store) {
    skip_wb = false;
  }
  if(skip_wb || res->is_store()) {
    if(res->space.is_shared()) {
      m_ldst_unit_sm->m_current_num_shared_mem_inst--;
    }else {
      m_ldst_unit_sm->m_current_num_normal_mem_inst--;
    }
    m_ldst_unit_sm->get_SM()->instruction_retirement(res.get());
    res = nullptr;
  }
  if(!is_ldgsts || is_ldgsts_store) {
    m_entries[id].release();
    m_entries_id_free_list.push(id);
  }
  return res;
}

std::shared_ptr<warp_inst_t> PendingRequestTable::pop_entries(unsigned int icnt_id) {
  std::shared_ptr<warp_inst_t> res = nullptr;
  bool something_pop = false;
  while(are_entries_to_pop_icnt_id(icnt_id) && res == nullptr) {
    res = pop_entry(icnt_id);
    something_pop = true;
  }
  if (something_pop) {
    m_ldst_unit_sm->m_last_inst_gpu_sim_cycle = m_ldst_unit_sm->get_SM()->get_gpu()->gpu_sim_cycle;
    m_ldst_unit_sm->m_last_inst_gpu_tot_sim_cycle = m_ldst_unit_sm->get_SM()->get_gpu()->gpu_tot_sim_cycle;
  }
  return res;
}

bool PendingRequestTable::are_entries_to_process_coalescing() {
  return !m_current_entries_id_being_processed.empty();
}

unsigned int PendingRequestTable::oldest_selection_policy() {
  unsigned int id = m_entries_id_pending_list_to_process.front();
  bool is_safe_to_select = true;
  unsigned int value_of_increment = 0;
  if(is_entry_going_to_global_memory(id) && is_entry_going_to_l1d(id)) { 
    value_of_increment = m_entries[id].get_inst()->accessq_count();
    is_safe_to_select = m_ldst_unit_sm->can_entry_be_selected_for_processing(value_of_increment);
  }

  if(is_safe_to_select) {
    m_ldst_unit_sm->increment_num_reserved_associativity_currently_processing(value_of_increment);
    m_entries_id_pending_list_to_process.erase(m_entries_id_pending_list_to_process.begin());
  }else {
    id = std::numeric_limits<unsigned int>::max();
  }
  return id;
}

bool PendingRequestTable::is_entry_going_to_global_memory(unsigned int id) {
  return (m_entries[id].get_inst()->space == global_space )|| (m_entries[id].get_inst()->space == local_space ) || (m_entries[id].get_inst()->space == param_space_local );
}

bool PendingRequestTable::is_entry_going_to_l1d(unsigned int id) {
  bool res = true;
  if((m_entries[id].get_inst()->cache_op == CACHE_GLOBAL) || (m_ldst_unit_sm->get_L1D() == NULL)) {
    res = false;
  }
  return res;
}

unsigned int PendingRequestTable::same_last_warp_id() {
  unsigned int id = std::numeric_limits<unsigned int>::max();
  for(auto it = m_entries_id_pending_list_to_process.begin(); it != m_entries_id_pending_list_to_process.end(); it++) {
    if(m_entries[*it].get_inst()->warp_id() == m_last_warp_id) {
      id = *it;
      m_entries_id_pending_list_to_process.erase(it);
      break;
    }
  }
  return id;
}

unsigned int PendingRequestTable::same_last_pc() {
  unsigned int id = std::numeric_limits<unsigned int>::max();
  for(auto it = m_entries_id_pending_list_to_process.begin(); it != m_entries_id_pending_list_to_process.end(); it++) {
    if(m_entries[*it].get_inst()->pc == m_last_pc) {
      id = *it;
      m_entries_id_pending_list_to_process.erase(it);
      break;
    }
  }
  return id;
}

unsigned int PendingRequestTable::warp_id_N_cluster_priority_and_oldest_inside_each_cluster() {
  std::vector<cluster_prt_candidate> clusters;
  unsigned int candidate_cluster_id = std::numeric_limits<unsigned int>::max();
  unsigned int num_warpid_N_clusters = m_ldst_unit_sm->get_SM()->get_config()->number_of_clusters_for_prt_selection;
  clusters.resize(num_warpid_N_clusters);
  auto it_to_select = m_entries_id_pending_list_to_process.begin();
  for(auto it = m_entries_id_pending_list_to_process.begin(); it != m_entries_id_pending_list_to_process.end(); it++) {
    auto &entry = m_entries[*it];
    unsigned int wid = entry.get_inst()->warp_id();
    unsigned int cluster_id = wid / num_warpid_N_clusters;
    bool changed = false;
    if(entry.get_assignation_cycle() < clusters[cluster_id].m_cycle) {
      clusters[cluster_id].m_cycle = entry.get_assignation_cycle();
      changed = true;
      clusters[cluster_id].m_id = entry.get_id();
      if(cluster_id == candidate_cluster_id) {
        it_to_select = it;
      }
    }
    if(changed && (candidate_cluster_id == std::numeric_limits<unsigned int>::max())) {
      candidate_cluster_id = cluster_id;
      it_to_select = it;
    }else if(changed && (cluster_id < candidate_cluster_id)) {
      candidate_cluster_id = cluster_id;
      it_to_select = it;
    }
  }
  assert(candidate_cluster_id != std::numeric_limits<unsigned int>::max());
  m_entries_id_pending_list_to_process.erase(it_to_select);
  return clusters[candidate_cluster_id].m_id;
}

unsigned int PendingRequestTable::dep_counters_waiting(bool checking_warp_id) {
  unsigned int id = std::numeric_limits<unsigned int>::max();
  bool found = false;
  for(auto it = m_entries_id_pending_list_to_process.begin(); !found && (it != m_entries_id_pending_list_to_process.end()); it++) {
    for(unsigned int wid = 0; (wid < m_ldst_unit_sm->get_SM()->get_config()->max_warps_per_shader) && !found; wid++) {
      auto &waiting_deps_of_warp = m_ldst_unit_sm->get_SM()->m_interwarp_coal_warps_waiting_dep_counter->m_waiting_dep_counters_per_warp[wid].m_waiting_dep_counters;
      for(auto it_deps = waiting_deps_of_warp.begin(); !found && (it_deps != waiting_deps_of_warp.end()); it_deps++) {
        if(m_entries[*it].get_inst()->get_extra_trace_instruction_info().get_control_bits().get_is_new_write_barrier()) {
          bool has_dep_id = it_deps->first == m_entries[*it].get_inst()->get_extra_trace_instruction_info().get_control_bits().get_id_new_write_barrier();
          has_dep_id = checking_warp_id ? (wid == m_entries[*it].get_inst()->warp_id()) : has_dep_id;
          if(has_dep_id) {
            found = true;
            id = *it;
            m_entries_id_pending_list_to_process.erase(it);
          }
        }      
      }
    }
  }
  return id;
}

void PendingRequestTable::management_entries_to_process() {
  while(!m_entries_id_finishing_processed.empty()) {
    unsigned int id = m_entries_id_finishing_processed.front();
    m_entries_id_finishing_processed.erase(m_entries_id_finishing_processed.begin());
    bool erased = false;
    for(auto it_proc = m_current_entries_id_being_processed.begin(); it_proc != m_current_entries_id_being_processed.end(); it_proc++) {
      if(*it_proc == id) {
        m_current_entries_id_being_processed.erase(it_proc);
        erased = true;
        break;
      }
    }
    assert(erased);
  }
  bool checking_warp_id = (m_selection_policy == PRTSelectionPolicies::DEP_COUNT_WAIT_CHECKING_WARP_ID_THEN_OLDEST);
  bool can_continue = true; // this logic may be need to implemented in other policies if interwarp coalescing is used. As I have discarded that line of research, I have not invested time in doing it.
  while(can_continue && (m_current_entries_id_being_processed.size() < m_max_num_entries_to_process_concurrently)
      && !m_entries_id_pending_list_to_process.empty()) {
    unsigned int id = std::numeric_limits<unsigned int>::max();
    switch(m_selection_policy) {
      case PRTSelectionPolicies::OLDEST:
        id = oldest_selection_policy();
        if(id != std::numeric_limits<unsigned int>::max()) {
          m_current_entries_id_being_processed.push_back(id);
        }else {
          can_continue = false;
        }
        break;
      case PRTSelectionPolicies::SAME_LAST_WARP_ID_THEN_OLDEST:
        id = same_last_warp_id();
        if(id == std::numeric_limits<unsigned int>::max()) {
          id = oldest_selection_policy();
          if(id != std::numeric_limits<unsigned int>::max()) {
            m_current_entries_id_being_processed.push_back(id);
          }else {
            can_continue = false;
          }
        }
        break;
      case PRTSelectionPolicies::SAME_LAST_INST_PC_THEN_OLDEST:
        id = same_last_pc();
        if(id == std::numeric_limits<unsigned int>::max()) {
          id = oldest_selection_policy();
          if(id != std::numeric_limits<unsigned int>::max()) {
            m_current_entries_id_being_processed.push_back(id);
          }else {
            can_continue = false;
          }
        }
        break;
      case PRTSelectionPolicies::WARPID_N_CLUSTERS_WITH_OLDEST:
        id = warp_id_N_cluster_priority_and_oldest_inside_each_cluster();
        assert(id != std::numeric_limits<unsigned int>::max());
        m_current_entries_id_being_processed.push_back(id);
        break;
      case PRTSelectionPolicies::DEP_COUNT_WAIT_GENERIC_THEN_OLDEST:
      case PRTSelectionPolicies::DEP_COUNT_WAIT_CHECKING_WARP_ID_THEN_OLDEST:
        id = dep_counters_waiting(checking_warp_id);
        if(id == std::numeric_limits<unsigned int>::max()) {
          id = oldest_selection_policy();
          if(id != std::numeric_limits<unsigned int>::max()) {
            m_current_entries_id_being_processed.push_back(id);
          }else {
            can_continue = false;
          }
        }
        break;
      default:
        std::cout << "Error: Invalid selection policy" << std::endl;
        fflush(stdout);
        abort();
    }
  }
}

void PendingRequestTable::get_accesses_to_coalescing(std::vector<mem_access_t*> &current_accs) {
  auto it_proc = m_current_entries_id_being_processed.begin();
  bool inserted = false;
  while( (current_accs.size() < m_max_num_entries_to_process_concurrently ) && (it_proc != m_current_entries_id_being_processed.end())) {
    unsigned int id = *it_proc;
    mem_access_t *acc = get_next_processed_access(id);
    assert(acc != nullptr);
    current_accs.push_back(acc);
    it_proc++;
    inserted = true;
  }
  if(!inserted) {
    m_ldst_unit_sm->get_SM()->m_sm_stats.m_stats_map["gpgpu_n_stall_dispatch_to_subpipeline_mem"]->increment_with_integer(1);
  }
}

void PendingRequestTable::get_access_to_next_stage(std::queue<mem_access_t*> &current_accs) {
  auto it_proc = m_current_entries_id_being_processed.begin();
  bool inserted = false;
  while( (current_accs.size() < m_max_num_entries_to_process_concurrently ) && (it_proc != m_current_entries_id_being_processed.end())) {
    unsigned int id = *it_proc;
    mem_access_t *acc = get_next_processed_access(id);
    assert(acc != nullptr);
    current_accs.push(acc);
    it_proc++;
    inserted = true;
  }
  if(!inserted && (current_accs.size() == m_ldst_unit_sm->get_SM()->get_config()->number_of_coalescers ) ) {
    m_ldst_unit_sm->get_SM()->m_sm_stats.m_stats_map["gpgpu_n_stall_dispatch_to_subpipeline_mem"]->increment_with_integer(1);
  }
}

mem_access_t* PendingRequestTable::get_next_processed_access(unsigned int id) {
  assert(id < m_max_num_entries);
  mem_access_t *res_acc = nullptr;
  assert(!m_entries[id].is_free());
  if((m_entries[id].get_inst()->active_count() == 0) || (m_entries[id].get_inst()->op == MEMORY_MISCELLANEOUS_OP) || 
      (m_entries[id].get_inst()->is_any_kind_of_barrier()) ) {
    m_entries[id].get_inst()->skip_wb = true;
    m_entries_id_finishing_processed.push_back(id);
    res_acc = new mem_access_t(m_ldst_unit_sm->get_SM()->get_config()->gpgpu_ctx);
    res_acc->set_space(miscellaneous_space);
    res_acc->set_write(false);
    res_acc->set_last_access(true);
    res_acc->set_size(32);
    if(m_entries[id].get_inst()->op == GRID_BARRIER_OP) {
      res_acc->set_type(GRID_BARRIER_ACC);
    }
  }else if((m_entries[id].get_inst()->op == TEXTURE_OP) && m_entries[id].get_inst()->accessq_empty()) {
    // It seems that texture accesses are not being processed because traces have not captured them properly
    m_entries_id_finishing_processed.push_back(id);
    bool is_write = m_entries[id].get_inst()->is_store();
    mem_access_type access_type = TEXTURE_ACC_R;
    res_acc = new mem_access_t(access_type, 0, 32, is_write, m_ldst_unit_sm->get_SM()->get_config()->gpgpu_ctx);
    res_acc->set_space(tex_space);
    res_acc->set_last_access(true);
  }else if((m_entries[id].get_inst()->space == sstarr_space) || (m_entries[id].get_inst()->space == shared_space)) {
    assert(m_entries[id].get_inst()->has_dispatch_delay());
    bool is_write = m_entries[id].get_inst()->is_store();
    mem_access_type access_type = is_write ? LOCAL_ACC_W : LOCAL_ACC_R;
    res_acc = new mem_access_t(access_type, 0, 32, is_write, m_ldst_unit_sm->get_SM()->get_config()->gpgpu_ctx);
    res_acc->set_space(shared_space);
    if(!m_entries[id].get_inst()->dispatch_delay()) {
      m_entries[id].get_inst()->accessq_clear();
      res_acc->set_last_access(true);
      m_entries_id_finishing_processed.push_back(id);
    }else {
      m_ldst_unit_sm->get_SM()->m_sm_stats.m_stats_map["gpgpu_n_shmem_bkconflict"]->increment_with_integer(1);
      m_ldst_unit_sm->get_SM()->get_stats()->gpgpu_n_shmem_bank_access[m_ldst_unit_sm->get_SM()->get_sid()]++;
    }
  }else {
    if(!m_entries[id].get_inst()->accessq_empty()) {
      res_acc = new mem_access_t(m_entries[id].get_inst()->accessq_back());
      m_entries[id].get_inst()->accessq_pop_back();
    }else {
      bool is_write = m_entries[id].get_inst()->is_store();
      mem_access_type access_type = is_write ? LOCAL_ACC_W : LOCAL_ACC_R;
      res_acc = new mem_access_t(access_type, 0, 32, is_write, m_ldst_unit_sm->get_SM()->get_config()->gpgpu_ctx);
    }
    
    res_acc->set_space(m_entries[id].get_inst()->space);
    if(is_entry_going_to_global_memory(id)) {
      res_acc->set_l1d_bank(m_ldst_unit_sm->get_SM()->get_config()->m_L1D_config.set_bank(res_acc->get_addr()));
    }
    if((m_entries[id].get_inst()->cache_op == CACHE_GLOBAL) || (m_ldst_unit_sm->get_L1D() == NULL) ||
       (m_entries[id].get_inst()->space.is_global() && (m_ldst_unit_sm->get_SM()->get_config()->gmem_skip_L1D && (CACHE_L1 != m_entries[id].get_inst()->cache_op)) )) {
      res_acc->set_l1d_bypass(true);
    }
    
    if(m_entries[id].get_inst()->accessq_empty()) {
      res_acc->set_last_access(true);
      m_entries_id_finishing_processed.push_back(id);
      if(is_entry_going_to_global_memory(id) && is_entry_going_to_l1d(id)) {
        m_ldst_unit_sm->decrement_num_reserved_associativity_currently_processing(m_entries[id].get_total_num_accesses_to_do());
      }
    }else {
      //STATS//
      if(res_acc->is_l1d_bypass()) {
        // it goes directly to l2
        m_ldst_unit_sm->get_SM()->m_sm_stats.m_stats_map["gpgpu_n_directly_to_l2_coalescing_conflicts"]->increment_with_integer(1);
      }else if(res_acc->get_space() == const_space) {
        // It goes to constant cache
        m_ldst_unit_sm->get_SM()->m_sm_stats.m_stats_map["gpgpu_n_cmem_coalescing_conflicts"]->increment_with_integer(1);
      }else if((res_acc->get_space() == tex_space ) || (res_acc->get_space() == surf_space )) {
        // It goes to texture cache 
      }else {
        // It goes to l1d
        m_ldst_unit_sm->get_SM()->m_sm_stats.m_stats_map["gpgpu_n_l1cache_coalescing_conflicts"]->increment_with_integer(1);
      }
      
    }
  }
  m_entries[id].increment_num_pending_accesses_to_solve();
  res_acc->set_inst(m_entries[id].get_inst().get());
  res_acc->get_access_coal_info().m_pcs_requesting.insert(res_acc->get_inst()->pc);
  res_acc->get_access_coal_info().m_warp_id_requesting.insert(res_acc->get_inst()->warp_id());
  res_acc->get_access_coal_info().m_prts_requesting.push_back(res_acc->get_inst()->m_prt_id);
  if(res_acc->get_inst()->get_extra_trace_instruction_info().get_control_bits().get_is_new_write_barrier()) {
    res_acc->get_access_coal_info().m_dep_counters_id_requesting.insert(res_acc->get_inst()->get_extra_trace_instruction_info().get_control_bits().get_id_new_write_barrier());
  }
  m_last_warp_id = res_acc->get_inst()->warp_id();
  m_last_pc = res_acc->get_inst()->pc;
  return res_acc;
}

void PendingRequestTable::print(FILE *fout) const {
  // Print summary information
  fprintf(fout, "\nPending Request Table Summary:\n");
  fprintf(fout, "Total entries: %u\n", m_max_num_entries);
  fprintf(fout, "Number of free entries : %zu\n", m_entries_id_free_list.size());
  fprintf(fout, "Number of entries pending to free: %zu\n", m_entries_id_pending_list_to_free.size());
  fprintf(fout, "Number of entries being processed: %zu\n", m_current_entries_id_being_processed.size());
  fprintf(fout, "Number of entries pending to process: %zu\n", m_entries_id_pending_list_to_process.size());
  unsigned int num_pending_list_to_free = 0;
  for(unsigned int i = 0; i < m_entries_id_pending_list_to_free.size(); i++) {
    fprintf(fout, "Pending list to free in icnt: %u: %zu\n", i, m_entries_id_pending_list_to_free[i].size());
    std::queue<unsigned int> pending_list_to_free_copy = m_entries_id_pending_list_to_free[i];
    while(!pending_list_to_free_copy.empty()) {
      fprintf(fout, "Entry ID: %u\n", pending_list_to_free_copy.front());
      num_pending_list_to_free++;
      pending_list_to_free_copy.pop();
    }
  }
  fprintf(fout, "Total number of entries pending to free: %u\n", num_pending_list_to_free);
    
  // Print all entries
  fprintf(fout, "\nDetailed Entry Status:\n");
  for (unsigned int i = 0; i < m_max_num_entries; i++) {
      m_entries[i].print(fout);
  }
}

} // namespace remodel
