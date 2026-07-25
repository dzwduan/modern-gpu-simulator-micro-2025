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

#include "interwarp_coalescing_unit.h"

#include <algorithm>
#include <iostream>
#include <limits>

#include "ldst_unit_sm.h"
#include "sm.h"

namespace remodel {

InterWarpCoalescingUnit::InterWarpCoalescingUnit(ldst_unit_sm* mem_unit, 
  unsigned int num_tables, 
  unsigned int max_size_per_table) 
  : m_ldst_unit_sm(mem_unit),
    m_num_tables(num_tables),
    m_max_size_per_table(max_size_per_table) {
  m_selection_policy = mem_unit->get_SM()->get_config()->interwarp_coalescing_selection_policy;
  m_warppool_current_policy = InterWarpCoalescingSelectionPolicies::IWCOAL_OLDEST;
  m_intercoalescing_tables.resize(num_tables);
  m_last_greedy_warp_id = 0;
}

InterWarpCoalescingUnit::~InterWarpCoalescingUnit() {
  // Clean up any remaining entries in the tables
  for (auto& table : m_intercoalescing_tables) {
    for (auto& entry : table) {
      if (entry.second) {
        delete entry.second;
      }
    }
    table.clear();
  }
}

new_addr_type InterWarpCoalescingUnit::get_addr_signature(new_addr_type addr, memory_space_t space) {
  constexpr unsigned int SPACE_BITS = 4; // If we add more _memory_space_t we must increased it
  return (static_cast<new_addr_type>(space.get_type()) << (sizeof(new_addr_type)*8 - SPACE_BITS)) | 
         (addr & ((static_cast<new_addr_type>(1) << (sizeof(new_addr_type)*8 - SPACE_BITS)) - 1));
}

bool InterWarpCoalescingUnit::insert_access(mem_access_t* acc) {
  assert(acc);
  assert(!acc->is_write());
  bool inserted = false;
  new_addr_type signature = get_addr_signature(acc->get_addr(), acc->get_space());
  
  unsigned int table_idx = 0;
  if(m_num_tables > 1) {
    // Figure out the table in case that there are several
    std::cout << "Error: Multiple tables not supported yet" << std::endl;
    fflush(stdout);
    abort();
  }
  
  // Check if we already have this address in the table
  auto& target_table = m_intercoalescing_tables[table_idx];
  auto it = target_table.find(signature);
  if (it != target_table.end()) {
    // QUE PASA SI VIENE L1 BYPASS y HAY L1D ya ahi o viceversa? De momento que haga lo que decida el primer acceso.
    
    // Append information to the existing entry
    it->second->get_access_coal_info().m_pcs_requesting.insert(acc->get_inst()->pc);
    it->second->get_access_coal_info().m_warp_id_requesting.insert(acc->get_inst()->warp_id());
    it->second->get_access_coal_info().m_prts_requesting.push_back(acc->get_inst()->m_prt_id);
    unsigned int size_acc = std::max(acc->get_size(), it->second->get_size());
    it->second->set_size(size_acc);
    it->second->set_sector_mask(it->second->get_sector_mask() | acc->get_sector_mask());
    if(acc->get_inst()->get_extra_trace_instruction_info().get_control_bits().get_is_new_write_barrier()) {
      it->second->get_access_coal_info().m_dep_counters_id_requesting.insert(acc->get_inst()->get_extra_trace_instruction_info().get_control_bits().get_id_new_write_barrier());
    }
    inserted = true;
    delete acc;
    m_ldst_unit_sm->get_SM()->m_sm_stats.m_stats_map["total_accesses_coalesced"]->increment_with_integer(1);
  }else if(target_table.size() < m_max_size_per_table) {
    acc->set_cycle_inserted_inter_coal(m_ldst_unit_sm->get_SM()->get_current_gpu_cycle());
    target_table[signature] = acc;
    inserted = true;
    m_ldst_unit_sm->get_SM()->m_sm_stats.m_stats_map["total_accesses_not_coalesced"]->increment_with_integer(1);
  }
  return inserted;
}

bool InterWarpCoalescingUnit::access_is_candidate_to_be_inserted(mem_access_t *acc) {
  bool res = true;
  if( acc->is_write() || (acc->get_space() == miscellaneous_space) ||
      (acc->get_space() == tex_space) || (acc->get_space() == surf_space)) { // As texture and surface addresses are not generated at all by traces, we are not confident to coalesce them.
    res = false;
  }else if(acc->get_access_coal_info().m_dep_counters_id_requesting.empty()) {
    res = false;
  }// Falta el de STRONG y el de Atomics
  return res;
}

bool InterWarpCoalescingUnit::can_pop_access() {
  bool can_pop = false;
  for (auto& table : m_intercoalescing_tables) {
    if(!table.empty()) {
      can_pop = true;
      break;
    }
  }
  return can_pop;
}

InterWarpCoalescingSelectionPolicies InterWarpCoalescingUnit::get_warppool_selection_policy() {
  return m_warppool_current_policy;
}

void InterWarpCoalescingUnit::change_warppool_current_policy(InterWarpCoalescingSelectionPolicies new_policy) {
  m_warppool_current_policy = new_policy;
}

pop_interwarp_result InterWarpCoalescingUnit::pop_policy_oldest() {
  pop_interwarp_result res;
  res.m_it_to_pop = m_intercoalescing_tables[0].begin();
  unsigned long long cycle_to_pop = std::numeric_limits<unsigned long long>::max();
  for(unsigned int idx_table = 0; idx_table < m_num_tables; idx_table++) {
    for(auto it = m_intercoalescing_tables[idx_table].begin(); it != m_intercoalescing_tables[idx_table].end(); it++) {
      if(it->second->get_cycle_inserted_inter_coal() < cycle_to_pop) {
        res.m_it_to_pop = it;
        cycle_to_pop = it->second->get_cycle_inserted_inter_coal();
        res.m_found = true;
        res.m_table_idx = idx_table;
      }
    }
  }
  return res;
}

pop_interwarp_result InterWarpCoalescingUnit::pop_policy_gtl_warpid() {
  pop_interwarp_result res_greedy;
  pop_interwarp_result res_lowest;
  pop_interwarp_result res;
  unsigned int lowest_warp_id_candidate = std::numeric_limits<unsigned int>::max();
  for(unsigned int idx_table = 0; (idx_table < m_num_tables) && !res_greedy.m_found; idx_table++) {
    for(auto it = m_intercoalescing_tables[idx_table].begin(); (it != m_intercoalescing_tables[idx_table].end()) && !res_greedy.m_found; it++) {
      if(it->second->get_access_coal_info().m_warp_id_requesting.find(m_last_greedy_warp_id) != it->second->get_access_coal_info().m_warp_id_requesting.end()) {
        res_greedy.m_it_to_pop = it;
        res_greedy.m_found = true;
        res_greedy.m_table_idx = idx_table;
        lowest_warp_id_candidate = m_last_greedy_warp_id;
      }else {
        for(auto wid : it->second->get_access_coal_info().m_warp_id_requesting) {
          if(wid < lowest_warp_id_candidate) {
            lowest_warp_id_candidate = wid;
            res_lowest.m_it_to_pop = it;
            res_lowest.m_found = true;
            res_lowest.m_table_idx = idx_table;
          }
        }
      }
    }
  }
  if(res_greedy.m_found) {
    res = res_greedy;
  }else {
    res = res_lowest;
  }
  if(res.m_found) {
    m_last_greedy_warp_id = lowest_warp_id_candidate;
  }
  return res;
}

pop_interwarp_result InterWarpCoalescingUnit::pop_policy_dep_counters(bool checking_warp_id) {
  pop_interwarp_result res;
  assert(m_num_tables == 1);// DE MOMENTO
  unsigned int idx_table = 0;
  unsigned long long cycle_to_pop = std::numeric_limits<unsigned long long>::max();
  for(auto it_acc = m_intercoalescing_tables[idx_table].begin(); (it_acc != m_intercoalescing_tables[idx_table].end()); it_acc++) {
    for(unsigned int wid = 0; (wid < m_ldst_unit_sm->get_SM()->get_config()->max_warps_per_shader); wid++) { 
      auto &waiting_deps_of_warp = m_ldst_unit_sm->get_SM()->m_interwarp_coal_warps_waiting_dep_counter->m_waiting_dep_counters_per_warp[wid].m_waiting_dep_counters;
      for(auto it_deps = waiting_deps_of_warp.begin(); (it_deps != waiting_deps_of_warp.end()); it_deps++) {
        bool has_dep_id = it_acc->second->get_access_coal_info().m_dep_counters_id_requesting.find(it_deps->first) != it_acc->second->get_access_coal_info().m_dep_counters_id_requesting.end();
        if(has_dep_id)  {
          bool found = checking_warp_id ? (it_acc->second->get_access_coal_info().m_warp_id_requesting.find(wid) != it_acc->second->get_access_coal_info().m_warp_id_requesting.end()) : true;
          if(found && (it_acc->second->get_cycle_inserted_inter_coal() < cycle_to_pop)) {
            res.m_it_to_pop = it_acc;
            res.m_found = true;
            res.m_table_idx = 0; // DE MOMENTO
            cycle_to_pop = it_acc->second->get_cycle_inserted_inter_coal();
          }
        }
      }
    }
  }
  return res;
}

mem_access_t* InterWarpCoalescingUnit::pop_access(bool need_to_drain_intercoalescing_unit) {
  mem_access_t* res = nullptr;
  pop_interwarp_result pop_info;
  bool checking_warp_id = (m_selection_policy == DEP_COUNT_WAIT_OLDEST_INST_IBUFFER_CHECKING_WARP_ID) || (m_selection_policy == DEP_COUNT_WAIT_DETECTED_AT_DECODE_CHECKING_WARP_ID);
  switch(m_selection_policy) {
    case InterWarpCoalescingSelectionPolicies::IWCOAL_OLDEST:
      pop_info = pop_policy_oldest();
      break;
    case GTL_WARPID:
      pop_info = pop_policy_gtl_warpid();
      if(!pop_info.m_found) {
        pop_info = pop_policy_oldest();
      }
      break;
    case WARPPOOL_HYBRID:
      if(m_warppool_current_policy == IWCOAL_OLDEST) {
        pop_info = pop_policy_oldest();
      }else {
        pop_info = pop_policy_gtl_warpid();
      }
      break;
    case InterWarpCoalescingSelectionPolicies::DEP_COUNT_WAIT_OLDEST_INST_IBUFFER_GENERIC:
    case InterWarpCoalescingSelectionPolicies::DEP_COUNT_WAIT_OLDEST_INST_IBUFFER_CHECKING_WARP_ID:
    case InterWarpCoalescingSelectionPolicies::DEP_COUNT_WAIT_DETECTED_AT_DECODE_GENERIC:
    case InterWarpCoalescingSelectionPolicies::DEP_COUNT_WAIT_DETECTED_AT_DECODE_CHECKING_WARP_ID:
      pop_info = pop_policy_dep_counters(checking_warp_id);
      if(!pop_info.m_found && m_ldst_unit_sm->get_SM()->is_any_subcore_problems_of_fordward_progress() && m_ldst_unit_sm->get_prt().is_full()) {
        pop_info = pop_policy_oldest();
      }
      break;
    default:
      std::cout << "Error: Invalid selection policy" << std::endl;
      fflush(stdout);
      abort();
  }
  if(need_to_drain_intercoalescing_unit && !pop_info.m_found) {
    pop_info = pop_policy_oldest();
  }
  if(pop_info.m_found) {
    res = pop_info.m_it_to_pop->second;
    m_intercoalescing_tables[pop_info.m_table_idx].erase(pop_info.m_it_to_pop);
  }
  return res;
}

bool InterWarpCoalescingUnit::is_empty() {
  bool res = true;
  for(unsigned int idx_table = 0; idx_table < m_num_tables; idx_table++) {
    if(!m_intercoalescing_tables[idx_table].empty()) {
      res = false;
      break;
    }
  }
  return res;
}

} // namespace remodel
