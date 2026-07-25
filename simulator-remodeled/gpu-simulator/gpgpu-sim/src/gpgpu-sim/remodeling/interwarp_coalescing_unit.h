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

#pragma once

#include <map>
#include <vector>

#include "../../abstract_hardware_model.h"

namespace remodel {

class ldst_unit_sm;

struct pop_interwarp_result {
  pop_interwarp_result() : m_found(false), m_table_idx(0) {}
  bool m_found;
  unsigned int m_table_idx;
  std::map<new_addr_type, mem_access_t *>::iterator m_it_to_pop;
};

class InterWarpCoalescingUnit {
  public:
  InterWarpCoalescingUnit(ldst_unit_sm * mem_unit, unsigned int num_tables, unsigned int max_size_per_table);
  ~InterWarpCoalescingUnit();

  new_addr_type get_addr_signature(new_addr_type addr, memory_space_t space);
  bool insert_access(mem_access_t *acc);
  InterWarpCoalescingSelectionPolicies get_warppool_selection_policy();
  void change_warppool_current_policy(InterWarpCoalescingSelectionPolicies new_policy);
  pop_interwarp_result pop_policy_oldest();
  pop_interwarp_result pop_policy_gtl_warpid();
  pop_interwarp_result pop_policy_dep_counters(bool checking_warp_id);
  mem_access_t* pop_access(bool need_to_drain_intercoalescing_unit);
  bool can_pop_access();
  bool access_is_candidate_to_be_inserted(mem_access_t *acc);
  bool is_empty();
  private:
    ldst_unit_sm *m_ldst_unit_sm;
    std::vector<std::map<new_addr_type, mem_access_t*>> m_intercoalescing_tables;
    unsigned int m_num_tables;
    unsigned int m_max_size_per_table;
    InterWarpCoalescingSelectionPolicies m_selection_policy;
    InterWarpCoalescingSelectionPolicies m_warppool_current_policy;
    unsigned int m_last_greedy_warp_id;
};

} // namespace remodel
