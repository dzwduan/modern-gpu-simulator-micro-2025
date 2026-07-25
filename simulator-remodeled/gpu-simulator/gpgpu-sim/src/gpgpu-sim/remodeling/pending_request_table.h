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

#include <cstdio>
#include <limits>
#include <memory>
#include <queue>
#include <vector>

#include "../../abstract_hardware_model.h"

namespace remodel {

class ldst_unit_sm;

class PendingRequestTableEntry {
  public:
    PendingRequestTableEntry();

    void assign_entry(std::shared_ptr<warp_inst_t> &inst);
    void set_id(unsigned int id);
    void decrement_num_pending_accesses_to_solve();
    void increment_num_pending_accesses_to_solve();
    void release();

    std::shared_ptr<warp_inst_t>& get_inst();
    unsigned int get_id() const;
    unsigned int get_num_pending_accesses_to_solve() const;
    unsigned long long get_assignation_cycle() const;
    unsigned int get_total_num_accesses_to_do() const;
    void set_assignation_cycle(unsigned long long cycle);
    bool is_free() const;
    bool is_pending_to_receive_requests() const;
    void print(FILE *fout) const;

  private:
    std::shared_ptr<warp_inst_t> m_inst;
    unsigned int m_id;
    unsigned int m_num_pending_accesses_to_solve;
    bool m_is_free;
    unsigned long long m_assignation_cycle;
    unsigned int m_total_num_accesses_to_do; // Only useful for global memory accesses with active threads
};

struct cluster_prt_candidate {

  cluster_prt_candidate() : m_id(std::numeric_limits<unsigned int>::max()), m_cycle(std::numeric_limits<unsigned int>::max()) {}
  unsigned int m_id;
  unsigned int m_cycle;
};

class PendingRequestTable {
  public:
    PendingRequestTable(unsigned int max_num_entries, ldst_unit_sm *ldst_unit_sm);
    
    void assign_entry(std::shared_ptr<warp_inst_t> &inst);
    void reactivate_entry(std::shared_ptr<warp_inst_t> &inst);
    void solve_access(unsigned int id);
    void get_accesses_to_coalescing(std::vector<mem_access_t*> &current_accs);
    void get_access_to_next_stage(std::queue<mem_access_t*> &current_accs);
    mem_access_t* get_next_processed_access(unsigned int id);
    std::shared_ptr<warp_inst_t> pop_entry(unsigned int icnt_id);
    std::shared_ptr<warp_inst_t> pop_entries(unsigned int icnt_id);

    bool is_full();
    bool is_empty();

    bool are_entries_to_pop_icnt_id(unsigned int icnt_id);
    bool are_entries_to_process_coalescing();

    unsigned int oldest_selection_policy();
    unsigned int same_last_warp_id();
    unsigned int same_last_pc();
    unsigned int warp_id_N_cluster_priority_and_oldest_inside_each_cluster();
    unsigned int dep_counters_waiting(bool checking_warp_id);

    void management_entries_to_process();

    bool is_entry_going_to_global_memory(unsigned int id);

    bool is_entry_going_to_l1d(unsigned int id);

    void print(FILE *fout) const;
  private:
    unsigned int m_max_num_entries;
    unsigned int m_max_num_entries_to_process_concurrently;
    std::vector<PendingRequestTableEntry> m_entries;
    std::queue<unsigned int> m_entries_id_free_list; 
    std::vector<unsigned int> m_entries_id_pending_list_to_process;
    // One queue per subcore and one extra for icnt of LDGST
    std::vector<std::queue<unsigned int>> m_entries_id_pending_list_to_free;
    std::vector<unsigned int> m_current_entries_id_being_processed; 
    std::vector<unsigned int> m_entries_id_finishing_processed; 
    ldst_unit_sm *m_ldst_unit_sm;
    PRTSelectionPolicies m_selection_policy;
    unsigned int m_last_warp_id;
    address_type m_last_pc;
};

} // namespace remodel
