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

#include <vector>
#include <queue>
#include <memory>
#include <deque>

#include "../stats.h"
#include "../shader.h"
#include "access_queue.h"
#include "functional_unit.h"
#include "interwarp_coalescing_unit.h"
#include "pending_request_table.h"

class mem_fetch_interface;
class shader_core_stats;
namespace remodel {

class coalescingAddressStats;
class coalescingStatsPerSm;
class ldst_unit_sm;

uint64_t calculate_constant_address(uint64_t reg_offset_value, traced_operand& op_c);

class ldst_unit_sm : public functional_unit_shared_sm_part {
 public:
  ldst_unit_sm(
    std::vector<register_set_uniptr*> result_ports,
    std::vector<register_set_uniptr*> reception_ports,
    mem_fetch_interface *icnt,
    mem_fetch_interface *icnt_L1C_L1_half_C,
    std::shared_ptr<shader_core_mem_fetch_allocator> mf_allocator,
    SM *core,
    const shader_core_config *config,
    const memory_config *mem_config,
    shader_core_stats *stats,
    unsigned sid,
    unsigned tpc,
    unsigned int max_size_arbiter_to_subpipeline_reg_per_subcore);
  
  ~ldst_unit_sm() override;
  // modifiers
  void issue(register_set_uniptr &inst, unsigned int icnt_id);
  void cycle() override;

  read_only_cache *get_L1C();
  l1_cache *get_L1D();

  SM* get_SM();

  void fill(mem_fetch *mf);
  void flush();
  void invalidate();
  void writeback(unsigned int icnt_id);

  // accessors
  virtual unsigned clock_multiplier() const;

  bool can_issue(const warp_inst_t *inst) const override;

  bool is_dispatch_reg_empty(unsigned int icnt_id) const;

  virtual void active_lanes_in_pipeline();
  virtual bool stallable() const { return true; }
  bool response_buffer_full() const;
  void print(FILE *fout) const;
  void print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                         unsigned &dl1_misses);
  void get_cache_stats(unsigned &read_accesses, unsigned &write_accesses,
                       unsigned &read_misses, unsigned &write_misses,
                       unsigned cache_type);
  void get_cache_stats(cache_stats &cs);

  void get_L1D_sub_stats(struct cache_sub_stats &css) const;
  void get_L1C_sub_stats(struct cache_sub_stats &css) const;
  void get_L1T_sub_stats(struct cache_sub_stats &css) const;


  coalescingStatsPerSm *get_coalescingStatPerSm_l1d();
  coalescingStatsPerSm *get_coalescingStatPerSm_const();
  coalescingStatsPerSm *get_coalescingStatPerSm_sharedmem();

  void reset_coalescingHistory();

  PendingRequestTable& get_prt();

  unsigned int get_reserved_idx_icnt_to_shmem();

  // All the logic related to this functions is because there can be the case that different entries are compiting for the L1D cache and there might be not enough associativity to process all of them without having reservation fails, which leads to a deadlock.
  bool can_entry_be_selected_for_processing(unsigned int value);
  void increment_num_reserved_associativity_currently_processing(unsigned int value);
  void decrement_num_reserved_associativity_currently_processing(unsigned int value);

  // for debugging
  unsigned long long m_last_inst_gpu_sim_cycle;
  unsigned long long m_last_inst_gpu_tot_sim_cycle;
  unsigned int m_current_num_shared_mem_inst;
  unsigned int m_current_num_normal_mem_inst;

 protected:
  ldst_unit_sm(std::vector<register_set_uniptr*> result_ports, std::vector<register_set_uniptr*> reception_ports, mem_fetch_interface *icnt,
            mem_fetch_interface *icnt_L1C_L1_half_C, std::shared_ptr<shader_core_mem_fetch_allocator> mf_allocator, SM *core,
            const shader_core_config *config, // MOD. Fix WAR at baseline.
            const memory_config *mem_config, shader_core_stats *stats,
            unsigned sid, unsigned tpc, l1_cache *new_l1d_cache, unsigned int max_size_arbiter_to_subpipeline_reg_per_subcore);

  void init(mem_fetch_interface *icnt, mem_fetch_interface *icnt_L1C_L1_half_C,
            std::shared_ptr<shader_core_mem_fetch_allocator> mf_allocator,
            SM *core,
            const shader_core_config *config,
            const memory_config *mem_config, shader_core_stats *stats, unsigned sid, unsigned tpc);

  virtual mem_stage_stall_type process_cache_access(
      cache_t &cache, new_addr_type address, warp_inst_t &inst,
      std::list<cache_event> &events, mem_fetch *mf,
      enum cache_request_status status);
  mem_stage_stall_type process_memory_access_queue(cache_t &cache, mem_access_t *acc, bool is_const_cache);

  long double get_second_key_pending_writes(warp_inst_t *inst, int idx); // MOD. VPREG

  void global_shared_latency_queue_cycle();

  const memory_config *m_memory_config;
  mem_fetch_interface *m_icnt;
  mem_fetch_interface *m_icnt_L1C_L1_half_C;
  std::shared_ptr<shader_core_mem_fetch_allocator> m_mf_allocator;
  SM *m_core;
  unsigned m_sid;
  unsigned m_tpc;

  tex_cache *m_L1T;        // texture cache
  read_only_cache *m_L1C;  // constant cache
  l1_cache *m_L1D;         // data cache

  std::list<mem_fetch *> m_response_fifo;
  mem_fetch *m_next_global;
  unsigned m_num_writeback_clients;

  std::vector<enum mem_stage_stall_type> m_mem_rc_icnt_and_subcores;

  shader_core_stats *m_stats;

  // std::vector<std::deque<mem_fetch *>> l1d_latency_queue;
  std::vector<std::deque<std::shared_ptr<l1d_queue_element>>> l1d_latency_queue;
  std::deque<mem_fetch *> constant_cache_l1_latency_queue;
  
  void L1_constant_cache_latency_queue_cycle();

  void print_L1_constant_latency_queue(FILE *f);


  void L1_latency_queue_cycle();

  void print_L1_latency_queue(FILE *f); // MOD. VPREG

  void reset_is_this_l1d_bank_allocated_this_cycle();

  void cache_cycles();

  // Phases of cycle(), declared in the order cycle() runs them.
  void service_writeback_clients();
  void solve_missed_accesses_of_caches();
  void process_response_fifo();
  void dispatch_accesses_to_caches();
  void stage_l1d_accesses_through_tlb();
  void route_next_accesses_to_subpipelines();
  void refill_next_accesses_from_prt();
  void issue_incoming_memory_instructions();
  void update_interwarp_coalescing_warppool_policy();

  void shared_dispatch();
  void execute_miscellaneous_dispatch();
  void execute_cache_dispatch(AccessQueue *qu, cache_t *cache, std::function<mem_stage_stall_type(cache_t&, mem_access_t*)> func_process);  
  mem_stage_stall_type dispatch_to_memory_access_queue_l1Dcache(cache_t &cache, mem_access_t *acc);
  mem_stage_stall_type dispatch_to_memory_access_queue_l1Ccache(cache_t &cache, mem_access_t *acc);
  mem_stage_stall_type dispatch_to_memory_access_queue_l1Tcache(cache_t &cache, mem_access_t *acc);
  void dispatch_access_directly_to_l2();

  unsigned long long get_instruction_id(warp_inst_t* inst, unsigned int idx);

  void solve_next_missed_access(cache_t *cache,  bool is_constant);
  void pending_access_logic(std::vector<unsigned int> &prt_list);
  bool is_possible_to_push_to_wb_icnt(unsigned int icnt_id, bool is_ldgsts);
  void push_to_wb_icnt(warp_inst_t inst, unsigned int icnt_id);
  
  // These two queues does not have a limitited size in order to simplify the programmings
  // A queue for the icnt that has the pending movements to shared memory. Only used for LDGSTS
  std::unique_ptr<warp_inst_t> m_ldgsts_icnt_between_ldg_and_sts_part1;
  std::unique_ptr<warp_inst_t> m_ldgsts_icnt_between_ldg_and_sts_part2;
  // A queue per subcore that has the pending writebacks from this unit.
  std::vector<std::unique_ptr<warp_inst_t>> m_pending_wbs_per_subcore;

  std::vector<warp_inst_t> m_evaluating_wb_icnt_and_subcores;
  unsigned int m_max_size_arbiter_to_subpipeline_reg_for_icnt_and_subcores;

  std::vector<unsigned> m_writeback_arb_icnt_and_subcores;  // round-robin arbiter for writeback contention between L1T, L1C, shared for each subcore
  unsigned int m_writeback_arb_between_icnt_and_subcores;  // round-robin arbiter for writeback contention subcores
  unsigned int m_dispatch_subpipeline_arb_between_icnt_and_subcores;  // round-robin arbiter for dispatching instructions to subpipelines (shared, constant, texture l1D) between subcores

  unsigned int m_num_icnt_and_subcores_clients;
  unsigned int m_reserved_idx_icnt_to_shmem; // IDX of vectors structures devoted to the icnt to the shared memory. Used for the LDGSTS

  bool is_already_dispatched_to_shared_mem_this_cycle;
  bool is_already_dispatched_to_texture_mem_this_cycle;
  bool is_already_dispatched_to_constant_mem_this_cycle;

  unsigned int m_num_cycles_to_wait_to_issue_another_mem_inst_from_the_subcores;

  int m_num_reserved_associativity_currently_processing;

  std::vector<std::unique_ptr<warp_inst_t>> m_global_shared_latency_queue_for_ldgsts;

  coalescingAddressStats* m_coalescing_stats_l1d;
  coalescingAddressStats* m_coalescing_stats_const;
  coalescingAddressStats* m_coalescing_stats_sharedmem;

  PendingRequestTable* m_prt;

  // One access queue per l1d bank
  AccessQueue m_access_queue_to_l1c;
  AccessQueue m_access_queue_to_l1t;
  std::vector<AccessQueue*> m_access_queue_to_l1d_preTLB;
  std::vector<AccessQueue*> m_access_queue_to_l1d_postTLB;
  AccessQueue m_access_queue_to_shmem;
  AccessQueue m_access_queue_to_bypass_to_l2;
  AccessQueue m_access_queue_to_miscellaneous;
  std::queue<mem_access_t*> m_next_access_to_queue;
  std::vector<mem_access_t*> m_next_access_to_intercoalescing;
  InterWarpCoalescingUnit *m_intercoalescing_unit;
  

  mem_access_t **m_shmem_pipeline;
  register_set_uniptr m_ldgsts_aux = register_set_uniptr(1, "ldgsts_aux");
};

} // namespace remodel
