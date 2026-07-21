# Legacy shader retirement — shrink the shader_core_ctx_wrapper interface

Branch: `refactor/legacy-shader-retirement`. Governing design:
`docs/plans/2026-07-17-legacy-retirement-design.md` §2 (Option B), §7 step 4,
§10. Predecessor ledger: `2026-07-17-legacy-retire-step3.md`.
Start HEAD `6e58020`.

One commit. `SM` is the sole implementor of `shader_core_ctx_wrapper`; the
remodeling internals hold concrete `SM*` (`functional_unit::m_sm`,
`Subcore::m_sm`, `ldst_unit_sm::m_core`, and the `SM*` returned by
`get_SM()`/`get_sm()`) and call methods on `SM` directly, never through the
wrapper. Only `simt_core_cluster::m_core`, `shader_memory_interface::m_core`,
`perfect_memory_interface::m_core`, `barrier_set_t::m_shader`,
`L0_icnt::m_shader`, and `shd_warp_t::m_shader`/`get_shader()` are wrapper-typed.
The interface is shrunk to exactly the methods those pointers invoke; every other
virtual was dead interface. This is interface-only surgery with zero behavior
change.

## Environment

| Fact | Value |
| --- | --- |
| Build | `cd simulator-remodeled && source ./gpu-simulator/setup_environment_no_git.sh && make -j$(nproc) -C ./gpu-simulator/` |
| Gate | `python3 -m unittest discover -s tests` and `OMP_NUM_THREADS=1 python3 tests/remodeled_trace/run_regression.py check` (from repo root) |
| Binary | `simulator-remodeled/gpu-simulator/bin/release/accel-sim.out` |
| Binary SHA-256 (clean rebuild) | `607f6c121486a9b1f55d26e54fce1e911715332c16748160e2db7523cf35c923` |

Binary hashes are build-specific (an incremental build over the pre-change tree
produced a different hash, `d6e39a12…`); they are not a reproducible oracle. The
byte-identical goldens (`observed_not_golden 0`) are the behavioral oracle.

## Virtual count

| | Pure virtuals (`= 0;`) | Total `virtual` (incl. dtor) |
| --- | ---: | ---: |
| Before (`6e58020`) | 88 | 89 |
| After | 44 | 45 |
| Removed | 44 | 44 |

## Gate table

| State | build | unittest (count) | check (passed/total, observed_not_golden) |
| --- | ---: | --- | --- |
| after shrink | 0 | 0 (21) | 0 (4/4, 0) |

`run_regression.py check` reported `source_commit 6e58020`, `passed 4, failed 0,
total 4, observed_not_golden 0`; `tracked_worktree_dirty` was `true` only because
the check ran against the uncommitted edits. Goldens are byte-identical — this is
pure dead-interface removal, no simulated behavior change.

## Method (empirical keep-set derivation)

1. Listed all 88 pure virtuals (`grep -cE '= 0;' shader_core_wrapper.h`).
2. Enumerated every `shader_core_ctx_wrapper*` holder tree-wide; the only ones are
   the six named above. Collected the method names invoked through each
   (`m_core[..]->`, `m_core->`, `m_shader->`, `get_shader()->`) across `shader.cc`,
   `shader.h`, `trace-driven/trace_driven.{cc,h}`, `remodeling/l0_icnt.{cc,h}`.
3. For each candidate that did not appear, confirmed by grep that its only call
   sites use a concrete `SM*` receiver (verified `ldst_unit_sm::m_core` is
   `SM *m_core`, `get_SM()`/`get_sm()` return `SM*`, `m_sm`/`shared_sm`/
   `m_shared_sm` are `SM*`), a differently-signed `shd_warp_t` method, or have no
   caller at all.
4. The build is the authority: an over-removal would fail the link with
   `no member named X in 'shader_core_ctx_wrapper'`. The build was green on the
   first attempt, so no method had to be re-added.

## KEEP set — 44 pure virtuals (wrapper-called)

Grouped by the wrapper-typed pointer that invokes them (some methods have more
than one caller; listed once under a primary caller):

- `simt_core_cluster::m_core[..]` (and the trace cluster) — 32:
  `cycle`, `init`, `reinit`, `cache_flush`, `cache_invalidate`,
  `accept_fetch_response`, `accept_ldst_unit_response`,
  `fetch_unit_response_buffer_full`, `ldst_unit_response_buffer_full`,
  `set_kernel`, `get_kernel`, `get_pdom_stack_top_info` (3-arg),
  `get_n_active_cta`, `get_not_completed`, `isactive`, `get_current_occupancy`,
  `issue_block2core`, `can_issue_1block`, `display_pipeline`, `print_cache_stats`,
  `get_cache_stats`, `get_L0I_sub_stats`, `get_L1I_sub_stats`, `get_L1D_sub_stats`,
  `get_L1C_sub_stats`, `get_L1T_sub_stats`, `get_icnt_power_stats`,
  `increment_sm_stat_by_integer`, `create_gpu_per_sm_stats`,
  `reset_cycless_access_history`, `gather_gpu_per_sm_stats`,
  `gather_gpu_per_sm_single_stat`.
- `shader_memory_interface::m_core` / `perfect_memory_interface::m_core` — 1:
  `inc_simt_to_mem`.
- `L0_icnt::m_shader` — 2: `get_num_subcores`,
  `set_subcore_req_fetch_L1I_priority`.
- `shd_warp_t` / `barrier_set_t` via `m_shader` / `get_shader()` — 9: `get_gpu`,
  `get_config`, `num_cycles_to_stall_SM`, `get_kernel_info`, `ptx_thread_done`,
  `warp_waiting_at_mem_barrier`, `warp_waiting_at_barrier`,
  `warp_waiting_grid_barrier`, `broadcast_barrier_reduction`.

## REMOVED set — 44 pure virtuals (never called through the wrapper)

- Per-op stat family (32): `incload_stat`, `incstore_stat`, `incialu_stat`,
  `incimul_stat`, `incimul24_stat`, `incimul32_stat`, `incidiv_stat`,
  `incfpalu_stat`, `incfpmul_stat`, `incfpdiv_stat`, `incdpalu_stat`,
  `incdpmul_stat`, `incdpdiv_stat`, `incsqrt_stat`, `inclog_stat`, `incexp_stat`,
  `incsin_stat`, `inctensor_stat`, `inctex_stat`, `inc_const_accesses`,
  `incsfu_stat`, `incsp_stat`, `incmem_stat`, `incregfile_reads`,
  `incregfile_writes`, `incnon_rf_operands`, `incspactivelanes_stat`,
  `incsfuactivelanes_stat`, `incfuactivelanes_stat`, `incfumemactivelanes_stat`,
  `incexecstat`, `mem_instruction_stats`.
  (`SM::incexecstat` calls the other `inc*_stat` methods on `this`; `ldst_unit_sm`
  calls `m_core->mem_instruction_stats` on a concrete `SM*`. All internal to SM.)
- Accessors / other (12): `get_stats`, `get_sid`, `get_shd_warp`,
  `warp_inst_complete`, `get_pdom_stack_top_info` (4-arg
  `(warp_id, const warp_inst_t*, pc, rpc)`), `decrement_atomic_count`,
  `store_ack`, `inc_store_req`, `dec_inst_in_pipeline`, `get_current_gpu_cycle`,
  `from_local_pc_to_global_pc_address`, `from_global_pc_address_to_local_pc`.

Each removed method survives on `SM` (`remodeling/sm.h` + `sm.cc`) as SM's own
member — SM still uses it internally or is called through a concrete `SM*`.

## Correction to the coordinator keep-set

The coordinator flagged 15 methods as "verify-and-likely-keep". Verification split
them: 3 are genuinely wrapper-called and KEPT
(`gather_gpu_per_sm_stats`, `gather_gpu_per_sm_single_stat`,
`reset_cycless_access_history` — all `m_core[..]` cluster calls). The other 12 are
reached only through a concrete `SM*`, a differently-signed `shd_warp_t` method,
or have no caller, and were **REMOVED**: `get_shd_warp`, `get_stats`, `get_sid`,
`warp_inst_complete`, `store_ack`, `decrement_atomic_count`,
`dec_inst_in_pipeline`, `get_current_gpu_cycle`,
`from_global_pc_address_to_local_pc`, `from_local_pc_to_global_pc_address`,
`inc_store_req`, `inc_const_accesses`. Evidence per method:

- `get_shd_warp` — `m_sm->get_shd_warp(..)` in `functional_unit.cc`/`subcore.cc`
  (`m_sm` is `SM*`).
- `get_stats` — only `get_SM()->get_stats()` (SM*) and cache-object `get_stats()`.
- `get_sid` — `shared_sm->get_sid()`/`get_SM()->get_sid()` (SM*), `mf->get_sid()`.
- `warp_inst_complete`, `inc_const_accesses` — no call site anywhere.
- `store_ack`, `decrement_atomic_count`, `inc_store_req`, `mem_instruction_stats`,
  `get_current_gpu_cycle` — `m_core->..` in `ldst_unit_sm.cc` where
  `ldst_unit_sm::m_core` is `SM *`.
- `dec_inst_in_pipeline` — the wrapper form takes `warp_id`; the only calls are
  the no-arg `shd_warp_t::dec_inst_in_pipeline()`, a different method.
- `from_global_pc_address_to_local_pc`, `from_local_pc_to_global_pc_address` —
  `shared_sm->..`/`m_shared_sm->..` (SM*).
- `get_pdom_stack_top_info` 4-arg — the only wrapper-pointer call is the 3-arg
  `m_core[cid]->get_pdom_stack_top_info(tid, pc, rpc)`; the 4-arg
  `(warp_id, warp_inst_t*, pc, rpc)` overload has no wrapper caller.

No method had to be re-added after the build.

## `override` strips on SM (`remodeling/sm.h`)

A method declared `override` that overrides nothing is a compile error. Of the 44
removed virtuals, 12 carried an `override` on their `SM` declaration (confirmed
none of these is declared in `core_t`/`abstract_hardware_model.h`, so removing the
wrapper declaration left them overriding nothing). The `override` specifier was
stripped from each; the method itself stays:

`get_current_gpu_cycle`, `get_sid`, `get_stats`, `get_shd_warp`,
`get_pdom_stack_top_info` (4-arg), `warp_inst_complete`, `dec_inst_in_pipeline`,
`store_ack`, `inc_store_req`, `mem_instruction_stats`,
`from_local_pc_to_global_pc_address`, `from_global_pc_address_to_local_pc`.

The 3-arg `get_pdom_stack_top_info` (KEEP) retains its `override`. The other 32
removed methods' `SM` declarations were already plain (no `override`) —
`decrement_atomic_count`, `inc_const_accesses`, `incexecstat`, and the whole
`inc*_stat` family except `mem_instruction_stats`.

## Reverse-include ledger

Run in `gpgpu-sim/src`:
`grep -rln '#include.*remodeling/' --include=*.cc --include=*.h . | grep -v '/remodeling/' | sort`

```
abstract_hardware_model.cc
gpgpu-sim/gpu-sim.h
gpgpu-sim/scoreboard.cc
gpgpu-sim/scoreboard_reads.cc
gpgpu-sim/shader.cc
gpgpu-sim/shader_core_wrapper.h
```

6 inbound files — **unchanged** from the step-3 end state.
`shader_core_wrapper.h` stays on the ledger via its sanctioned
`#include "remodeling/new_stats.h"` seam (the `Element_stats` methods
`create_gpu_per_sm_stats`/`gather_gpu_per_sm_stats`/
`gather_gpu_per_sm_single_stat`, all KEPT). Re-typing `Element_stats` is
stage-4-reorg work and was not attempted; the include is retained by design.

## Diff scope

`git diff --stat` (two files):

```
 remodeling/sm.h               | 24 +++++------
 shader_core_wrapper.h         | 47 ----------------------
 2 files changed, 12 insertions(+), 59 deletions(-)
```

No config files, no golden files, no behavior-bearing source touched.

## Independent rerun

An independent read-only agent rebuilt from a clean object tree
(`make clean` + rebuild) and reran the full gate on the committed shrink,
modifying no source and committing nothing. It reported:

- HEAD `13e59045e0a652f7ae52e99d1d3fb30ef04ad444`; tracked tree clean (only the
  pre-existing untracked `.codegraph/`);
- `BUILD_EXIT 0` (binary produced), `UNITTEST_EXIT 0` (21 tests, OK),
  `CHECK_EXIT 0`;
- regression `summary` `passed 4, failed 0, total 4, observed_not_golden 0`;
  `source` `commit 13e5904…, tracked_worktree_dirty false`; all 4 cases
  `status: passed`, `golden_set_status: approved`, no mismatches;
- clean-rebuild binary SHA-256
  `607f6c121486a9b1f55d26e54fce1e911715332c16748160e2db7523cf35c923`, matching the
  regression report's internal `inputs.binary.sha256` cross-check.

All three gates passed independently from a clean object tree; goldens
byte-identical.

(This section was written after the shrink commit and folded into it by amend;
the amend touched only this validation record — the `sm.h`/`shader_core_wrapper.h`
source verified above is unchanged by it.)
