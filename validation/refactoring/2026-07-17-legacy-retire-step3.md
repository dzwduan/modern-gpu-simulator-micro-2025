# Legacy shader retirement — delete the legacy pipeline core

Branch: `refactor/legacy-shader-retirement`. Governing design:
`docs/plans/2026-07-17-legacy-retirement-design.md` §1a, §1c, §7 step 3, §9
rulings 1/2, §10. Predecessor ledger: `2026-07-17-legacy-retire-step2.md`.
Start HEAD `ff28e2a`.

Two commits. `SM` (the only core the trace path instantiates) is a sibling of
`shader_core_ctx` — both derive from `core_t` and `shader_core_ctx_wrapper` — so
deleting `shader_core_ctx` and its exclusive machinery does not touch `SM`.

## Environment

| Fact | Value |
| --- | --- |
| Build | `cd simulator-remodeled && source ./gpu-simulator/setup_environment_no_git.sh && make -j$(nproc) -C ./gpu-simulator/` |
| Gate | `python3 -m unittest discover -s tests` and `OMP_NUM_THREADS=1 python3 tests/remodeled_trace/run_regression.py check` (from repo root) |
| Binary | `simulator-remodeled/gpu-simulator/bin/release/accel-sim.out` |
| Final binary SHA-256 | `120cc4d9a0282170a3f1b1bcc342617078f0592a684a2eba22e794be5ee7feaf` |

## Commits

| Item | Hash | Subject |
| --- | --- | --- |
| 3a | `896cbf0` | refactor: delete legacy pipeline machinery and the dead shader_core_ctx core |
| 3b | `8b26c6e` | refactor: remove the dead LOOG result-bus interface surface |

## Gate table

Every commit was built and gated on its own state; the final HEAD was also
verified from a fully clean object tree (`make clean` then rebuild).

| State | build | unittest (count) | check (passed/total, observed_not_golden) |
| --- | ---: | --- | --- |
| baseline `ff28e2a` | 0 | 0 (21) | 0 (4/4, 0) |
| 3a `896cbf0` | 0 | 0 (21) | 0 (4/4, 0) |
| 3b `8b26c6e` | 0 | 0 (21) | 0 (4/4, 0) |
| HEAD clean rebuild | 0 | 0 (21) | 0 (4/4, 0) |

`run_regression.py check` reported `tracked_worktree_dirty: false` and
`source_commit 8b26c6e`; goldens are byte-identical (this is pure dead-code
deletion plus a golden-neutral semantic normalization — see 3b).

## Commit 3a — legacy machinery + `shader_core_ctx`

### What died

Classes (declarations in `gpgpu-sim/shader.h`, method bodies in `shader.cc` and
`gpu-sim.cc`):

- warp schedulers: `scheduler_unit`, `lrr_scheduler`, `rrr_scheduler`,
  `gto_scheduler`, `oldest_scheduler`, `two_level_active_scheduler`,
  `swl_scheduler` (incl. the tail inline `scheduler_unit::get_sid`);
- operand collector `opndcoll_rfu_t` and its nested classes;
- fixed-latency EX units `simd_function_unit`, `pipelined_simd_unit`, `sfu`,
  `dp_unit`, `tensor_core`, `int_unit`, `sp_unit`, `specialized_unit`;
- legacy `ldst_unit : public pipelined_simd_unit`;
- `struct insn_latency_info` (dead);
- `class shader_core_ctx` — the whole class + every `shader_core_ctx::` body in
  `shader.cc` and the seven `shader_core_ctx::` bodies in `gpu-sim.cc`
  (`mem_instruction_stats`, `can_issue_1block`, `find_available_hwtid`,
  `occupy_shader_resource_1block`, `release_shader_resource_1block`,
  `issue_block2core`, `dump_warp_state`).

Free functions / data / members:

- `register_bank(...)` and `coalesced_segment(...)` (both had no surviving
  caller after the cluster died);
- the `pipeline_stage_name_decode[]` string table (its only user was the
  deleted `shader_core_ctx::create_front_pipeline`);
- `shd_warp_t::m_scheduler` + `set_scheduler`/`get_scheduler`; the step-2
  `class scheduler_unit;` forward decl in `shader.h`; the stray
  `class scheduler_unit;` in `remodeling/ibuffer_remodeled.h`;
- the legacy `friend class` lines in `shader_core_stats`
  (`shader_core_ctx`, `ldst_unit`, `scheduler_unit`, and the phantom
  `TwoLevelScheduler`/`LooseRoundRobbinScheduler`); only
  `friend class simt_core_cluster;` (a KEEP class) was retained;
- the `ResultBus`/`ResultBusses` result-bus machinery: `result_bus.h` and
  `result_bus.cc` deleted, and the `#include "result_bus.h"` dropped from
  `shader.h`.

### KEEP verification

- `enum pipeline_stage_name_t` + `N_PIPELINE_STAGES` retained — they size
  `shader_core_config::pipe_widths[N_PIPELINE_STAGES]`; `pipe_widths` and its
  `-gpgpu_pipeline_widths` parse are untouched.
- `enum scheduler_prioritization_type` / `enum concrete_scheduler` retained
  (read by `gpgpu_sim_config::init`).
- All shared facilities intact: `thread_ctx_t`, `function_call_entry_info`,
  `shd_warp_t`, `barrier_set_t`, `ifetch_buffer_t`, `specialized_unit_params`,
  `shader_core_config`, `shader_core_stats(_pod)`,
  `shader_core_mem_fetch_allocator`, `simt_core_cluster`,
  `shader_memory_interface`, `perfect_memory_interface`, and the
  `simt_core_cluster/shader_memory_interface/shader_core_mem_fetch_allocator/cache_t`
  forward declarations.
- Post-delete structural check: 0 `class`/`struct` definitions of the deleted
  types remain in `shader.h`; the scoreboard-named `shader_core_stats` counters
  (`num_scheduler_stall_cycle_*`) were retained (they print into stdout and are
  a step-5 concern, not step-3).

### External-breakage fixes (the cascade)

1. **`check_kernel_launch_limitation` retained, not deleted — design correction.**
   The design (§1a, §9 ruling 1, R7) states its sole caller is the legacy
   `shader_core_ctx::issue_block2core` and instructs deletion. Verified against
   live code this is **wrong**: the sole caller is the KEPT
   `simt_core_cluster::issue_block2core()` (`shader.cc`, immediately after
   `m_core[core]->issue_block2core(*kernel)`), and the legacy
   `shader_core_ctx::issue_block2core` in `gpu-sim.cc` never called it. The
   function updates live stats (`total_number_of_kernels_limited_by_*`) on every
   block issue on the supported path. It was restored verbatim (declaration in
   `shader.h`, definition re-inserted in `shader.cc`). Deleting it would have
   broken the link or altered a KEEP method's behavior; retaining it is
   golden-neutral. `coalesced_segment` and `register_bank`, by contrast, had
   zero surviving callers and were correctly deleted.

2. **Stray `class shader_core_ctx;` forward decl removed from
   `remodeling/l0_icnt.h`** (parallel to the `ibuffer_remodeled.h`
   `scheduler_unit` decl the design named). `l0_icnt.h` uses only
   `shader_core_ctx_wrapper`, so the decl was dead; leaving a forward decl of a
   deleted class is misleading. Zero logic change.

3. **Orphaned `result_bus.o` removed.** The link globs `build/.../*.o`; after
   `result_bus.cc` was deleted its stale object still carried an undefined
   `register_bank` reference (link error). Removing the orphaned object cleared
   it — the standard hygiene step for deleted sources recorded in
   `2026-07-17-dead-weight.md`.

4. **`shader.cc` remodeling includes trimmed.** With `shader_core_ctx` gone,
   `shader.cc` no longer names `SM`, `L0_icnt`, `num_bytes_cache_req`, or
   `Element_stats`; the `remodeling/{sm.h,new_stats.h,l0_icnt.h}` includes were
   dropped (build stays green). `remodeling/{ibuffer_remodeled.h,
   warp_dependency_state.h}` were kept — the `shd_warp_t` ctor/dtor `new`/`delete`
   `IBuffer_Remodeled`/`Dependency_State`, so `shader.cc` stays on the ledger by
   design (deferred to stage 4).

### Deviation from the commit split (build-forced)

`result_bus.{h,cc}` were deleted in **3a**, not 3b. They are entangled with the
3a cluster: `result_bus.cc` needs `opndcoll_rfu_t`'s full definition
(`m_rf->shader_core()`) and `register_bank`, while `shader_core_ctx` embeds a
`ResultBusses` member. Splitting them across two green commits is impossible
(deleting either half first dangles the other). 3b therefore handles only the
LOOG *interface* surface, which is cleanly separable (in the wrapper, `SM`,
`abstract_hardware_model`, and `ldst_unit_sm`, none entangled with the operand
collector).

## Commit 3b — LOOG / RRS interface surface

Semantic normalization: the remodeled pending-write first key is always the
warp id. Deleted:

- `shader_core_wrapper.h`: the `get_loog_rrs()` and `get_is_loog_enabled()`
  pure virtuals and the `class RRS;` forward declaration;
- `remodeling/sm.h` + `remodeling/sm.cc`: the `SM::get_is_loog_enabled` (read an
  unregistered flag) and `SM::get_loog_rrs` (threw) overrides;
- `shader.h`: the `shader_core_config::is_loog_enabled` member;
- `abstract_hardware_model.h`: the `rrs_id_type` typedef, the `warp_inst_t`
  `m_cu_rrs_id` member, and its `= -1` initializer;
- `remodeling/ldst_unit_sm.{h,cc}`: the uncalled
  `ldst_unit_sm::get_first_key_pending_writes` (returned `m_cu_rrs_id` only under
  the never-true LOOG branch, else `warp_id()`).

Golden-neutrality evidence:

- `is_loog_enabled` is **unregistered** — `grep is_loog_enabled gpu-sim-config.cc`
  is empty, and it appears in **no** shipped config (`grep -rl is_loog_enabled
  configs/` is empty). So there is no config-file change and no golden
  re-approval; the member read was indeterminate and only ever reached through
  the dead helper.
- `m_cu_rrs_id` was only ever initialized to `-1` and read once, inside the
  uncalled helper; the surviving key is `warp_id()` on every path.
- Post-delete `grep` for `RRS|get_loog_rrs|get_is_loog_enabled|is_loog_enabled|rrs_id_type|m_cu_rrs_id|get_first_key_pending_writes`
  over `gpgpu-sim/src` + `trace-driven` is empty.
- `check` remained 4/4 byte-identical after 3b.

Out of scope (left untouched per the design): the `loog_frontend_size` /
`loog_rrs_size` / `loog_memory_queues_size` config members and the
`warp_inst_t::m_loog_queue_idx_entry` member — the design named only
`is_loog_enabled` and `m_cu_rrs_id`. `is_improved_result_bus` (a registered
config option, now set-but-unused) was also retained to avoid an out-of-scope
config-hash change.

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

6 inbound files — **unchanged** from the step-2 end state. Honest accounting:
`shader.cc` **REMAINS** on the ledger by design (its `shd_warp_t` ctor/dtor
`new`/`delete` `IBuffer_Remodeled`/`Dependency_State`, so it keeps
`remodeling/{ibuffer_remodeled.h,warp_dependency_state.h}`); it did not reach
zero. Step 3 reduced `shader.cc`'s remodeling includes from 5 to 2 but did not
change its ledger membership. The design's step-3 prediction ("shader.cc reverse-
include removed") assumed the original step-2 plan (a new `shd_warp.cc`); the
actual step 2 folded the ctor/dtor into `shader.cc` instead, so `shader.cc`'s
severance is deferred to stage 4 with the rest of the god-file split.

Of the 6, the changes this step touched: `shader.cc` (3a, includes trimmed),
`shader_core_wrapper.h` (3b, LOOG virtuals removed — still includes
`remodeling/new_stats.h` for `Element_stats`, the sanctioned L3→L2 seam),
`abstract_hardware_model.cc` (unchanged this step — only `abstract_hardware_model.h`
was edited, and its `functional_unit` reference count stays 4). `gpu-sim.h`,
`scoreboard.cc`, `scoreboard_reads.cc` untouched (scoreboard removal is step 5).

`abstract_hardware_model.h` `functional_unit` reference count: **4** (unchanged).

R9 outbound edge (remodeling → trace-driven, tracked separately, unchanged):
`remodeling/{ibuffer_remodeled.cc,subcore.cc,sm.cc}` include
`../../../../trace-driven/trace_driven.h` (3 active; `sm.h` has it commented).
Step 3 neither added nor removed this L2→L4 edge; it remains deferred to stage 4.

## Line-count delta (`ff28e2a` -> HEAD)

| File | Baseline | HEAD | Delta |
| --- | ---: | ---: | ---: |
| shader.h | 3652 | 1919 | -1733 |
| shader.cc | 4812 | 2109 | -2703 |
| gpu-sim.cc | 1659 | 1387 | -272 |
| shader_core_wrapper.h | 163 | 160 | -3 |
| abstract_hardware_model.h | 2172 | 2169 | -3 |
| remodeling/sm.cc | 1978 | 1972 | -6 |
| remodeling/sm.h | 404 | 402 | -2 |
| remodeling/ldst_unit_sm.cc | 2080 | 2072 | -8 |
| remodeling/ldst_unit_sm.h | 392 | 391 | -1 |
| remodeling/ibuffer_remodeled.h | — | — | -2 |
| remodeling/l0_icnt.h | — | — | -1 |
| result_bus.h | 67 | (deleted) | -67 |
| result_bus.cc | 118 | (deleted) | -118 |

`git diff --stat ff28e2a HEAD`: 13 files changed, 87 insertions(+),
5007 deletions(-).

## Independent rerun

An independent read-only agent reran the full gate from a clean object tree
(`make clean` + rebuild, unittest, check), modifying no source and committing
nothing. It reported:

- HEAD `8b26c6ee0e509d4f0556babb909ece1257ac4d1d`; worktree clean of tracked
  changes (only the pre-existing untracked `.codegraph/`);
- `BUILD_EXIT 0`, `UNITTEST_EXIT 0` (21 tests, OK), `CHECK_EXIT 0`;
- regression summary passed 4, failed 0, total 4, observed_not_golden 0,
  `tracked_worktree_dirty false`, `source_commit 8b26c6e`.

All three gates passed independently; goldens byte-identical.

## Structural spot-check

- Post-3a `shader.cc` retains every KEEP method family and no deleted-class
  bodies: `shader_core_mem_fetch_allocator::` 2, `shader_core_stats::` 12,
  `barrier_set_t::` 7, `shd_warp_t::` 8, `simt_core_cluster::` 25,
  `shader_core_config::` 2, `warp_inst_t::print` 1, `gpgpu_sim::shader_print*` 4,
  `check_kernel_launch_limitation` present; grep for
  `(scheduler_unit|opndcoll_rfu_t|pipelined_simd_unit|simd_function_unit|ldst_unit|shader_core_ctx|sfu|tensor_core|sp_unit|dp_unit|int_unit)::`
  in `shader.cc` = 0.
- `abstract_hardware_model.cc` had no changes this step (only the header).

## Review response

The independent review (gpt-5.6-sol) of `ff28e2a..2c09793` found the runtime
deletion internally consistent on clean builds, with one P1 about incremental
builds:

- The parent link (`gpgpu-sim/Makefile`) globs `$(SIM_OBJ_FILES_DIR)/gpgpu-sim/*.o`,
  so after `result_bus.cc` was deleted, a stale `result_bus.o` left in a
  previously-built object tree would still be linked and would drag in the
  removed `register_bank` symbol, failing the link. The executor's manual
  `rm result_bus.o` fixed only the local (gitignored) object tree and was not
  part of the change, so a fresh incremental build over `ff28e2a` would fail.

Fix (committed): `src/gpgpu-sim/Makefile` gains a `prune-orphan-objects`
prerequisite of `all` that removes any `$(OUTPUT_DIR)/*.o` whose source `.cc`
no longer exists, before the objects are (re)built and the parent glob-links
them. This makes incremental builds robust to source deletion in this
directory — the same pattern that will recur as later stages delete sources.

Verification of the reviewer's exact scenario: a stale `result_bus.o` was
planted in the object tree (copied from another object so it references the
removed symbol), then a normal `make` was run. Output showed
`Pruning orphan object .../result_bus.o`, the object was removed, and the build
linked with exit 0 (no `undefined reference to register_bank`). Post-fix gate:
unit tests exit 0 (21 OK); regression check exit 0, 4/4 passed, goldens
unchanged.
