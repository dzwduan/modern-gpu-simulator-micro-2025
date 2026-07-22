# Stage four S1 — wrap the remodeling SM model in `namespace remodel`

Branch: `dev_dzw`. Governing design:
`docs/plans/2026-07-22-stage4-reorganization-design.md` §3 (goals), §4 S1.
Start HEAD `4f90d02`. Pure structure-preserving change: symbol-scoping only, no
timing/statistic semantics touched, goldens byte-identical.

The original structure review flagged that the remodeling SM model lived at global
scope (missing namespace) and exported global free functions (ODR risk). S1 moves
every remodeling class, struct, enum, and free function into `namespace remodel`,
and qualifies the external consumers. No renames, no file splits, no logic edits,
no config changes.

## Environment

| Fact | Value |
| --- | --- |
| Build | `cd simulator-remodeled && source ./gpu-simulator/setup_environment_no_git.sh && make -j$(nproc) -C ./gpu-simulator/` |
| Gate | `python3 -m unittest discover -s tests` (21 tests) and `OMP_NUM_THREADS=1 python3 tests/remodeled_trace/run_regression.py check` (from repo root) |
| Binary | `simulator-remodeled/gpu-simulator/bin/release/accel-sim.out` |

## Commit

| Hash | Subject |
| --- | --- |
| `c2630acdebe0a9ab73694db8f3cbc7bb60b23f27` | refactor: scope the remodeling SM model in a namespace |

## Gate table (clean rebuild — `make clean` + full build)

| build | unittest (count) | check (passed/total, observed_not_golden) | goldens |
| ---: | --- | --- | --- |
| 0 | 0 (21) | 0 (4/4, 0) | byte-identical, unchanged |

`check` reported `passed 4, failed 0, total 4, observed_not_golden 0`; all four
cases `status: passed` with empty `golden_mismatches` and `golden_set_status:
approved`. No `goldens.json` edit and no config `sha256` change (config hashes
identical to HEAD `4f90d02`), so no golden re-approval was required.

## Files wrapped in `namespace remodel { … }` (23)

`namespace remodel {` opens after the pragma-once/includes (and after any
non-remodeling forward declarations, see below); `} // namespace remodel` closes at
EOF. Headers: `sm.h`, `subcore.h`, `ldst_unit_sm.h`, `functional_unit.h`,
`register_file.h`, `ibuffer_remodeled.h`, `l0_icnt.h`, `stream_buffer.h`,
`first_level_instruction_cache.h`, `warp_dependency_state.h`, `new_stats.h`,
`fusedMemory/coalescingStats.h`. Sources: the paired `.cc` for each of the above
(`sm.cc`, `subcore.cc`, `ldst_unit_sm.cc`, `functional_unit.cc`, `register_file.cc`,
`ibuffer_remodeled.cc`, `l0_icnt.cc`, `stream_buffer.cc`,
`first_level_instruction_cache.cc`, `warp_dependency_state.cc`,
`fusedMemory/coalescingStats.cc`).

## Free functions moved into the namespace (7 declared + 1 file-local)

- `l0_icnt.{h,cc}`: `num_bytes_cache_req`, `get_pc_of_request`.
- `sm.{h,cc}`: `translate_warp_id_of_sm_to_subcore`, `get_reg_type_eval`,
  `check_is_reserved_regs_remodeling`, `translate_reg_to_global_id`.
- `ldst_unit_sm.{h,cc}`: `calculate_constant_address`.
- `functional_unit.cc`: the file-local helper `find_next_stage_index` (defined and
  used only in that translation unit) is also inside the namespace — verified it has
  no cross-TU caller.

## External consumers qualified (7 files)

1. `gpgpu-sim/src/abstract_hardware_model.h` — `class functional_unit;` →
   `namespace remodel { class functional_unit; }`; the `warp_inst_t` members
   `m_fu_assigned`, `set_fu_assigned`, `get_fu_assigned` → `remodel::functional_unit`.
2. `gpgpu-sim/src/gpgpu-sim/shader.h` — the four forward decls
   (`coalescingStatsAcrossSms`, `Subcore`, `IBuffer_Remodeled`, `Dependency_State`)
   wrapped in `namespace remodel { … }`; the `shd_warp_t` members/accessors
   (`m_IBuffer_remodeled`, `m_dependency_state`, `m_subcore`, `get_IBuffer_remodeled`,
   `get_dependency_state`) and the `simt_core_cluster` stats members/methods
   (`gather_stats`, `gather_single_stat`, `create_gpu_per_cluster_stats`,
   `m_cluster_stats`) → `remodel::` (`Element_stats` / `coalescingStatsAcrossSms` /
   `Subcore` / `IBuffer_Remodeled` / `Dependency_State`).
3. `gpgpu-sim/src/gpgpu-sim/shader.cc` — the `shd_warp_t` constructor allocations
   `new IBuffer_Remodeled(...)` and `new Dependency_State(...)` → `new remodel::…`.
4. `gpgpu-sim/src/gpgpu-sim/shader_core_wrapper.h` — `class coalescingStatsAcrossSms;`
   wrapped; the three pure-virtual stats method signatures
   (`create_gpu_per_sm_stats`, `gather_gpu_per_sm_stats`,
   `gather_gpu_per_sm_single_stat`) → `remodel::Element_stats` /
   `remodel::coalescingStatsAcrossSms`.
5. `gpgpu-sim/src/gpgpu-sim/gpu-sim.h` — `m_gpu_per_sm_stats` (`Element_stats`) and the
   three `m_coalescing_stats_across_sms_*` members → `remodel::`.
6. `gpgpu-sim/src/gpgpu-sim/gpu-sim.cc` — a function-local `using remodel::AllowedTypesStats;`
   in `gpgpu_sim::create_gpu_per_sm_stats` covers the enum references there (the only
   unqualified remodeling name in the file; the stats are member-accessed on
   `m_gpu_per_sm_stats`, needing no qualification). No `using` was added to any header.
7. `trace-driven/trace_driven.cc` — `new SM(...)` in the cluster factory →
   `new remodel::SM(...)`; the two `calculate_constant_address(...)` calls →
   `remodel::calculate_constant_address(...)`.

The other named files (`main.cc`, `gpgpu-sim/kernel-scheduler.cc`,
`gpgpu-sim/icnt-handler.cc`, `gpgpu-sim/gpu-cache.*`, `gpgpu-sim/l2cache.*`,
`gpgpu-sim/gpu-sim-config.cc`) were grepped and reference no remodeling symbol
directly, so needed no edit — confirmed by the clean build.

## Unusual handling (forward declarations / ADL)

- **Non-remodeling forward decls stay at global scope.** Inside `namespace remodel`,
  an unqualified reference to a global type (e.g. `shader_core_config`, `warp_inst_t`,
  `register_set_uniptr`, `read_only_cache`) resolves to the global one only if no
  `remodel::` shadow is declared. So the namespace opens *after* the block of
  non-remodeling forward decls in each header; the remodeling forward decls are inside.
- **Two reorder cases.** In `functional_unit.h` the non-remodeling
  `class register_set_uniptr;`, and in `register_file.h` the non-remodeling
  `class warp_inst_t;` / `class traced_operand;`, trailed the remodeling forward
  decls. They were moved above the `namespace remodel {` boundary so they remain
  global forward declarations (declaration set unchanged; purely relocated within the
  same forward-decl block).
- **External-header forward decls re-scoped.** `abstract_hardware_model.h`,
  `shader.h`, and `shader_core_wrapper.h` forward-declare remodeling types; those
  decls were placed inside `namespace remodel { … }` so they name the correct
  `remodel::` types (the `warp_inst_t::m_fu_assigned` pointer and the cluster stats
  references then bind to `remodel::functional_unit` / `remodel::Element_stats` /
  `remodel::coalescingStatsAcrossSms`).
- **ADL.** `calculate_constant_address(uint64_t, traced_operand&)` takes a global
  util type, so argument-dependent lookup does not reach `namespace remodel`; the
  external caller in `trace_driven.cc` therefore needs the explicit `remodel::`
  qualifier (added). Remodeling-internal callers are inside the namespace and resolve
  unqualified.
- No `friend` declarations naming remodeling classes and no explicit `::Type` global
  qualifiers of remodeling types exist in the remodeling tree (grepped), so neither
  required extra handling.

## Reverse-include ledger — unchanged at 4

S1 adds **zero** `#include` lines (verified: the diff contains no added `#include`),
so include topology is untouched and the ledger is unchanged. The four low→high
reverse edges (grep `remodeling/.*\.h` in the non-remodeling tree):

- `abstract_hardware_model.cc` → `remodeling/register_file.h` (L0→L2, real violation,
  pending S5).
- `shader.cc` → `remodeling/ibuffer_remodeled.h` + `remodeling/warp_dependency_state.h`
  (L1→L2, real violation, pending S5).
- `gpu-sim.h` → `remodeling/new_stats.h` + `remodeling/fusedMemory/coalescingStats.h`
  (L3→L2, sanctioned).
- `shader_core_wrapper.h` → `remodeling/new_stats.h` (L3↔L2, sanctioned).

Namespacing does not change these edges; the two real violations remain the S5
target.

## Change size

30 files, +142 / -47: 23 remodeling files (namespace open/close, two forward-decl
reorders) plus the 7 external consumers. The trailing-newline normalization on a few
remodeling files that previously lacked a final newline is incidental and touches no
golden.

## Independent rerun

Build and gate were executed in-session on a **clean object tree** (`make clean` +
full `-j` rebuild, no `--gc-sections`, so any missed qualification would have failed
the link loudly): build exit `0`, unit tests exit `0` (21 tests), regression `check`
exit `0` (`{"passed": 4, "failed": 0, "total": 4, "observed_not_golden": 0}`, all
cases `passed`, no `golden_mismatches`). A fully independent read-only agent rerun is
delegated to the coordinator's pre-push verification and review step per the stage-four
execution model; this record binds the in-session clean-rebuild result to the commit
above.
