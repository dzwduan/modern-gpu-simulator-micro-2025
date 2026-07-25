# Stage 4 S7 — spelling, leftover comments and magic numbers

Branch: `dev_dzw`. Governing design:
`docs/plans/2026-07-22-stage4-reorganization-design.md` §4 S7. Structure-preserving
cleanup only: no timing algorithm, statistic semantic or interface behavior
changes. Goldens byte-identical throughout; a single controlled re-approval
covers the config-hash change caused by renaming four option names.

## Environment

| Fact | Value |
| --- | --- |
| Start HEAD | `5a137f6` (branch `dev_dzw`) |
| End HEAD | `2f89f2c` |
| Build | `cd simulator-remodeled && source ./gpu-simulator/setup_environment_no_git.sh && make -j$(nproc) -C ./gpu-simulator/` |
| Gate (from repo root) | `python3 -m unittest discover -s tests` and `OMP_NUM_THREADS=1 python3 tests/remodeled_trace/run_regression.py check` |
| Binary | `simulator-remodeled/gpu-simulator/bin/release/accel-sim.out` |
| Asserts | active (no `-DNDEBUG` in any Makefile), so the new bounds assert is live |

## Commits

| # | Hash | Subject |
| --- | --- | --- |
| 1 | `0d357ce` | refactor: correct misspelled identifiers in the remodeled core |
| 2 | `cb1c0c5` | refactor: correct the misspelled config option names |
| 3 | `2f89f2c` | refactor: translate leftover comments and name the remodeling constants |

The split is required for isolation: commit 1 is a pure C++ symbol rename gated
against **unchanged** goldens, so the only commit that touches goldens is
commit 2, whose entire delta is two config `sha256` values.

## Gate table

| State | build | unittest (count) | check (passed/total) | goldens |
| --- | ---: | --- | --- | --- |
| baseline `5a137f6` | 0 | 0 (21) | 0 (4/4), observed_not_golden 0 | unchanged |
| commit 1 `0d357ce` | 0 | 0 (21) | 0 (4/4), observed_not_golden 0 | UNCHANGED (isolation proof) |
| commit 2 `cb1c0c5` | 0 | 0 (21) | 0 (4/4), observed_not_golden 0 | re-approved (config sha256 only) |
| commit 3 `2f89f2c` | 0 | 0 (21) | 0 (4/4), observed_not_golden 0 | unchanged |
| final rerun on committed `2f89f2c` | 0 | 0 (21) | 0 (4/4), observed_not_golden 0 | unchanged |

`check` reports `tracked_worktree_dirty: true` throughout because of a
pre-existing uncommitted `.gitignore` edit that belongs to another change; it is
left unstaged and is not part of any commit here.

## Commit 1 — C++ identifier spelling

Eight misspelled word families, eleven distinct symbols. No option string
literal and no `.config` file is touched by this commit; a C++ member and its
option name are independent, so the member can be corrected while the option
name waits for the sweep in commit 2.

| Old spelling | New spelling | Kind | Files |
| --- | --- | --- | --- |
| `is_subcore_with_problems_of_fordward_progress` | `..._forward_progress` | `Subcore` method | `gpgpu-sim/remodeling/subcore.{h,cc}`, `gpgpu-sim/remodeling/sm.cc` |
| `is_any_subcore_problems_of_fordward_progress` | `..._forward_progress` | `SM` method | `gpgpu-sim/remodeling/sm.{h,cc}`, `gpgpu-sim/remodeling/interwarp_coalescing_unit.cc` |
| `writeback_latch_proccess` | `writeback_latch_process` | `Subcore` method | `gpgpu-sim/remodeling/subcore.{h,cc}` |
| `finilized_warps_assignation` | `finalized_warps_assignation` | `Subcore` method | `gpgpu-sim/remodeling/subcore.{h,cc}`, `gpgpu-sim/remodeling/sm.cc` |
| `erase_orifinal_mf` | `erase_original_mf` | local variable | `gpgpu-sim/remodeling/l0_icnt.cc` |
| `reset_cycless_access_history` | `reset_cycles_access_history` | virtual method across the wrapper seam | `gpgpu-sim/{gpu-sim.h,gpu-sim.cc,shader.h,shader_core_wrapper.h}`, `gpgpu-sim/remodeling/sm.{h,cc}` |
| `memory_intermidiate_stages_subcore_unit` | `memory_intermediate_stages_subcore_unit` | `shader_core_config` member | `abstract_hardware_model.cc`, `gpgpu-sim/{shader.h,gpu-sim-config.cc}`, `gpgpu-sim/remodeling/subcore.cc` |
| `dp_shared_intermidiate_stages` | `dp_shared_intermediate_stages` | `shader_core_config` member | `abstract_hardware_model.cc`, `gpgpu-sim/{shader.h,gpu-sim-config.cc}`, `gpgpu-sim/remodeling/subcore.cc` |
| `is_memory_miscelanous` | `is_memory_miscellaneous` | `warp_inst_t` predicate | `abstract_hardware_model.h`, `gpgpu-sim/remodeling/subcore.cc` |
| `memmory_max_concurrent_requests_shmem_per_sm` | `memory_max_concurrent_requests_shmem_per_sm` | `shader_core_config` member | `gpgpu-sim/{shader.h,gpu-sim-config.cc}`, `gpgpu-sim/remodeling/ldst_unit_sm.cc` |
| `memmory_max_concurrent_requests_standard_per_sm` | `memory_max_concurrent_requests_standard_per_sm` | `shader_core_config` member | `gpgpu-sim/{shader.h,gpu-sim-config.cc}`, `gpgpu-sim/remodeling/ldst_unit_sm.cc` |

Paths are relative to `simulator-remodeled/gpu-simulator/gpgpu-sim/src`.
14 files changed, 50 insertions, 50 deletions — a symmetric diff, as a pure
rename must be.

### No-mixed-spelling evidence, immediately after commit 1

```
$ for w in fordward proccess finilized orifinal cycless intermidiate miscelanous memmory; do
    grep -rn "$w" --include="*.cc" --include="*.h" gpgpu-sim/src trace-driven main.cc; done
gpgpu-sim/src/gpgpu-sim/gpu-sim-config.cc:896:  option_parser_register(opp, "-memory_intermidiate_stages_subcore_unit", ...
gpgpu-sim/src/gpgpu-sim/gpu-sim-config.cc:924:  option_parser_register(opp, "-memmory_max_concurrent_requests_shmem_per_sm", ...
gpgpu-sim/src/gpgpu-sim/gpu-sim-config.cc:928:  option_parser_register(opp, "-memmory_max_concurrent_requests_standard_per_sm", ...
gpgpu-sim/src/gpgpu-sim/gpu-sim-config.cc:1013:  option_parser_register(opp, "-dp_shared_intermidiate_stages", ...
```

Exactly the four option **string literals**, nothing else. Six of the eight
families (`fordward`, `proccess`, `finilized`, `orifinal`, `cycless`,
`miscelanous`) are already at zero hits, and the two remaining families survive
only inside those literals — no file carries a mix of old and new spelling for
the same symbol.

## Commit 2 — the four option names, the sweep and the golden re-approval

| Old option | New option |
| --- | --- |
| `-dp_shared_intermidiate_stages` | `-dp_shared_intermediate_stages` |
| `-memory_intermidiate_stages_subcore_unit` | `-memory_intermediate_stages_subcore_unit` |
| `-memmory_max_concurrent_requests_shmem_per_sm` | `-memory_max_concurrent_requests_shmem_per_sm` |
| `-memmory_max_concurrent_requests_standard_per_sm` | `-memory_max_concurrent_requests_standard_per_sm` |

No compatibility alias: an old name is simply an unknown option.

### Sweep coverage

Searched by repo-relative literal, by bare option name and by the misspelled
word alone (`intermidiate`, `memmory`) across the whole tracked tree.

| Area | Carries the old names? | Action |
| --- | --- | --- |
| `gpgpu-sim/configs/tested-cfgs/*/gpgpusim.config` | yes — 31 files × 4 lines = 124 lines | all renamed |
| `gpgpu-sim/src/gpgpu-sim/gpu-sim-config.cc` | yes — the 4 `option_parser_register` literals | renamed |
| `gpu-simulator/configs/tested-cfgs/*/trace.config` | no — trace-side configs carry no `-memory_*`/`-dp_*` option | none |
| `util/tuner/**` (`config_template`, `NVIDIA_GeForce_RTX_4090`, `GPU_Microbenchmark` emitters) | no | none |
| `util/job_launching/**` yml `extra_params` emitters | no | none |
| any tracked `*.yml/*.yaml/*.py/*.sh/*.cu/Makefile*` | no | none |

33 files changed in commit 2 (31 configs + `gpu-sim-config.cc` + `goldens.json`).

### Retired-name loud-failure demo

Scratch configs outside the repo tree (`/tmp/s7_scratch/bad.config`,
`bad2.config` — copies of `SM89_RTX4090/gpgpusim.config` with one old option
line appended), run against the checked-in HALF fixture trace
(`tests/remodeled_trace/fixtures/half_pipeline_sm89.tar.gz`) and the SM89
`trace.config`:

```
$ accel-sim.out -trace /tmp/s7_scratch/half_pipeline_sm89/traces/dynamic_trace.pb \
    -config /tmp/s7_scratch/bad.config \
    -config .../configs/tested-cfgs/SM89_RTX4090/trace.config
```

| Appended old option | exit | message |
| --- | ---: | --- |
| `-dp_shared_intermidiate_stages 1` | 1 | `GPGPU-Sim ** ERROR: Unknown Option: '-dp_shared_intermidiate_stages'` |
| `-memory_intermidiate_stages_subcore_unit 3` | 1 | `GPGPU-Sim ** ERROR: Unknown Option: '-memory_intermidiate_stages_subcore_unit'` |
| `-memmory_max_concurrent_requests_shmem_per_sm 4` | 1 | `GPGPU-Sim ** ERROR: Unknown Option: '-memmory_max_concurrent_requests_shmem_per_sm'` |
| `-memmory_max_concurrent_requests_standard_per_sm 8` | 1 | `GPGPU-Sim ** ERROR: Unknown Option: '-memmory_max_concurrent_requests_standard_per_sm'` |

All four fail at option parsing, before any simulation, with a non-zero exit.

### Golden re-approval

`observe` on the swept tree, then a script assertion that every case's stats are
byte-identical to the approved goldens (exact `repr` comparison per stat, no
tolerance, plus identical key sets) and that the flattened comparison contract
differs only in config `sha256` fields:

```
sm89_shared_lat_rtx4090_observation: stats identical: True (20 stats)  contract diffs: ['configs.gpgpusim.sha256']
sm89_half_pipeline_rtx4090_semantic: stats identical: True (20 stats)  contract diffs: ['configs.gpgpusim.sha256']
sm89_fp64_dispatch_rtx4090_semantic: stats identical: True (20 stats)  contract diffs: ['configs.gpgpusim.sha256']
ampere_pathfinder_sm86_observation:  stats identical: True (20 stats)  contract diffs: ['configs.gpgpusim.sha256']

RESULT: ALL BYTE-IDENTICAL, CONTRACT DIFFERS ONLY IN CONFIG sha256
```

Exit 0. Byte-identical stats mean the rename changed no simulated behavior; only
config-file provenance moved, which is exactly what the hash lock exists to
surface for deliberate approval.

New gpgpusim config hashes:

- `SM89_RTX4090/gpgpusim.config`: `cfcdee5a…` → `3d3f6a43c3909cc29f7df67da6ea807e3766c1f8826d52832008cc46f901ef1d`
- `SM86_RTX3080/gpgpusim.config`: `e82ab102…` → `3245e2c0c51a5f8932d4ee9913f6a9519e5e06cf86e062bd3dc03336655bff71`

Minimal `goldens.json` diff — 5 lines: 4 case hashes (SM89 ×3, SM86 ×1) plus
`source_commit`, set to commit 1 `0d357ce691fcf032dadeec6de5f531da3ba2fce6`.
Following the convention established for the earlier latency-convergence
re-approval, the combined sweep+re-approval commit cannot reference its own
hash, so provenance points at the code commit whose behavior the byte-identical
stats reflect. Commit 3 is behavior-neutral against the same goldens (4/4,
observed_not_golden 0), so the field is left pointing at `0d357ce`.

`check` against the re-approved goldens: exit 0, 4/4 passed.

## Commit 3 — comments and magic numbers

### Comment translations

Seven Spanish sites matched the surveyed patterns; an eighth
(`ldst_unit_sm.cc`, `VER QUE HACER`) was found by a wider Spanish sweep and is
handled with them. Six carry real information and were translated into English
statements of the limitation; two were removed as scratch.

| Site | Original | Disposition |
| --- | --- | --- |
| `remodeling/new_stats.h` — `Single_stat_abstract::get_value` | `// VER COMO HACER` | Translated. The interface exposes only an integer accessor, so `Single_stat_double` truncates its `double` on read; a lossless accessor does not exist. |
| `remodeling/new_stats.h` — next line | `// virtual double get_value() const = 0;   // VER COMO HACER` | **Deleted.** A commented-out alternative declaration; its only information (the missing double-typed accessor) is now stated in the translated comment above it, so keeping the dead declaration would duplicate it. |
| `remodeling/interwarp_coalescing_unit.cc` — `insert_access` merge branch | `// QUE PASA SI VIENE L1 BYPASS y HAY L1D ya ahi o viceversa? De momento que haga lo que decida el primer acceso.` | Translated. Merging ignores a cache-path mismatch (L1-bypass access meeting an L1D entry or the reverse); the path chosen by the access that created the entry wins. |
| `remodeling/interwarp_coalescing_unit.cc` — `access_is_candidate_to_be_inserted` | `}// Falta el de STRONG y el de Atomics` | Translated and moved onto its own line. States that accesses with STRONG memory ordering and atomics are not rejected here and that filter is still missing. |
| `remodeling/interwarp_coalescing_unit.cc` — `pop_policy_dep_counters` assert | `assert(m_num_tables == 1);// DE MOMENTO` | Translated: the policy only supports a single table so far. |
| `remodeling/interwarp_coalescing_unit.cc` — same function, `m_table_idx` | `res.m_table_idx = 0; // DE MOMENTO` | Translated: the single supported table. |
| `remodeling/subcore.cc` — `get_fu`, `SP_OP` case | `/// INCLUIR AQUI IMAD` | Translated and moved above the `if`: IMAD is deliberately kept out of the INT-pipeline reroute; routing it there is not supported yet. |
| `remodeling/ldst_unit_sm.cc` — `dispatch_to_memory_access_queue_l1Dcache` | `// unsigned int max_num_accesses_per_cycle = …;// VER QUE HACER` | **Deleted.** A fully commented-out local computation with no live reader and no statement about the design; pure scratch. |

The `abort()`-stubbed multi-table path in `insert_access` keeps its existing
English message and is untouched.

### Magic numbers

All new constants are `constexpr` inside `namespace remodel`, in the header that
already owns the code reading them. No new config option was introduced.

| Value | Named constant | Defined in | Used in |
| ---: | --- | --- | --- |
| 16 | `SASS_INSTRUCTION_SIZE_IN_BYTES` | `remodeling/ibuffer_remodeled.h` | `ibuffer_remodeled.cc` `get_next_pc_to_fetch_request` (2 sites) |
| 255 | `RESERVED_REG_NUMBER` | `remodeling/sm.h` | `sm.cc` `check_is_reserved_regs_remodeling` |
| 63 | `RESERVED_UREG_NUMBER` | `remodeling/sm.h` | `sm.cc` `check_is_reserved_regs_remodeling` |
| 7 | `RESERVED_PRED_NUMBER` | `remodeling/sm.h` | `sm.cc` `check_is_reserved_regs_remodeling` |
| 7 | `RESERVED_UPRED_NUMBER` | `remodeling/sm.h` | `sm.cc` `check_is_reserved_regs_remodeling` |
| 256 | `GLOBAL_ID_BASE_UREG` | `remodeling/sm.h` | `sm.cc` `translate_reg_to_global_id` |
| 512 | `GLOBAL_ID_BASE_PRED` | `remodeling/sm.h` | `sm.cc` `translate_reg_to_global_id` |
| 520 | `GLOBAL_ID_BASE_UPRED` | `remodeling/sm.h` | `sm.cc` `translate_reg_to_global_id` |
| 6 | `NUM_WAIT_BARRIER_MASK_BITS` | `remodeling/subcore.h` | `subcore.cc` `wait_barriers_to_check_generic` |
| 63 | `MAX_WAIT_BARRIER_COUNTER_VALUE` | `remodeling/warp_dependency_state.h` | `warp_dependency_state.cc` `Wait_Barrier::increase_counter` |
| 4 | `ADDR_SIGNATURE_SPACE_BITS` | `remodeling/interwarp_coalescing_unit.h` | `interwarp_coalescing_unit.cc` `get_addr_signature` |
| 5 | `NUM_WRITEBACK_CLIENTS` | `remodeling/ldst_unit_sm.h` | `ldst_unit_sm.cc` `init` |

The `RESERVED_*` numbers are the trace encodings of the architectural discard
registers RZ / URZ / PT / UPT; the `GLOBAL_ID_BASE_*` values lay the four
register files out in one flat id space (regular `[0,256)`, uniform from 256,
predicates from 512, uniform predicates from 520).

**Wait-barrier bitset width.** The width cannot come from
`shader_core_config::num_wait_barriers_per_warp`: that field is a runtime
`unsigned int` parsed from the config, and `std::bitset` needs a compile-time
width. The mask itself is a fixed-width field of the trace control bits, so the
constant is the correct model. The config value is now asserted against it
(`assert(m_config->num_wait_barriers_per_warp <= NUM_WAIT_BARRIER_MASK_BITS)`)
immediately before the loop that indexes the bitset, closing the out-of-range
indexing hole. No shipped config sets `-num_wait_barriers_per_warp` at all, so
every config uses the registered default 6 and the assert cannot fire on any of
them; asserts are live in the release build and the gate passes.

### `override` specifiers

`remodeling/ldst_unit_sm.h` had one method that overrides a base virtual without
saying so: `void print(FILE *fout) const` against
`functional_unit::print(FILE*) const`. `override` added; the compiler verifies
the claim. Verified with `g++ -Wsuggest-override -fsyntax-only` on
`ldst_unit_sm.cc` — before: one warning, on `ldst_unit_sm::print`; after: zero
warnings from that header.

The neighbouring `clock_multiplier`, `active_lanes_in_pipeline` and `stallable`
are **not** overrides — the base either does not declare them or declares them
non-virtual — so `override` would not compile there and they are left alone.

Deliberately not fixed: `functional_unit.h:231`,
`'virtual void remodel::functional_unit_shared_sm_part::issue(register_set_uniptr&)' was hidden [-Woverloaded-virtual]`.
Changing overload visibility can change overload resolution at call sites, which
is not a structure-preserving edit; left as-is by design.

## Dependency-direction ledger

Reverse includes of `remodeling/` from outside it, from
`simulator-remodeled/gpu-simulator/gpgpu-sim/src`, unchanged at **2**:

```
$ grep -rln '#include.*remodeling/' --include="*.cc" --include="*.h" . \
    | grep -v '^\./gpgpu-sim/remodeling/' | sort
gpgpu-sim/gpu-sim.h
gpgpu-sim/shader_core_wrapper.h
```

S7 added and removed no `#include ".../remodeling/..."` line, so the include
topology is untouched. Both remaining edges are L3→L2 downward includes, which
§2 of the design classifies as allowed.

## Out-of-scope references (reported, not changed)

- **Documentation prose.** The old spellings survive in `docs/plans/*` and
  `validation/refactoring/*` (historical plan and record text, which prior-stage
  policy says not to rewrite) and in `docs/detailed-design/{00,03B,03C,06,07,08,09}`
  (living design documents). Following the precedent set by the latency
  convergence record, a design-document refresh is a separate task, not part of a
  code/config sweep. The affected design docs are the ones naming
  `proccess`, `intermidiate`, `memmory` and the two renamed option names.
- **Vendored AccelWattch XML.** `proccess` appears in all 31
  `configs/tested-cfgs/*/accelwattch_sass_sim.xml` files, but as `proccessors`
  inside an upstream prose comment ("complexity effective proccessors paper").
  It is unrelated to any symbol here and the files are vendored power-model data.
- **`-Woverloaded-virtual` in `functional_unit.h`** — see above.
- **`-Wsuggest-override` elsewhere in `remodeling/`** — `new_stats.h` (2 sites,
  `Single_stat_*::print`) and `sm.h` (6 sites on `SM` methods) also miss
  `override`. Outside the stated scope of this step (`ldst_unit_sm.h`); each is a
  one-word, compiler-verified change if the coordinator wants it folded in later.
- **`.gitignore`** — a pre-existing uncommitted edit belonging to another change;
  left unstaged.

## Independent rerun

Independent read-only rerun of the gate is the coordinator's responsibility per
the roadmap execution model and is not part of this record. All commands above
are reproducible from a clean checkout of HEAD `2f89f2c`.
