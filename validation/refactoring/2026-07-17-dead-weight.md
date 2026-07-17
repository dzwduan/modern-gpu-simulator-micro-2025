# Dead-Weight Removal Verification Record

## Status

Stage 阶段二（死重清除与默认值修复）of
`docs/plans/2026-07-17-structure-refactor-roadmap.md` is complete. Every scope
item is a self-contained commit that rebuilt clean and passed the full gate
(`python3 -m unittest discover -s tests` exit 0, 21 tests;
`python3 tests/remodeled_trace/run_regression.py check` exit 0, 4/4 passed,
byte-identical to the approved goldens). Two scope items were narrowed to keep
the gate green and are reported below rather than forced:

- Item 3 keeps the `-network_mode`/`-inter_config_file` options and the `.icnt`
  files as vestigial config, because the goldens hash-lock the tested-config
  files and removing `-network_mode` from them fails the golden config
  provenance contract, which is off-limits.
- Item 2 leaves two always-empty `POWER_FLAGS` blocks in
  `remodeling/{Makefile,fusedMemory/Makefile}` untouched to respect the hard
  remodeling/ boundary; they have zero build effect.

## Environment

| Fact | Value |
| --- | --- |
| Start HEAD | `e4fe5c8` (branch `dev_dzw`) |
| Build | `cd simulator-remodeled && source ./gpu-simulator/setup_environment_no_git.sh && make -j$(nproc) -C ./gpu-simulator/` |
| Gate | `python3 -m unittest discover -s tests` and `OMP_NUM_THREADS=1 python3 tests/remodeled_trace/run_regression.py check` (from repo root) |
| Binary | `simulator-remodeled/gpu-simulator/bin/release/accel-sim.out` |
| Baseline binary SHA-256 | `f0cee49a6c3c2fe263833cbdef9dbbfd3d18d3c9e16b46960de3bd0db3199125` |

## Commits

| Item | Hash | Subject | Binary SHA-256 |
| --- | --- | --- | --- |
| 1 | `80c2419` | refactor: remove unused gmmu, page table walker, and unreachable tlb branches | `e641ae09…` |
| 2 | `7bcfcea` | refactor: remove accelwattch power simulation subsystem | `9e89d00e…` |
| 3 | `6c2e731` | refactor: retire the intersim2 interconnect implementation | `0b8fcf0a…` |
| 4 | `75be86a` | refactor: remove unused opencl, cuobjdump-to-ptxplus, and debug tooling | `2bb7c633…` |
| 5 | `6fff2c1` | build: drop dead archive rules and stop rebuilding every gpgpu-sim object | `7c04b85d…` |
| 6 | `ae9e35d` | feat: validate the supported remodeled trace configuration at startup | `a4c982d1…` |
| 7 | `35e6385` | refactor: remove dead debug code from the remodeling sm and subcore | `3d7c51ae…` |
| 8 | (this record) | docs: record dead-weight removal stage completion | — |

## Gate table

Every code/Makefile commit rebuilt from a clean object tree (item 5 ran a
`make clean` first; the others were verified after removing orphaned objects
left by their own source deletions). Exit codes:

| Commit | build | unittest (count) | check (passed/total) |
| --- | ---: | --- | --- |
| baseline `e4fe5c8` | 0 | 0 (21) | 0 (4/4) |
| item 1 `80c2419` | 0 | 0 (21) | 0 (4/4) |
| item 2 `7bcfcea` | 0 | 0 (21) | 0 (4/4) |
| item 3 `6c2e731` | 0 | 0 (21) | 0 (4/4) |
| item 4 `75be86a` | 0 | 0 (21) | 0 (4/4) |
| item 5 `6fff2c1` | 0 | 0 (21) | 0 (4/4) |
| item 6 `ae9e35d` | 0 | 0 (21) | 0 (4/4) |
| item 7 `35e6385` | 0 | 0 (21) | 0 (4/4) |

## Per-item evidence

Reference-grep discipline: for each deletion, grep repo-wide for the
repo-relative literal, relative-path forms (`../<dir>/`), and bare filenames,
excluding this stage's own roadmap/plan documents.

### Item 1 — gmmu, page table walker, dead TLB branches (`80c2419`)

- `GMMU` is never constructed; the only non-self references to `gmmu.h` were
  `gpu-sim.cc` (include, removed) and the timing comment at
  `local_interconnect.cc:318` (`// This extra node is for the LL-TLB and GMMU`);
  that `n_mem = m_n_mem + 1` line is a timing-affecting latch and is retained
  with its comment. `PageTableWalker` was referenced only by `gmmu.h`.
- Deleted `remodeling/{gmmu.cc,gmmu.h,page_table_walker.cc,page_table_walker.h}`;
  post-delete grep for `gmmu`/`page_table_walker`/`GMMU`/`PageTableWalker` over
  `src` returns nothing but the retained timing comment.
- TLB placeholder: `tlb_acc` is hardcoded to `HIT` in `l0_icnt.cc` and
  `ldst_unit_sm.cc`; the `MISS`/`MSHR_HIT`/`RESERVATION_FAIL` branches are
  unreachable and removed. `TLB_MISS_ACC_DATA` is never assigned to any
  `mem_fetch` (only enum-definition and switch-case reads remain), so the
  `ldst_unit_sm.cc` pop-and-drop branch is dead and removed. Timing latches
  (`if(tlb_acc == HIT)` push-if-space bodies) are retained. The byte-identical
  goldens confirm no branch was live.

### Item 2 — AccelWattch power subsystem (`7bcfcea`)

- The build compiled with `-DGPGPUSIM_POWER_MODEL` (set by
  `gpgpu-sim/setup_environment` when `src/accelwattch/` exists), so the power
  model was active, not dead. Removal deletes `src/accelwattch/` (107 files),
  `power_interface.{cc,h}`, `power_stat.{cc,h}`, `util/accelwattch/` (25 files,
  incl. the 44 MB `validation.tgz`), and `util/hw_stats/` (8 files). The
  `util/hw_stats/` referrer flagged in the repo-hygiene stage lived in
  `util/accelwattch/accelwattch_hw_profiler/profile_validation_perf.sh`, deleted
  in the same commit, so the blocker is resolved.
- All power statistics collection was either guarded by the runtime
  `g_power_simulation_enabled` flag (off in every tested config — confirmed no
  `-power_simulation_enabled` in `configs/tested-cfgs/`) or only maintained
  power accumulators, so removal is behavior-preserving; the goldens are
  byte-identical.
- `gpgpu_sim::get_scaling_coeffs()` now returns `NULL`. `remodeling/sm.cc`
  consults it and `m_scaling_coeffs` only under `if (g_power_simulation_enabled)`
  (off), so this avoids touching remodeling/ and never dereferences null on the
  supported path.
- Retained as vestigial (not power_stat-typed, non-power referrers or the
  remodeling/ boundary): the `power_config` option group and per-config
  `accelwattch_sass_sim.xml` (so configs still parse), `set_dram_power_stats`
  (dram/l2cache), `get_icnt_power_stats` (shader/sm), and the two always-empty
  `POWER_FLAGS` blocks in `remodeling/{Makefile,fusedMemory/Makefile}`.
- `README.md` AccelWattch feature line removed; `format-code.sh` accelwattch
  clang-format lines removed. `gpgpu-sim/{CHANGES,COPYRIGHT}` and the license
  headers keep their AccelWattch/Booksim attribution (third-party copyright is
  out of deletion scope).
- Build note: removing the power objects from `libcudart.so` exposed a latent
  link-order fragility — the trace driver link places `-lcudart` before its
  objects and the default is `--as-needed`, so the shared library was dropped
  and every gpgpu-sim symbol went undefined. Fixed by force-linking libcudart
  (`-Wl,--no-as-needed -lcudart -Wl,--as-needed`) in `gpu-simulator/Makefile`;
  this is a build-correctness change with no runtime effect.

### Item 3 — intersim2 interconnect (`6c2e731`)

- `git rm -r src/intersim2/` (128 files). The `INTERSIM` network mode aborted at
  startup, so it was dead; `icnt_wrapper_init` now initializes the local
  crossbar unconditionally (a default configuration no longer reaches the abort
  path). Removed the `INTERSIM`/`LOCAL_XBAR` enum and the `$(INTERSIM)` build
  recursion and link inputs from `gpgpu-sim/Makefile`.
- intersim2 also provided the `Stats`/`Module` statistics classes (declared in
  `src/gpgpu-sim/intersim2_stats.h`) that `dram.cc`, `gpu-sim.cc`,
  `cuda-sim.cc`, and `statwrapper.cc` use for general statistics (e.g. the DRAM
  request-queue histogram). Those implementations were relocated from
  `intersim2/{stats.cpp,module.cpp}` into
  `src/gpgpu-sim/intersim2_stats.cc` (behavior-identical) so intersim2/ could
  still be fully deleted.
- **Config sweep narrowed and reported.** `goldens.json` records the SHA-256 of
  each tested `gpgpusim.config` as a provenance ("configs") contract. Removing
  `-network_mode 2` from `SM89_RTX4090/gpgpusim.config` and
  `SM86_RTX3080/gpgpusim.config` changes those hashes and fails the golden
  contract (observed: `golden_mismatch`, 4/4 failed). Updating goldens is
  outside this stage's boundary. Per the brief's escape hatch, the
  `-network_mode` and `-inter_config_file` options and `g_network_mode` are kept
  (parsed for compatibility, no longer selecting an implementation) and the
  `.icnt` files remain; the config sweep and option deletion are deferred to a
  stage that may re-approve goldens. `g_network_config_filename` is never read,
  so the retained option is inert.

### Item 4 — opencl, cuobjdump-to-ptxplus, debug tooling (`75be86a`)

- `git rm -r gpgpu-sim/{libopencl,cuobjdump_to_ptxplus,debug_tools}` (3 + 15 +
  133 = 151 files); removed the `libOpenCL.so`/`no_opencl_support`/
  `cuobjdump_to_ptxplus` Makefile targets, the OpenCL `NVOPENCL_*` setup in
  `setup_environment`, and their gitignore entries. `libcuda/` and `cuda-sim/`
  stay.
- Retained couplings (both PTX-mode only, outside the remodeled trace contract
  and not exercised by the gate): `cuda-sim/ptx_loader.cc` runtime-invokes the
  cuobjdump-to-PTXplus binary in PTXplus mode; `cuda-sim/cuda-sim.cc` writes to
  `debug_tools/WatchYourStep/data/` under the Watch-Your-Step debug mode
  (`WYS_EXEC_PATH`). Remaining `cuobjdump_to_ptxplus`/`libopencl` hits are
  comments or upstream CHANGES.

### Item 5 — build hygiene (`6fff2c1`)

- The final link (`libcudart.so` in `gpgpu-sim/Makefile`) consumes the loose
  `.o` files, so the `libgpu_uarch_sim.a` and `libgpu_remodeling_uarch_sim.a`
  archive rules were dead; replaced with `all: $(OBJS)` default targets.
- The `%.o: %.cc remodeling fusedMemory` rule listed the phony subdir builds as
  prerequisites of every object, forcing a full recompile every build; they are
  now order-only prerequisites of the overall target. Removed the
  remodeling/Makefile rules for sources absent from that directory
  (`option_parser`, `dram_sched`, and the cuda-sim `ptx.tab.h` recipe).
- Clean rebuild: `make clean` (removes `build/`, `lib/`, `bin/`, and
  traces_enhanced obj) then a full rebuild succeeded and passed the gate (item 5
  row above).
- Incremental no-op: after the full build, a repeated `make` recompiled **zero**
  of the ~24 `src/gpgpu-sim` objects and zero remodeling/fusedMemory objects
  (before this commit, all of them recompiled every build). Two residual
  recompiles remain and are outside this commit's scope — `main.o` (the
  top-level `version` phony prerequisite regenerates the version header) and
  `cuda-sim.o` (the cuda-sim Makefile regenerates `ptx.tab.h`). Both are in the
  top-level and cuda-sim Makefiles, not the two Makefiles this item fixes.

### Item 6 — defaults and startup validation (`ae9e35d`)

- Default alignment: `-gpgpu_sub_core_model` default `0` → `1` (the supported
  contract value). Mismatched-default sweep of the 2026-07-15 contract:
  `is_SM_remodeling_enabled` already defaults to `1`;
  `is_ibuffer_remodeled_enabled` defaults to `0` (mismatched) but the plan
  treats it as a required-capability assertion (P1-05), so it is enforced by the
  startup check rather than changed. Every tested config sets all three
  explicitly, so the default change does not alter regression behavior.
- Startup validation `gpgpu_sim_config::validate_supported_trace_contract`
  (in the config layer, `gpu-sim-config.cc`) runs after both configs are parsed
  (`main.cc`). It fatal-errors when the remodeled core is combined with PTX
  execution (P1-03), sub-core model off (P1-04), the remodeled IBuffer off
  (P1-05), or a fixed-latency pipeline shallower than the trace latency routed
  to it. The depth check mirrors the trace-latency sizing in
  `remodeling/subcore.cc`/`sm.cc` (SP, SFU, and shared DP are sized up from the
  trace latency; INT-when-separate, tensor, and non-shared DP use the configured
  depth), so it never rejects a configuration the simulator would otherwise size
  up — including the SM89 SFU case (configured 21, trace 23, sized to 23). Trace
  latencies are passed as values so the config layer does not depend on the
  trace-driven layer.
- Removed the tracked generated `simulator-remodeled/gpgpu_inst_stats.txt` and
  added it to `simulator-remodeled/.gitignore`.

### Item 7 — remodeling dead code (`35e6385`)

- Deleted the `#if 0` `warp_inst_complete` printf; the
  `check_if_non_released_reduction_barrier` probe (no callers — grep found only
  its declaration and definition) and its `sm.h` declaration; the uncalled
  `Register_file_cache::print` (its loop called `entry.flush()` instead of
  printing — deleted per the deletion principle since it has no callers, which
  removes the state-mutating loop) and its declaration; and the commented-out
  `is_any_waiting` instrumentation in `Subcore::issue` plus the commented
  `std::cout` traces in `SM::instruction_retirement`, `SM::issue_warp`, and
  `Subcore::fetch`. Behavior-preserving; goldens byte-identical.

## Startup-validation machine demo

Scratch config outside the repo tree
(`<scratch>/invalid_ibuffer.config`, a copy of `SM89_RTX4090/gpgpusim.config`
with `-is_ibuffer_remodeled_enabled 0`), run against the checked-in HALF
fixture trace (extracted to a scratch directory):

```
cd simulator-remodeled
source ./gpu-simulator/setup_environment_no_git.sh
LC_ALL=C OMP_DYNAMIC=FALSE OMP_NUM_THREADS=1 \
  gpu-simulator/bin/release/accel-sim.out \
  -trace <scratch>/half_pipeline_sm89/traces/dynamic_trace.pb \
  -config <scratch>/invalid_ibuffer.config \
  -config gpu-simulator/configs/tested-cfgs/SM89_RTX4090/trace.config
```

- Exit code: `1`
- Error text: `GPGPU-Sim config error: unsupported remodeled trace configuration: -is_ibuffer_remodeled_enabled got 0, expected 1.`

## Dependency-direction ledger

Reverse includes of `remodeling/` from outside it (run from
`simulator-remodeled/gpu-simulator/gpgpu-sim/src`):

```
$ grep -rln '#include.*remodeling/' --include="*.cc" --include="*.h" . \
    | grep -v '^\./gpgpu-sim/remodeling/' | sort
abstract_hardware_model.cc
gpgpu-sim/gpu-sim.h
gpgpu-sim/scoreboard.cc
gpgpu-sim/scoreboard_reads.cc
gpgpu-sim/shader.cc
gpgpu-sim/shader_core_wrapper.h
gpgpu-sim/shader.h
```

`grep -c functional_unit abstract_hardware_model.h` = 4 (unchanged).

This is the safety-net baseline (8 files) minus `gpgpu-sim/gpu-sim.cc`, which
dropped off because item 1 removed its only direct remodeling/ include
(`remodeling/gmmu.h`); gpu-sim.cc still reaches remodeling types transitively
through `gpu-sim.h`. The ledger did not grow; no new low-to-high include was
introduced. `intersim2_stats.cc` is a new L1/L3 file that includes only its own
header, so it adds no reverse dependency on remodeling/.

## Tracked-size delta

| Measure | Baseline `e4fe5c8` | HEAD | Delta |
| --- | ---: | ---: | ---: |
| Tracked bytes | 86,550,870 | 31,311,701 | −55,239,169 (−52.7 MB) |
| Tracked files | 1,438 | 1,011 | −427 |

`git diff --stat e4fe5c8 HEAD`: 453 files changed, 249 insertions(+),
187,709 deletions(-). The size reduction is dominated by the 44 MB AccelWattch
`validation.tgz`, the intersim2 tree, and the debug_tools cub/GL headers.

## Deviations and blockers

- **Item 3 config sweep / `-network_mode` option deletion — blocked and
  deferred.** Cause: goldens hash-lock the tested `gpgpusim.config` files;
  goldens are off-limits. The dead intersim2 network implementation is removed
  and LOCAL_XBAR is unconditional, so the acceptance "default startup no longer
  aborts" is met; only the config-file/option cleanup is deferred.
- **Item 2 — two always-empty `POWER_FLAGS` blocks retained** in
  `remodeling/{Makefile,fusedMemory/Makefile}` to respect the hard remodeling/
  boundary; zero build effect. Also retained: `power_config` options + XML data,
  `set_dram_power_stats`, `get_icnt_power_stats` (non-power-typed vestiges).
- **Item 2 — `gpu-simulator/Makefile` link line changed** (`-Wl,--no-as-needed`
  around `-lcudart`) to keep the trace driver linkable after the power objects
  left libcudart.so. Build-correctness only.
- **Item 4 — two PTX-mode couplings retained** (`ptx_loader.cc` cuobjdump
  invocation; `cuda-sim.cc` Watch-Your-Step data path). Out of the remodeled
  trace contract; not exercised by the gate.
- **Item 6 — `is_ibuffer_remodeled_enabled` default left at `0`** (enforced by
  the startup check, not changed), per "change only defaults the plan explicitly
  fixes."
- **Item 5 — two residual incremental recompiles** (`main.o`, `cuda-sim.o`) from
  the top-level and cuda-sim Makefiles, outside this item's scope.

## Independent rerun

Independent read-only rerun of the gate is the coordinator's responsibility per
the roadmap execution model and is not part of this record. The commands above
are reproducible from a clean checkout of HEAD.
