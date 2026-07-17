# Safety-Net Verification Record

## Status

The in-flight remodeled trace semantic repair work is committed in coherent
units, the regression golden set is approved, and the check-mode gate passes
with exit code 0 across all four cases. This record binds the build, unit tests,
regression observations, and golden gate to checked-in commits and a permanent
upstream baseline tag.

Scope decisions bound to this record:

- official execution mode: remodeled trace mode only;
- no simulator logic was changed beyond the already-present working-tree diff;
- goldens lock every deterministically observed statistic (required stats are a
  subset), so later refactoring stages must keep identical regression output.

## Checked-in artifacts

| Artifact | Purpose |
| --- | --- |
| `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc` | Registers and prints the remodeled per-SM dispatch and shared-throttle stats |
| `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc` | Pipeline bounds guards, shared-stage placement, and constructor-supplied throttle source |
| `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc` | HALF fall-through fix, trace-latency SP pipeline sizing, dispatch counters |
| `simulator-remodeled/gpu-simulator/main.cc` | Trace config parse ordering |
| `tests/remodeled_trace/cases.json` | Runnable remodeled trace cases, including the HALF and FP64 semantic cases |
| `tests/remodeled_trace/goldens.json` | Approved golden set for all four cases |
| `tests/remodeled_trace/fixtures/half_pipeline_sm89.tar.gz` (+ `.metadata.json`) | SM89 HALF dispatch trace fixture and provenance |
| `tests/remodeled_trace/fixtures/fp64_dispatch_sm89.tar.gz` (+ `.metadata.json`) | SM89 FP64 shared-DP dispatch trace fixture and provenance |
| `tests/remodeled_trace/workloads/half_pipeline.cu`, `fp64_dispatch.cu` | Deterministic microbenchmark sources |
| `tests/remodeled_trace/generate_semantic_fixtures.py` (+ test) | Deterministic fixture generator |
| `docs/plans/2026-07-15-remodeled-trace-p0-p1-semantic-repair.md` | Semantic repair specification |
| `docs/plans/2026-07-17-structure-refactor-roadmap.md` | Staged refactor roadmap |

## Environment

| Fact | Value |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS |
| Kernel | Linux 5.15.0-173-generic |
| Compiler | g++ (Ubuntu 11.4.0) 11.4.0 |
| CUDA toolchain | 12.6 (build path `cuda-12060`) |
| Python | 3.10.12 |
| Simulator binary | `simulator-remodeled/gpu-simulator/bin/release/accel-sim.out` |
| Binary SHA-256 | `e7a1627f5dde3f9d597a5b96cb27cacc6cda45dca3c7ad0602f47254c9acc17d` |

## Local verification

All commands run from the repository root. The build command additionally sourced
`./gpu-simulator/setup_environment_no_git.sh` from `simulator-remodeled`.

| Command | Exit code | Result |
| --- | ---: | --- |
| `make -j$(nproc) -C ./gpu-simulator/` | 0 | Linked `accel-sim.out` |
| `python3 -m unittest discover -s tests -v` | 0 | 21 tests passed |
| `python3 tests/remodeled_trace/run_regression.py list` | 0 | Four cases listed; manifest validated |
| `python3 tests/remodeled_trace/run_regression.py observe` (run 1) | 0 | 4/4 `observed_not_golden`; no failures or missing stats |
| `python3 tests/remodeled_trace/run_regression.py observe` (run 2) | 0 | Byte-identical stats and comparison contracts to run 1 |
| `python3 tests/remodeled_trace/run_regression.py check` | 0 | 4/4 `passed`; no schema errors, no mismatches |

The two observe runs produced identical parsed statistics and comparison
contracts for every case, establishing determinism before the goldens were
approved. The check-mode run re-executed each case against the approved goldens
and matched exactly.

## Simulator observations

Every case ran with `LC_ALL=C`, `OMP_DYNAMIC=FALSE`, `OMP_NUM_THREADS=1`, found
the completion marker, and exited 0.

Core statistics:

| Field | shared_lat SM89 | half_pipeline SM89 | fp64_dispatch SM89 | pathfinder SM86 |
| --- | ---: | ---: | ---: | ---: |
| `gpu_tot_sim_cycle` | 192,072 | 4,086 | 13,042 | 28,795 |
| `gpu_tot_sim_insn` | 32,881 | 8,448 | 8,544 | 804,960 |
| `gpu_tot_ipc` | 0.1712 | 2.0675 | 0.6551 | 27.9549 |
| `l2_total_cache_accesses` | 99 | 20 | 36 | 5,240 |
| `l2_total_cache_misses` | 99 | 20 | 36 | 3,455 |
| `total dram reads` | 96 | 16 | 24 | 3,035 |
| `total dram writes` | 0 | 0 | 0 | 0 |
| `gpgpu_n_shmem_bkconflict` | 0 | 0 | 0 | 0 |

Remodeled semantic statistics:

| Field | shared_lat SM89 | half_pipeline SM89 | fp64_dispatch SM89 | pathfinder SM86 |
| --- | ---: | ---: | ---: | ---: |
| `remodeled_dispatch_half_to_sp` | 0 | 64 | 0 | 0 |
| `remodeled_dispatch_half_to_int` | 0 | 0 | 0 | 0 |
| `remodeled_dispatch_sp_to_sp` | 5,158 | 3 | 4 | 5,008 |
| `remodeled_dispatch_sp_to_int` | 0 | 0 | 0 | 0 |
| `remodeled_dispatch_int_to_sp` | 0 | 0 | 0 | 0 |
| `remodeled_dispatch_int_to_int` | 17,444 | 130 | 131 | 13,640 |
| `remodeled_dispatch_dp_to_dp` | 0 | 0 | 64 | 0 |
| `remodeled_dispatch_mem_to_mem` | 6,148 | 2 | 3 | 5,848 |
| `remodeled_shared_throttle_dp_events` | 0 | 0 | 64 | 0 |
| `remodeled_shared_throttle_dp_cycles` | 0 | 0 | 64 | 0 |
| `remodeled_shared_throttle_mem_events` | 6,148 | 2 | 3 | 5,848 |
| `remodeled_shared_throttle_mem_cycles` | 12,296 | 4 | 6 | 11,696 |

Semantic evidence for the repaired findings:

- HALF fall-through (P0-02): the HALF fixture dispatches 64 HALF instructions to
  the SP pipeline and zero to INT, and completes without a pipeline bounds
  abort. The SP pipeline is now sized from the parsed HALF/FP32 trace latency.
- DP shared-throttle source (P0-03): the FP64 fixture reports 64 DP dispatches
  with 64 DP throttle events and 64 DP throttle cycles. The SM89 config sets the
  DP shared-pipeline throttle to 1 cycle, so 64 events x 1 cycle = 64 cycles,
  confirming the throttle now uses the unit's own configured value.
- Throttle cross-check: for every case the MEM throttle cycles equal twice the
  MEM throttle events, matching the SM89/SM86 MEM throttle of 2 cycles.
- The `shared_lat` SM89 and `pathfinder` SM86 core statistics match the earlier
  `2026-07-15-remodeled-trace-baseline.md` observations.

## Commits and tag

| Ref | Hash | Subject |
| --- | --- | --- |
| A | `e37cb535d456ce328da4477885bfd7abd97a582b` | fix: correct remodeled trace dispatch, pipeline sizing, and throttle semantics |
| B | `1993175c1aca42eeeb9504688dba29f846a068bf` | test: add HALF and FP64 semantic trace fixtures and dispatch stat coverage |
| C | `c6f53bf3af63f04d695c4413f67477d1a31f358e` | docs: add semantic repair spec and structure refactor roadmap |
| D | `76675ca1b05d6f7822460a3387af5829c52087fd` | test: approve remodeled trace regression goldens |

Commit B carries `cases.json` and every fixture it references in a single
commit. The README testing section and this record are committed together on top
of commit D.

Tag: `upstream-import` annotates commit
`117f9dca3f46b1d85d2a1ec9ddac6b89d49399b3` ("Uploaded") as the permanent upstream
bulk-import baseline. No commit or tag was pushed.

## Open items

- Independent read-only review of this stage is the coordinator's
  responsibility per the roadmap execution model and is not part of this record.
- The RTX 4090 hardware cycle-error threshold remains unapproved; the
  hardware-comparison script is retained but is not a regression oracle.
- `refactoring-audit.html` and `.codegraph/` are intentionally left untracked
  for later stages and are not part of these commits.

## Review response

The independent read-only review of commits `e37cb53..ff33f2a` reported one
finding: the shared DP pipeline is sized only from the ptx-derived
`max_dp_latency` while placement indexes `trace latency - 1`, so the SM75
tested configs (trace dp latency 25 vs depth 19) abort at the placement bounds
check. A config sweep across all tested-cfgs confirmed exactly three
depth/latency violations of this class:

| config | unit | depth | trace latency |
| --- | --- | ---: | ---: |
| SM75_RTX2070_S | DP shared | 19 | 25 |
| SM75_RTX2080_TI | DP shared | 19 | 25 |
| SM89_RTX4090 | SFU | 21 (default) | 23 |

Mechanism reproduction (same architecture, no cross-arch noise): a scratch
copy of the SM89 gpgpusim.config with `-ptx_opcode_latency_dp 64,40,64,64,330`
run against the fp64 fixture trace.

- Before fix: exit 1, `Invalid shared pipeline placement: unit=DP_SM_shared
  op=4 latency=54 depth=40`.
- After fix (commit `f57d4f3`): exit 0, `GPGPU-Sim: *** exit detected ***`,
  no placement message.

Post-fix gate, run against the unchanged approved goldens:

- `python3 -m unittest discover -s tests` — exit 0, 21 tests OK.
- `python3 tests/remodeled_trace/run_regression.py check` — exit 0, 4/4
  passed. Unchanged goldens passing proves the sizing change introduces zero
  behavioral drift on the supported configurations.

Notes for later stages:

- No current fixture contains SFU instructions, so the SFU sizing fix is
  covered by the mechanism reproduction and code symmetry only. The stage-two
  startup validation of config combinations should include the
  depth-vs-trace-latency check for every fixed-latency unit, and an SFU
  fixture is a candidate addition.
- Manual simulator invocations must pin `OMP_NUM_THREADS=1` as the harness
  does; unpinned runs oversubscribe libgomp and crawl (a 3-second case ran
  past 100 seconds), which can masquerade as a hang.

## Dependency-direction baseline

Reverse includes of `remodeling/` from outside it (grep evidence, to shrink
monotonically from stage three onward and reach the target rules of the
roadmap's layering section):

- `src/abstract_hardware_model.cc` (L0 -> L2, must reach zero)
- `src/gpgpu-sim/scoreboard.cc`, `src/gpgpu-sim/scoreboard_reads.cc` (L1 -> L2, must reach zero)
- `src/gpgpu-sim/shader.cc`, `src/gpgpu-sim/shader.h` (dissolved by the legacy retirement stage)
- `src/gpgpu-sim/shader_core_wrapper.h` (becomes the formal L3-L2 contract)
- `src/gpgpu-sim/gpu-sim.cc`, `src/gpgpu-sim/gpu-sim.h` (L3 -> L2, allowed direction)

`abstract_hardware_model.h` additionally references `functional_unit` at 4
sites (L0 -> L2, must reach zero).
