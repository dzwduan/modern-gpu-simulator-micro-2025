# Stage 4 S3 — latency-config convergence to the trace namespace

Branch: `dev_dzw`. Governing design:
`docs/plans/2026-07-22-stage4-reorganization-design.md` §4 S3. Goal: make the
trace namespace the single authority for every latency by retiring the
vestigial PTX opcode-latency option namespace. Behavior-neutral; goldens
re-approved for config-hash changes only.

## Environment

| Fact | Value |
| --- | --- |
| Start HEAD | `9968a4c` (branch `dev_dzw`) |
| Build | `cd simulator-remodeled && source ./gpu-simulator/setup_environment_no_git.sh && make -j$(nproc) -C ./gpu-simulator/` |
| Gate | `python3 -m unittest discover -s tests` and `OMP_NUM_THREADS=1 python3 tests/remodeled_trace/run_regression.py check` (from repo root) |
| Binary | `simulator-remodeled/gpu-simulator/bin/release/accel-sim.out` |

## Commits

| # | Hash | Subject |
| --- | --- | --- |
| 1 | `65f6062` | refactor: size fixed-latency pipelines from trace latencies |
| 2 | `2aa561a` | refactor: retire the PTX opcode latency option surface |
| 3 | (this record) | docs: record stage four latency convergence verification |

Ordering follows the design's required split: commit 1 is the depth
re-derivation alone (the PTX options stay registered but unused) so the risky
part is gated against **unchanged** goldens; commit 2 deletes the option
surface, sweeps the configs, and re-approves the goldens for the config-hash
delta.

## Gate table

| State | build | unittest (count) | check (passed/total) | goldens |
| --- | ---: | --- | --- | --- |
| baseline `9968a4c` | 0 | 0 (21) | 0 (4/4) | unchanged |
| commit 1 `65f6062` | 0 | 0 (21) | 0 (4/4), observed_not_golden 0 | UNCHANGED (isolation proof) |
| commit 2 `2aa561a` | 0 | 0 (21) | 0 (4/4) | re-approved (config sha256 only) |

## What converged

Instruction latencies were already 100% trace-namespace
(`trace_config::set_latency` maps every op class from
`-trace_opcode_latency_initiation_*`). The only remaining PTX-namespace output
was the fixed-latency pipeline **depth** sizing via
`shader_core_config::max_{sp,int,dp}_latency` (plus the never-read
`max_{sfu,tensor_core}_latency`), computed in `set_pipeline_latency()` from the
`-ptx_opcode_latency_*` strings. Those depths are now derived from the trace
latency routed to each pipeline; the `max_*_latency` members and
`set_pipeline_latency()` are deleted; the PTX option surface is removed.

Scope note — `max_sfu_latency` (a fifth member, not in the four the audit
named) was removed together with the four: it has zero readers repo-wide
(grep over `src/**/*.{cc,h}` finds only its declaration and the assignment in
the deleted `set_pipeline_latency()`), so deleting the derivation function
orphaned it. It is PTX-derived dead weight of the same block.

## Pre-flight depth-equivalence audit (before editing)

Latency inputs (gpgpusim.config PTX + `-predicate_latency`/`-tensor_latency`;
trace.config `-trace_opcode_latency_initiation_*`, defaults where the file
omits a class):

| | SM89_RTX4090 | SM86_RTX3080 |
| --- | --- | --- |
| PTX fp[1] / int[1],int[5] / dp[1] | 4 / 4,0 / 64 | 4 / 4,0 / 64 |
| config predicate_latency | 13 | 13 |
| config tensor_latency | 32 | 32 |
| trace fp / half / int | 4 / 6 (default) / 4 | 2 / 3 / 2 |
| trace dp / sfu / tensor | 54 / 23 / 32 | 24 / 8 / 8 |
| trace predicate | 2 (default) | 13 |
| is_fp32_and_int_unified_pipeline | 0 (default) | 0 (default) |
| is_dp_pipeline_shared_for_subcores | 1 | 1 |

Old PTX-derived depths: `max_sp = fp[1]`, `max_int = max(int[1],int[5],
config_predicate)`, `max_dp = dp[1]`. Effective depth per unit, old vs new,
with the maximum fixed-latency placement (`latency-1`) each unit must cover:

**SM89_RTX4090**

| Unit | old depth | new depth | max routed latency | new placement < new depth |
| --- | ---: | ---: | ---: | --- |
| SP | max(4, 4, 6) = 6 | max(4, 6) = 6 | half 6 | 5 < 6 |
| INT | max(4,0,13) = 13 | max(4, 13) = 13 | int 4 / pred 2 → 4 | 3 < 13 |
| DP (shared) | max(64, 54) = 64 | 54 | dp 54 | 53 < 54 |
| SFU | max(cfg, 23) | max(cfg, 23) (unchanged) | sfu 23 | 22 < depth |
| TENSOR | 32 | 32 (unchanged) | tensor 32 | 31 < 32 |

**SM86_RTX3080**

| Unit | old depth | new depth | max routed latency | new placement < new depth |
| --- | ---: | ---: | ---: | --- |
| SP | max(4, 2, 3) = 4 | max(2, 3) = 3 | half 3 | 2 < 3 |
| INT | max(4,0,13) = 13 | max(2, 13) = 13 | int 2 / pred 13 → 13 | 12 < 13 |
| DP (shared) | max(64, 24) = 64 | 24 | dp 24 (no DP ops in trace) | 23 < 24 |
| SFU | max(cfg, 8) | max(cfg, 8) (unchanged) | sfu 8 | 7 < depth |
| TENSOR | 32 | 32 (unchanged) | tensor 8 | 7 < 32 |

The only depth changes are shrinks: SM89 shared-DP 64→54, SM86 SP 4→3, SM86
shared-DP 64→24. Fixed-latency placement is `stage = latency-1` and an
instruction traverses in `latency` cycles regardless of total depth, so stages
beyond `latency-1` are never occupied. Each new depth stays `>=` the maximum
routed latency, so every shrink removes only dead stages. INT depth is
identical old vs new (13) for both configs because the retained
`predicate_latency` term (13) dominates.

The non-shared DP else-branch (`max_dp_latency` → trace dp) is re-derived for
correctness but neither gate config exercises it (both set
`-is_dp_pipeline_shared_for_subcores 1`).

## Commit 1 — unchanged-golden isolation proof

Depth re-derivation (`remodeling/subcore.cc`, `remodeling/sm.cc`), deletion of
the five `max_*_latency` members and `set_pipeline_latency()`
(`shader.h`/`shader.cc`), and the narrowed startup validator
(`gpu-sim-config.cc`, `gpu-sim.h`, `main.cc`). The PTX options remain
registered and the configs still carry their PTX lines, so the goldens are
untouched. `check` against the **unchanged** goldens:

```
summary {'failed': 0, 'observed_not_golden': 0, 'passed': 4, 'total': 4}
  sm89_shared_lat_rtx4090_observation passed  mismatches=0
  sm89_half_pipeline_rtx4090_semantic passed  mismatches=0
  sm89_fp64_dispatch_rtx4090_semantic passed  mismatches=0
  ampere_pathfinder_sm86_observation  passed  mismatches=0
```

Exit 0. Zero stat mismatches means the depth shrinks are behavior-neutral in
isolation.

## Simplified startup validation — rationale

`validate_supported_trace_contract` mode checks (trace mode, sub-core, remodeled
IBuffer, power off) are unchanged. The depth checks are narrowed by the rule
**keep a check only when the runtime depth still carries a configured knob that
a user could set below the routed latency**:

| Unit | new runtime depth | check | reason |
| --- | --- | --- | --- |
| SP | max(trace fp, half[, int]) | dropped | depth == the routed latency; tautological |
| DP (shared / non-shared) | trace dp | dropped | depth == the routed latency; tautological |
| SFU | max(sfu_latency, trace sfu) | kept | retains the `sfu_latency` floor knob |
| TENSOR | tensor_latency (used directly) | kept | can be set below trace tensor |
| INT | max(trace int, predicate_latency) | kept, now vs **trace predicate** | INT also holds predicate ops; `predicate_latency` is not floored by the trace predicate latency |

The validator signature is slimmed to the latencies it now reads
(`trace_int`, `trace_sfu`, `trace_tensor`, `trace_predicate`); `main.cc` passes
`m_config->get_predicate_latency()`. The new INT check
(`max(trace_int, sc.predicate_latency) >= trace_predicate_latency`) was verified
against every tested config: all 31 `tested-cfgs` have
`-predicate_latency 13 >= trace predicate` (SM89 trace predicate 2 by default,
all others 13), so none newly fatals, and TENSOR (`tensor_latency 32 >= trace
tensor`) is unchanged and already passing.

## Commit 2 — option retirement, sweep, and golden re-approval

- `cuda_sim::ptx_opcocde_latency_options` loses the ten
  `-ptx_opcode_latency_*` / `-ptx_opcode_initiation_*` registrations and keeps
  `-cdp_latency`. The ten `opcode_{latency,initiation}_*` `char*` members are
  seeded in the `cuda_sim` constructor with their former option defaults so the
  fatal-stubbed PTX path (`ptx_instruction::set_opcode_and_latency`) keeps its
  compile-time defaults; the static PTX latency tables are untouched (PTX
  functional retirement is out of scope).
- Sweep: the option lines were removed from all 31
  `gpgpu-sim/configs/tested-cfgs/*/gpgpusim.config` plus
  `util/tuner/config_template/gpgpusim.config` and
  `util/tuner/NVIDIA_GeForce_RTX_4090/gpgpusim.config` (33 files, each carried
  the same 3-line "Instruction latencies" header + 10 option lines, all
  removed). The five `util/tuner/GPU_Microbenchmark/ubench/core/config_*.cu`
  emitters stop printing the retired options; each already prints the
  `-trace_opcode_latency_initiation_*` equivalent, so the measured
  `lat`/`init` remain consumed and the removal is clean (out-of-build CUDA; no
  `nvcc` in this environment, verified by inspection). No `trace.config` and no
  `util/job_launching` yml carried these options.

Re-approval evidence — `observe` on the swept tree, then an assertion that each
case's stats are byte-identical to the approved goldens and the comparison
contract differs only in the gpgpusim config sha256:

```
[each of 4 cases]  stats identical: True (20 stats)  contract diffs: ['configs.gpgpusim.sha256']
RESULT: ALL BYTE-IDENTICAL, CONTRACT DIFFERS ONLY IN gpgpusim sha256
```

New gpgpusim config hashes:
- `SM89_RTX4090/gpgpusim.config`: `a222bd3a…` → `cfcdee5a4b6b60506e443aafb51127d7d915b7d72e2ed038a3099502fee9193d`
- `SM86_RTX3080/gpgpusim.config`: `0ab42905…` → `e82ab1023abdca701241260d39689c2b404c4bf3567c23d5ffbcfccd3ae56d97`

Minimal goldens diff — 5 lines (4 case hashes: SM89 ×3, SM86 ×1) plus
`source_commit`, set to commit 1 `65f60622b98aab834b394ae6d3afe75fc1ce8d00`
(commit 2's parent and the behavior-defining commit; the combined
sweep+re-approval commit cannot reference its own hash, so provenance points to
the code commit whose behavior the byte-identical stats reflect). `check`
against the re-approved goldens: exit 0, 4/4 passed.

## Deleted-option loud-failure demo

Scratch config outside the repo tree (`/tmp/s3_scratch/bad.config`, a copy of
`SM89_RTX4090/gpgpusim.config` with `-ptx_opcode_latency_fp 4,4,4,4,39`
appended) run against the checked-in HALF fixture trace and the SM89
trace.config:

- Exit code: `1`
- Error text: `GPGPU-Sim ** ERROR: Unknown Option: '-ptx_opcode_latency_fp'`

## Dependency-direction ledger

Reverse includes of `remodeling/` from outside it (from
`simulator-remodeled/gpu-simulator/gpgpu-sim/src`), unchanged at **4**:

```
abstract_hardware_model.cc
gpgpu-sim/gpu-sim.h
gpgpu-sim/shader.cc
gpgpu-sim/shader_core_wrapper.h
```

S3 added or removed no `#include ".../remodeling/..."` line, so the include
topology is unchanged; `grep -c functional_unit abstract_hardware_model.h` = 4
(unchanged).

## Out-of-sweep-scope references (reported, not swept)

`ptx_opcode` still appears in documentation and generated artifacts, all outside
the config/emitter sweep scope:

- `docs/plans/2026-07-17-structure-refactor-roadmap.md`,
  `docs/plans/2026-07-22-stage4-reorganization-design.md`,
  `validation/refactoring/2026-07-17-safety-net.md` — historical plan/record
  prose (do not edit prior-stage records).
- `docs/detailed-design/08-配置矩阵.md` — design-matrix doc; a follow-up doc
  refresh, out of the code/config sweep.
- `util/tuner/GPU_Microbenchmark/{output.file,stats.txt}` — stale generated
  tuner outputs with no consumers found (`grep` over `*.py/*.sh/*.yml/Makefile`);
  regenerated on the next tuner run, which now emits trace-only.

## Independent rerun

Independent read-only rerun of the gate is the coordinator's responsibility.
Commands are reproducible from a clean checkout of HEAD `2aa561a`.
