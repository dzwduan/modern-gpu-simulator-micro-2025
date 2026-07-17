# Remodeled SM Architecture — Entry Page

This repository's only supported execution path is the remodeled, trace-driven
SM timing model in
`simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/`. It
replaces the legacy `shader_core_ctx` pipeline with a sub-core-partitioned SM
(`SM` owns an array of `Subcore`, each with a private 8-stage pipeline, L0
instruction cache, and register files) that resolves instruction dependencies
from compiler-embedded SASS control bits (stall counter, yield, wait barriers)
instead of a hardware scoreboard.

Per-cycle order (`SM::cycle()`, source of truth is the function itself):
L0↔L1 instruction interconnect, then the shared L1I/half-constant cache, then
the shared DP unit, then the shared load/store unit and Pending Request Table,
then every `Subcore::cycle()` (writeback → execute → read_rf → allocate →
control_stage → issue → decode → fetch, plus each subcore's private L0I/L0C
advance), then pending wait-barrier bookkeeping.

For the full design — dependency model, instruction supply path, execution
resources, shared memory pipeline, and source anchors — see the numbered
chapters under `docs/detailed-design/`, indexed at
`docs/detailed-design/README.md`.

## Fixed invariants vs. configurable support matrix

Earlier revisions of this page described the entire True-Path boolean set as
fixed invariants. That is only accurate for part of the set. The remodeled
trace product contract in
[`docs/plans/2026-07-15-remodeled-trace-p0-p1-semantic-repair.md`](docs/plans/2026-07-15-remodeled-trace-p0-p1-semantic-repair.md)
is the source of truth for what is fixed versus configurable; this section
summarizes it.

**Fixed** (required by the supported execution contract, §2 of that plan, plus
the remaining True-Path invariants not covered by the support matrix):

- `is_trace_mode = true`, `is_captured_from_binary = true`
- `is_SM_remodeling_enabled = true`, `gpgpu_sub_core_model = 1`
- `is_ibuffer_remodeled_enabled = true` — a required capability assertion for
  remodeled trace mode, not a runtime old/new IBuffer selector
- `is_remodeling_scoreboarding_enabled = false` — the one True-Path exception;
  dependency resolution always uses control bits, never the hardware scoreboard
- `is_dp_pipeline_shared_for_subcores`, `is_rf_cache_enabled`,
  `is_loog_enabled`, `is_vpreg_enabled` = `true`

**Configurable support matrix** (§3 of that plan; unlisted combinations are not
implicitly supported and must fail at configuration time rather than enter an
undefined mixed mode):

| Capability | Supported values |
| --- | --- |
| PRT selection | `OLDEST`, `SAME_LAST_WARP_ID_THEN_OLDEST`, `SAME_LAST_INST_PC_THEN_OLDEST`, `WARPID_N_CLUSTERS_WITH_OLDEST`, both dependency-counter policies |
| Interwarp coalescing | off; on with `OLDEST`; on with either dependency-counter PRT policy |
| Instruction prefetching | off / on |
| IBuffer coalescing | off / on (not an `is_*` flag) |
| Prefetch + IBuffer coalescing | on/on |
| FP32/INT topology | unified (shares the SP pipeline; FP32-to-INT steering is ignored because no independent INT pipeline exists) or separate |

## Source anchors

- `remodeling/sm.cc`: `SM::cycle`, `SM::instruction_retirement`
- `remodeling/subcore.cc`: `Subcore::cycle`, `Subcore::issue`, `Subcore::fetch`
- `remodeling/functional_unit.cc`: `functional_unit::release_read_barrier`
- `remodeling/ldst_unit_sm.cc`: `ldst_unit_sm::cycle`, `ldst_unit_sm::issue`,
  `PendingRequestTable`

All paths above are relative to
`simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/`. See
`docs/detailed-design/09-源码锚点索引.md` for the full anchor index.
