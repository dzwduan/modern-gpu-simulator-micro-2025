# Remodeled Trace P0/P1 Semantic Repair Spec

## 1. Objective

Close the P0 simulation-trust findings and the P1 support-contract findings
identified during the repository structure review, now tracked in
`docs/plans/2026-07-17-structure-refactor-roadmap.md`, without starting the P2
architecture refactor. Every behavioral change must be reproduced first,
verified with a checked-in test or fixture, and recorded in a checked-in
validation artifact.

## 2. Supported Execution Contract

The only supported execution mode is remodeled trace mode. Its required runtime
contract is:

- trace execution is enabled;
- `is_SM_remodeling_enabled=1`;
- `gpgpu_sub_core_model=1`;
- `is_ibuffer_remodeled_enabled=1`;
- the trace carries captured binary/control-bit information;
- the remodeled IBuffer is the only instruction-buffer implementation used by
  the supported path.

PTX execution and the legacy shader core remain available upstream code paths,
but they are outside this product contract and are not regression oracles. A
request that combines the remodeled core with PTX execution, the legacy
IBuffer, or a disabled sub-core model must fail during configuration rather
than enter an undefined mixed mode.

## 3. Product Capability Matrix

The following remodeled trace capabilities remain in product scope:

| Capability | Supported values in this increment | Required verification |
| --- | --- | --- |
| PRT selection | `OLDEST`, `SAME_LAST_WARP_ID_THEN_OLDEST`, `SAME_LAST_INST_PC_THEN_OLDEST`, `WARPID_N_CLUSTERS_WITH_OLDEST`, both dependency-counter policies | Every policy completes a memory-intensive trace without losing an entry; dependency policies use a compatible interwarp policy |
| Interwarp coalescing | off; on with `OLDEST`; dependency tracking modes required by PRT | A single-variable on/off run plus the two dependency-policy pairings |
| Instruction prefetching | off/on | Both values complete with required statistics |
| IBuffer coalescing | off/on | Both values complete with required statistics |
| Prefetch + IBuffer coalescing | on/on | The stream-buffer fill path completes without invalid memory access |
| FP32/INT topology | separate or unified | Unified mode ignores FP32-to-INT steering because no independent INT pipeline exists |

Other Cartesian products are not implicitly supported. They require a new
matrix entry before being described as valid.

## 4. Finding Disposition

### P0 simulation trust

| Finding | Disposition | Reproduction and oracle |
| --- | --- | --- |
| P0-01 uninitialized `m_scaling_coeffs` | Initialize the non-owning pointer to `nullptr`; retain existing conditional assignment and unconditional null-safe delete | A focused source-contract test fails when the member has no default initialization; normal-shutdown trace runs remain clean |
| P0-02 HALF fallthrough | Stop `HALF_OP` after selecting the SP pipeline; size the SP pipeline from parsed trace latency so HALF cannot index beyond the FP32-only depth | A checked-in SM89 HALF trace must report HALF instructions, zero HALF dispatches to INT, and complete without a pipeline bounds fault |
| P0-03 DP throttle source | Apply the functional unit's constructor-supplied throttle instead of the MEM configuration field | A checked-in SM89 FP64 trace must report DP dispatches and a DP throttle sum equal to the configured DP throttle per throttled completion |
| P0-04 missing oracle | Promote deterministic post-fix observations only | Two identical local runs plus an independent run are required before approving exact goldens |

Pipeline observations are counted at successful issue, not at scheduler probe,
so retries cannot inflate the semantic counters.

### P1 support contract

| Finding | Disposition | Verification |
| --- | --- | --- |
| P1-01 PRT preferred entry loss | Preferred selectors return an eligible ID without erasing it; one centralized selection operation reserves resources, erases pending state, and appends the current set exactly once | All six PRT policies complete the memory/cache fixture; focused tests cover preferred and fallback bookkeeping |
| P1-02 prefetch/coalescing UAF | Recreate a `mem_fetch` before dereference whenever the status records that the caller deleted the original request | Prefetch+IBuffer-coalescing cross-case completes; sanitizer use is recorded when available |
| P1-03 remodeled PTX downcast | Reject remodeled-core PTX mode at configuration time; keep defensive trace checks before trace-only casts | Invalid mixed mode is covered by the configuration contract and source tests |
| P1-04 legacy scheduler ID | Keep legacy/sub-core-off outside the supported matrix; reject sub-core-off when the remodeled core is enabled | Invalid remodeled configuration exits before simulation |
| P1-05 ineffective IBuffer flag | Define the flag as a required capability assertion for remodeled trace mode, not a runtime old/new selector | `0` is rejected for remodeled trace; `1` is exercised by every supported case |
| P1-06 impossible documented path | Allow unified FP32/INT with FP32-to-INT configured, while suppressing steering to a nonexistent INT unit | Canonical unified configuration completes and dispatches through SP |

## 5. Checked-in Test Assets

- CUDA sources for deterministic HALF and FP64 microbenchmarks;
- SM89 trace archives generated on the UUID-bound RTX 4090;
- metadata containing source, binary, tracer, archive, and member hashes;
- aggregate dispatch and shared-throttle statistics parsed by the regression
  harness;
- manifest support-matrix cases derived from a checked base configuration with
  hashed, explicit option overrides;
- approved post-fix goldens for deterministic core cases;
- a validation record containing pre-fix evidence, post-fix commands, exit
  codes, and summaries.

Generated binaries, raw trace directories, simulator logs, and temporary
effective configurations are not checked in.

## 6. Verification Order

1. Install and record `ruff` in a user-isolated tool environment.
2. Add observation counters and fixtures without changing defective control
   flow.
3. Capture the pre-fix semantic observations or explicit failure signatures.
4. Apply one finding's repair and run its focused verification before moving to
   the next finding.
5. Rebuild the simulator and run unit tests, semantic cases, exact golden cases,
   support-matrix cases, and the RTX 4090 comparison.
6. Record all final commands and outputs in a checked-in validation artifact.
7. Have an independent agent rerun the checked-in commands without modifying
   repository files.

## 7. Exit Criteria

P0 is complete only when:

- all three semantic defects have a pre-fix reproduction and post-fix oracle;
- HALF and FP64 fixtures are checked in with provenance;
- deterministic required fields match approved, schema-complete goldens across
  two local runs and one independent run;
- existing shared-memory and memory/cache cases still complete.

P1 is complete only when:

- every supported PRT policy and required interwarp pairing completes;
- prefetch, IBuffer coalescing, and their on/on combination complete;
- supported unified FP32/INT mode completes;
- unsupported remodeled mixed modes fail early with explicit diagnostics;
- documentation and executable configuration validation describe the same
  contract.

If a required fixture cannot be generated, a policy cannot be reached by a
checked-in case, or a required run is nondeterministic, work stops at the last
verified increment and the remaining coverage is reported rather than marked
complete.
