# Remodeled Trace Baseline Verification Record

## Status

The initial verification gate is runnable and produces deterministic observations. No simulator golden or hardware accuracy threshold is approved yet.

Scope decisions bound to this record:

- official execution mode: remodeled trace mode only;
- product capabilities retained: PRT alternative policies, interwarp coalescing, instruction prefetching, and IBuffer coalescing;
- hardware accuracy reference: RTX 4090, kept separate from deterministic simulator regression goldens.

## Checked-in artifacts

| Artifact | Purpose |
| --- | --- |
| `docs/plans/2026-07-15-remodeled-trace-refactoring-baseline.md` | Scope, baseline layers, risks, and acceptance criteria |
| `tests/remodeled_trace/run_regression.py` | Simulator observation and approved-golden guard |
| `tests/remodeled_trace/measure_shared_lat_rtx4090.py` | RTX 4090 sampling and simulator comparison |
| `tests/remodeled_trace/cases.json` | Runnable remodeled trace cases |
| `tests/remodeled_trace/goldens.json` | Explicitly unapproved golden set |
| `tests/remodeled_trace/fixtures/shared_lat_sm89.tar.gz` | Current-binary SM89 `shared_lat` trace fixture |
| `tests/remodeled_trace/fixtures/shared_lat_sm89.metadata.json` | Fixture provenance and member hashes |

Fixture archive SHA-256:

```text
f4bb8835e7498db396222bfe424d557aae5f1e07ac96720cbf262b90bd5bbaf3
```

## Local verification

| Command | Exit code | Result |
| --- | ---: | --- |
| `python3 -m unittest discover -s tests -v` | 0 | 18 tests passed |
| `python3 tests/remodeled_trace/run_regression.py list` | 0 | Listed SM89 `shared_lat` and SM86 `pathfinder` cases |
| `tar -tzf tests/remodeled_trace/fixtures/shared_lat_sm89.tar.gz` | 0 | Fixture archive readable |
| `python3 tests/remodeled_trace/run_regression.py observe --output /tmp/remodeled-trace-observations-final-v2.json` | 0 | Two observations completed; no failures or missing fields; comparison contracts recorded |
| Core-field `diff -u` between repeated SM89 observations | 0 | Inputs, process state, and all parsed stats identical |
| Core-field `diff -u` between repeated SM86 observations | 0 | Inputs, process state, and all parsed stats identical |
| `python3 tests/remodeled_trace/run_regression.py check --output /tmp/remodeled-trace-check-final-v2.json` | 4 | Expected refusal: both cases reported `missing_approved_golden`; simulator processes themselves exited 0 |
| `python3 tests/remodeled_trace/run_regression.py check --case ampere_pathfinder_sm86_observation --goldens /tmp/empty-approved-goldens.json --output /tmp/empty-approved-check.json` | 4 | Empty approved golden rejected as `invalid_approved_golden` after simulator process exited 0 |
| `python3 tests/remodeled_trace/measure_shared_lat_rtx4090.py --device 0 --warmups 1 --samples 10 --ncu-samples 5 --simulator-report /tmp/remodeled-trace-sm89-observation.json --output /tmp/rtx4090-shared-lat-comparison.json` | 0 | Hardware observation and simulator comparison generated |
| Source search for `smem_latency` and `m_latency_of_mem_operation_at_sm_structure` | 0 | Confirmed legacy and remodeled shared-memory latency controls are distinct |

The `check` exit code of 4 is the expected guard behavior, not a passing regression result.

## Simulator observations

Both cases ran with `OMP_NUM_THREADS=1`, found the completion marker, exited 0, and produced all required fields.

| Field | SM89 `shared_lat` | SM86 `pathfinder` |
| --- | ---: | ---: |
| `gpu_tot_sim_cycle` | 192,072 | 28,795 |
| `gpu_tot_sim_insn` | 32,881 | 804,960 |
| `gpu_tot_ipc` | 0.1712 | 27.9549 |
| `L2_total_cache_accesses` | 99 | 5,240 |
| `L2_total_cache_misses` | 99 | 3,455 |
| `total dram reads` | 96 | 3,035 |
| `total dram writes` | 0 | 0 |
| `gpgpu_n_shmem_bkconflict` | 0 | 0 |

Simulator binary SHA-256:

```text
31d0415bc8196004db6064008b94d4d085df12a09682457ab245e351808cd958
```

Source commit reported by the harness: `952ecad8b0aa894b521a8e81259cc0e50270ce94`.

The binary hash and current source commit are implementation provenance. Golden equality uses the checked trace/config identities plus trace path and runtime settings (`LC_ALL`, `OMP_DYNAMIC`, `OMP_NUM_THREADS`, timeout). This allows a rebuilt binary to be tested against an earlier approved behavior baseline without first editing the golden.

## RTX 4090 observation

Hardware identity:

- device index: 0;
- name: NVIDIA GeForce RTX 4090;
- UUID: `GPU-0a6e098f-a556-73f4-0a21-5151b876a20e`;
- compute capability: 8.9;
- driver: 565.57.01;
- Nsight Compute: 2024.3.2.0;
- `shared_lat` binary SHA-256: `d57c9cac81aabf32aa3b2f0539503f5e72d3c732683fcdfc4356ffb7b6bc23cc`.

The original target-index check found no compute process before sampling. GPU state changed from 210 MHz / 28 C / 8.62 W before sampling to 2520 MHz / 31 C / 84.06 W after sampling.

This sample set was collected before independent review changed `CUDA_VISIBLE_DEVICES` from numeric ordinal 0 to the queried GPU UUID. The final-code independent rerun below binds the measurement to `GPU-0a6e098f-a556-73f4-0a21-5151b876a20e` and confirms the original architecture-level result.

Direct, non-injected measurement after one warmup:

| Field | Samples | Min | Median | Max |
| --- | ---: | ---: | ---: | ---: |
| Shared-memory latency | 10 | 30.032715 cycles | 30.032715 cycles | 30.032715 cycles |
| Timed-region cycles | 10 | 61,507 | 61,507 | 61,507 |

Nsight whole-kernel measurement:

| Metric | Samples | Min | Median | Max |
| --- | ---: | ---: | ---: | ---: |
| `gpc__cycles_elapsed.avg` | 5 | 270,470.18 | 270,543.55 | 270,602.64 |
| `sm__cycles_elapsed.avg` | 5 | 270,471.88 | 270,543.56 | 270,602.80 |
| `sm__inst_executed.sum` | 5 | 32,885 | 32,885 | 32,885 |
| `smsp__inst_executed.sum` | 5 | 32,885 | 32,885 | 32,885 |
| `gpu__time_duration.sum` | 5 | 121,056 ns | 121,120 ns | 121,120 ns |

Comparison using aligned whole-kernel fields:

| Quantity | Simulator | Hardware median | Signed relative error | Absolute relative error |
| --- | ---: | ---: | ---: | ---: |
| Executed instructions | 32,881 | 32,885 | -0.012164% | 0.012164% |
| Whole-kernel cycles | 192,072 | 270,543.56 | -29.005148% | 29.005148% |

The instruction totals are closely aligned. The current simulator observation underestimates RTX 4090 whole-kernel cycles by about 29%; this is evidence of a performance-model discrepancy, not yet a localized cause.

Source-path verification excludes `-gpgpu_smem_latency` as the remodeled tuning control: that option feeds the legacy `ldst_unit`. The remodeled shared-memory path assigns each access from `memory_shared_memory_minimum_latency` plus instruction-dependent latency and advances it through `ldst_unit_sm::m_shmem_pipeline`; the SM89 config sets the minimum to 7 cycles. The hardware pointer-chasing result of about 30 cycles is not directly equal to that one pipeline segment because the remodeled model splits latency across subcore, coalescing, and SM stages. The 29% whole-kernel error therefore remains localized only to the broader timing model, not to one confirmed parameter.

## Independent agent rerun

The independent agent made no repository changes. It reran the final files and reported:

| Command | Exit code | Independent result |
| --- | ---: | --- |
| `python3 -m unittest discover -s tests -v` | 0 | 18 tests passed |
| `python3 tests/remodeled_trace/run_regression.py list` | 0 | Two cases listed |
| `tar -tzf tests/remodeled_trace/fixtures/shared_lat_sm89.tar.gz` plus SHA checks | 0 | Archive, archive hash, and member hashes match metadata |
| `python3 tests/remodeled_trace/run_regression.py observe --output /tmp/independent-remodeled-observations-after-review.json` | 0 | Both processes exited 0; no failures/missing stats; values match this record |
| `python3 tests/remodeled_trace/run_regression.py check --output /tmp/independent-remodeled-check-after-review.json` | 4 | Both cases correctly refused as `missing_approved_golden` |
| Empty approved golden integration check | 4 | Rejected as `invalid_approved_golden`; no bypass remains |
| `python3 tests/remodeled_trace/measure_shared_lat_rtx4090.py --device 0 --warmups 1 --samples 10 --ncu-samples 5 --simulator-report /tmp/independent-remodeled-observations.json --output /tmp/independent-rtx4090-comparison.json` | 0 | UUID-bound probe completed with no compute process before or after |

Independent RTX 4090 medians:

| Field | Independent median | Difference from primary sample |
| --- | ---: | ---: |
| Direct shared-memory latency | 30.032715 cycles | 0 |
| Direct timed-region cycles | 61,507 | 0 |
| `gpc__cycles_elapsed.avg` | 270,433.09 | -0.040832% |
| `sm__cycles_elapsed.avg` | 270,434.48 | -0.040319% |
| `sm__inst_executed.sum` | 32,885 | 0 |
| `smsp__inst_executed.sum` | 32,885 | 0 |
| `gpu__time_duration.sum` | 121,088 ns | -0.026420% |

The independent comparison reports instruction error `-0.012164%` and whole-kernel cycle error `-28.976512%`. The cycle-error result differs from the primary sample by 0.028636 percentage points, which is consistent with the observed hardware sampling spread.

## Diagnostic failures resolved during harness validation

| Failure | Exit code | Cause | Resolution |
| --- | ---: | --- | --- |
| Setup under shell `nounset` | 127 | `setup_environment_no_git.sh` reads undefined `IS_SERT` | Harness uses the repository-supported non-`nounset` setup contract and initializes `IS_SERT` |
| First harness simulator run | 3 | Setup path was passed as a positional argument and misread as `ACCELSIM_CONFIG` | Clear setup positional arguments before sourcing; covered by a unit test |
| First hardware probe | 2 | Nsight writes profiler/program text before the raw CSV header | Locate the raw `"ID",...` header before CSV parsing; covered by a unit test |
| Empty approved golden review case | 4 | An empty stats/input map previously had no comparison work and could pass | Approved schema must cover required stats and the full comparison contract; covered by unit and integration tests |
| Numeric GPU ordinal review case | N/A | CUDA ordinal can differ from the `nvidia-smi` index used by the busy check | `CUDA_VISIBLE_DEVICES` now uses the queried GPU UUID |
| Manifest trace filename review case | N/A | An unchecked `../` filename could escape the extracted trace root | Manifest trace root/file now use the same safe relative-path validation as archive members |

## Open verification obligations

- `goldens.json` remains `unapproved`; current observations are not regression pass oracles.
- The acceptable RTX 4090 cycle-error threshold has not been approved.
- HALF and FP64 checked-in trace fixtures are still missing.
- Stable per-pipeline dispatch statistics are still missing for HALF/SP/INT/DP/MEM semantic checks.
- PRT alternatives, interwarp coalescing, prefetching, and IBuffer coalescing do not yet have a completed support matrix.
- The trace fixture was originally generated using numeric CUDA ordinal 0. Its SM89 architecture and content hashes are verified, but its exact generating GPU UUID is intentionally recorded as unverified provenance.
- The independent rerun used Python 3.10.12. Python 3.8 syntax/runtime compatibility was reviewed but not executed on an Ubuntu 20.04 Python 3.8 environment.
- The GPU busy check has an unavoidable check-to-launch race; the independent run checked the UUID-bound GPU before and after sampling and found no compute process.
