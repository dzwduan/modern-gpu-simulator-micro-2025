# Legacy shader retirement — exec removal and single-core convergence

Branch: `refactor/legacy-shader-retirement`, starting point `2f67e34`.
Governing design: `docs/plans/2026-07-17-legacy-retirement-design.md` (§4.1, §5,
§7 step 1, §10 corrections).

## Baseline before surgery

- `python3 -m unittest discover -s tests` — exit 0, 21 tests OK.
- `python3 tests/remodeled_trace/run_regression.py check` — exit 0, 4/4 passed.

## Sub-commits

| Commit | Content | Build | Unit | Check |
| --- | --- | ---: | ---: | ---: |
| `db20070` | PTX perf-model entry point becomes a fatal stub; drop the argv defaults it solely used | 0 | 0 (21 OK) | 0 (4/4) |
| `2898a41` | Delete `exec_gpgpu_sim`, `exec_simt_core_cluster`, `exec_shader_core_ctx` and all their bodies | 0 | 0 (21 OK) | 0 (4/4) |
| `735aa37` | Trace cluster factory constructs `SM` unconditionally; delete `trace_shader_core_ctx` | 0 | 0 (21 OK) | 0 (4/4) |
| `62fe0cd` | Delete `-is_SM_remodeling_enabled` (option, member, validation early-return) + 39-file config sweep + golden reapproval | 0 | 0 (21 OK) | 0 (4/4) |

Net: 49 files changed, 15 insertions, 442 deletions.

## Zero-referrer evidence

- After the PTX stub, `grep -rn 'exec_gpgpu_sim\|exec_simt_core_cluster\|exec_shader_core_ctx'`
  over `gpgpu-sim/src`, `trace-driven`, `main.cc` matched only the definitions
  themselves; after `2898a41` it matches nothing.
- `trace_shader_core_ctx` after `735aa37`: no matches (the stale forward
  declaration in `remodeling/ibuffer_remodeled.h` was removed with it).
- `is_SM_remodeling_enabled` after `62fe0cd`: no matches in sources or configs.

## Preserved deliberately

- `core_t::updateSIMTStack` (base body): `SM` calls the inherited 2-arg form and
  does not override it. Only the 3-arg `trace_shader_core_ctx` overload died.
- Trace data types `trace_shd_warp_t`, `trace_kernel_info_t`,
  `trace_warp_inst_t`, `trace_function_info`: used by `SM`, untouched.
- `concrete_scheduler`, `scheduler_prioritization_type`, `pipeline_stage_name_t`
  and the options `-gpgpu_scheduler` / `-gpgpu_pipeline_widths`: retained per
  design §10 finding 3 — they are consumed by `gpgpu_sim_config::init()` and
  size `pipe_widths[N_PIPELINE_STAGES]` inside the KEEP class
  `shader_core_config`. Removing them would abort every run.

## Acceptance criterion: removed options fail loudly

Running the SM89 fixture against a config still carrying the deleted line,
before the sweep:

```
EXIT=1
GPGPU-Sim ** ERROR: Unknown Option: '-is_SM_remodeling_enabled'
```

This satisfies the roadmap requirement that deleted config options produce a
clear error rather than being silently ignored.

## Golden re-approval

Procedure per `2026-07-17-dead-weight.md`: `observe` on the committed tree, then
a script asserting per-case observed stats are byte-identical to the approved
goldens and that the comparison contract differs only in config `sha256`
fields.

```
ASSERTION PASSED: stats byte-identical across all 4 cases
config hash updates: 4 -> shared_lat/gpgpusim, half_pipeline/gpgpusim,
                          fp64_dispatch/gpgpusim, ampere_pathfinder/gpgpusim
```

Resulting `goldens.json` diff: 10 lines (4 config hashes + `source_commit`).
`check` after re-approval: exit 0, 4/4 passed. Stats identity proves the exec
removal and single-core convergence changed no simulated behavior.

## Dependency-direction ledger

Unchanged from the dead-weight baseline, as step 1 predicted (7 inbound files):

```
abstract_hardware_model.cc
gpgpu-sim/gpu-sim.h
gpgpu-sim/scoreboard.cc
gpgpu-sim/scoreboard_reads.cc
gpgpu-sim/shader.cc
gpgpu-sim/shader_core_wrapper.h
gpgpu-sim/shader.h
```

It did not grow. `shader.{h,cc}` still host the KEEP facilities and the legacy
pipeline machinery; those edges fall in the later steps.

## Open follow-ups for the next steps

- `shader_core_ctx` still exists and still carries the legacy pipeline; its
  `-Winconsistent-missing-override` warnings are pre-existing and resolve when
  the class is dissolved.
- `shader.cc` reports `total_num_sim_winsn_per_kernel` set-but-unused; inspect
  when the legacy body is dismantled rather than patching around it now.

## Review response

The independent review of `2f67e34..afe725c` found no blocking issue: "the code
and configuration removals are internally consistent with the trace-only
contract, and no remaining runtime references to the deleted implementations
were found." One P2 documentation mismatch was reported and fixed:

- `simulator-remodeled/gpu-simulator/README.md` described the `exec_`/`trace_`
  core class split (both now deleted) and advertised "vISA (PTX)
  execution-driven" support, which this step made fatal. It also linked the
  AccelWattch and gpgpu-sim4 documents deleted in earlier stages and described
  an upstream auto-clone setup this fork does not use. Every claim in the file
  was stale, so it is deleted rather than patched, consistent with the
  repo-hygiene stage's removal of its sibling upstream documents.
- `gpgpu-sim/setup_environment`, which `setup_environment_no_git.sh` sources on
  every build, printed a PTX/PTXPLUS advisory banner claiming PTX execution
  support. The banner block is removed; the script still reports
  `setup_environment succeeded`.

Post-fix gate: build exit 0; unit tests exit 0 (21 OK); regression check exit 0,
4/4 passed against unchanged goldens.

Remaining upstream document not touched in this step:
`gpgpu-sim/README.md` (503 lines, the upstream GPGPU-Sim manual) still documents
PTX execution and other removed subsystems. It is out of this step's scope and
is recorded here as a follow-up for the documentation pass.
