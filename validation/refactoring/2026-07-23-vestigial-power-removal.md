# Vestigial power surface removal

Collects the deferral recorded twice: `2026-07-17-dead-weight.md` item 2 kept
the AccelWattch option group and its XML data "so configs still parse" after
deleting the power backend, and `2026-07-17-legacy-retire-step5.md` kept
`-scoreboard_war_static_power` / `-scoreboard_war_dynamic_power` as belonging
to "the deferred vestigial-power cleanup". Working rules per
`docs/plans/2026-07-22-stage4-reorganization-design.md` (per-commit full gate,
golden drift stops the work).

Start HEAD `0369c30` (branch `dev_dzw`). Two commits: the first changes no
`.config` file and therefore runs against the unchanged goldens; the second
changes 31 configs and goes through a controlled golden re-approval.

## Environment

| Fact | Value |
| --- | --- |
| Build | `cd simulator-remodeled && source ./gpu-simulator/setup_environment_no_git.sh && make -j$(nproc) -C ./gpu-simulator/` |
| Gate (repo root) | `python3 -m unittest discover -s tests` and `OMP_NUM_THREADS=1 python3 tests/remodeled_trace/run_regression.py check` |
| Binary | `simulator-remodeled/gpu-simulator/bin/release/accel-sim.out` |

## Commits

| Hash | Subject |
| --- | --- |
| `c08b60a` | refactor: remove the unreferenced power option surface |
| `a15ecaa` | refactor: retire the scoreboard war power options |

## Gate table

Both commits were verified from a clean object tree (`make clean` followed by a
full rebuild) so no stale object could hide a missing include.

| Commit | build | unittest (count) | check (passed/total, observed_not_golden) | goldens |
| --- | ---: | --- | --- | --- |
| baseline `0369c30` | — | 0 (21) | 0 (4/4, 0) | unchanged |
| `c08b60a` | 0 | 0 (21) | 0 (4/4, 0) | untouched (no diff) |
| `a15ecaa` | 0 | 0 (21) | 0 (4/4, 0) | re-approved (config sha256 only) |

All four cases report `golden_mismatches: []` in both post-commit `check` runs.

## Unreachability argument

Three independent facts make the whole surface unreachable, not merely unused:

1. **The backend is gone.** `2026-07-17-dead-weight.md` item 2 (`7bcfcea`)
   deleted `src/accelwattch/`, `power_interface.{cc,h}`, `power_stat.{cc,h}`
   and the mcpat build path. `setup_environment` no longer defines
   `GPGPUSIM_POWER_MODEL`, so `-DGPGPUSIM_POWER_MODEL` is never on.
2. **The flag cannot be turned on.** `gpgpu_sim_config::validate_supported_trace_contract`
   rejected `-power_simulation_enabled 1` at startup with `exit(1)` (added as
   the fix for review finding 2 of the same record). Therefore
   `g_power_simulation_enabled` is false for every run that reaches the
   simulation loop.
3. **No config selects any of it.** Measured over
   `simulator-remodeled/gpu-simulator/gpgpu-sim/configs/**` and
   `simulator-remodeled/gpu-simulator/configs/**` before the change, each of
   `-accelwattch_xml_file`, `-power_simulation_mode`,
   `-accelwattch_hybrid_perfsim_*`, `-power_trace_enabled`,
   `-steady_power_levels_enabled`, `-power_per_cycle_dump`, `-dvfs_enabled`,
   `-aggregate_power_stats`, `-power_simulation_enabled`, `-hw_perf_file_name`,
   `-hw_perf_bench_name`, `-steady_state_definition`, `-power_trace_zlevel`
   appears in **0 files / 0 lines**. The only power-named options present were
   `-scoreboard_war_static_power` and `-scoreboard_war_dynamic_power`
   (31 files / 31 lines each).

The three `g_power_simulation_enabled` read sites were consequently
unreachable branches:

| Site | Function | Body |
| --- | --- | --- |
| `remodeling/functional_unit.cc` | `functional_unit::get_active_lanes_in_pipeline` | accumulates the pipeline active mask; with the guard false the function returns a constant `0` |
| `remodeling/sm.cc` | `SM::init` | the only assignment of `SM::m_scaling_coeffs` |
| `remodeling/sm.cc` | `SM::incexecstat` | the whole function body: the `sp_op` switch over the per-opcode power accounting helpers plus the `inc_const_accesses(1)` tail |

### Per-body disposition

- **`get_active_lanes_in_pipeline`** — the guarded loop writes only a function
  local, so removing it preserves the returned value (`0`). Its only callers
  were `functional_unit::active_lanes_in_pipeline` and
  `ldst_unit_sm::active_lanes_in_pipeline`, and a repo-wide grep found **zero**
  callers of either. The whole chain is deleted, which in turn leaves
  `SM::incspactivelanes_stat`, `incsfuactivelanes_stat`, `incfuactivelanes_stat`
  and `incfumemactivelanes_stat` callerless; they are deleted too.
- **`SM::init`** — the guarded assignment is the only writer of
  `m_scaling_coeffs`. The member is *not* initialised in the `SM` constructor
  and `~SM()` unconditionally executed `delete m_scaling_coeffs;`, i.e. a delete
  of an indeterminate pointer whenever `~SM` runs. Deleting the member together
  with that `delete` removes the only remaining reference and closes the latent
  undefined behaviour rather than preserving it; this is reported rather than
  silently folded in. `gpgpu_sim::get_scaling_coeffs()` (which had returned
  `NULL` since the backend removal) and the `PowerscalingCoefficients` struct
  then have no referrers and are deleted.
- **`SM::incexecstat`** — the guard is the entire body, so removing it leaves an
  empty function with two call sites (`functional_unit::issue`,
  `ldst_unit_sm`'s shared-memory dispatch). The function, its declaration and
  both call sites are deleted; the seventeen per-opcode helpers it drove
  (`incialu_stat` … `inctex_stat`), `inc_const_accesses`, and
  `inactive_lanes_accesses_sfu` then reach zero callers and are deleted.
  `inactive_lanes_accesses_nonsfu` is **retained**: `incmem_stat` still calls it
  on the live memory path.

`inst->sp_op` remains live and untouched: `trace-driven/trace_driven.cc` sets it
from `OpcodePowerMap` (`ISA_Def/accelwattch_component_mapping.h`) and consumes it
through `get_oprnd_type`, which drives operand typing, not power. That header and
its include therefore stay.

### Retained stat fields

The `shader_core_stats` arrays these helpers wrote (`m_num_ialu_acesses` …
`m_num_tex_acesses`, `m_active_exu_threads`, `m_active_exu_warps`,
`m_active_sp_lanes`, `m_active_sfu_lanes`, `m_active_fu_lanes`,
`m_active_fu_mem_lanes`) keep their declarations, allocation and free. They were
already `0` for every run — their writers only ran under the false guard, or in
the case of the active-lane counters were never called at all — so removing the
writers is value-preserving. They have no readers left in the tree (the power
backend was their consumer); collapsing the arrays themselves is a
`shader_core_stats` change outside this cleanup and is noted here as the
remaining follow-up. The twelve `warp_inst_t::is_fp()/is_dp()/is_sfu()/…`
`sp_op` classifiers in `abstract_hardware_model.h` are likewise callerless
power-model leftovers that predate this change and are left in place.

## What was deleted

### `c08b60a` — unreferenced power surface (no `.config` file touched)

Symbols and files:

- `gpu-sim.h`: `enum hw_perf_t`, `struct power_config` (ctor, dtor, `init`,
  `reg_options` declaration and all 25 members), `power_config` as a base of
  `gpgpu_sim_config`, the `power_config::init()` call in `gpgpu_sim_config::init`,
  the `m_valid = false/true` assignments (their only declaration was
  `power_config::m_valid`, and nothing read it — `memory_config::m_valid` and
  `shader_core_config::m_valid`, the two flags `mem_latency_stat.cc` asserts on,
  are separate members), and the `get_scaling_coeffs` declaration. −148 lines.
- `gpu-sim-config.cc`: `power_config::reg_options` in full (119 lines, 29 option
  registrations), its call from `gpgpu_sim_config::reg_options`, and the
  `-power_simulation_enabled` rejection in `validate_supported_trace_contract`.
  −124 lines.
- `gpu-sim.cc`: `gpgpu_sim::get_scaling_coeffs()`. −6 lines.
- `abstract_hardware_model.h`: `struct PowerscalingCoefficients` and its
  `COEFF_STRUCT` guard. −24 lines.
- `remodeling/sm.{h,cc}`: `m_scaling_coeffs` + its `delete` in `~SM`, the
  `SM::init` guard, `incexecstat`, seventeen `inc*_stat` opcode helpers,
  `inc_const_accesses`, `inactive_lanes_accesses_sfu`, and the four
  `inc*activelanes_stat` helpers. −316 / −27 lines.
- `remodeling/functional_unit.{h,cc}`: `get_active_lanes_in_pipeline`,
  `active_lanes_in_pipeline`, the `incexecstat` call in `issue`. −38 / −3 lines.
- `remodeling/ldst_unit_sm.{h,cc}`: the `active_lanes_in_pipeline` override and
  the `incexecstat` call with its now-unused `inst_ptr` local. −9 / −1 lines.

Option list removed from the parser (29 registrations): `-accelwattch_xml_file`,
`-power_simulation_enabled`, `-power_per_cycle_dump`, `-hw_perf_file_name`,
`-hw_perf_bench_name`, `-power_simulation_mode`, `-dvfs_enabled`,
`-aggregate_power_stats`, the seventeen `-accelwattch_hybrid_perfsim_*`
variants, `-power_trace_enabled`, `-power_trace_zlevel`,
`-steady_power_levels_enabled`, `-steady_state_definition`.

Data and build:

- 31 `accelwattch_sass_sim.xml` files under
  `gpgpu-sim/configs/tested-cfgs/*` — 19,003 lines, 1,155,277 bytes — removed
  with `git rm`. Zero referrers: no config set `-accelwattch_xml_file`, and the
  option itself is gone.
- `remodeling/Makefile` and `remodeling/fusedMemory/Makefile`: the always-empty
  `POWER_FLAGS` blocks and their use in the compile rule (the
  `remodeling/` boundary that made `2026-07-17-dead-weight.md` retain them no
  longer applies).

Documentation:

- `gpgpu-sim/README.md`: the "options used to enable AccelWattch" block and the
  GPUWattch/AccelWattch history paragraph, plus the power claim in the opening
  description. The AccelWattch citation request and the authorship/contributors
  section are attribution and are kept, as are `gpgpu-sim/{CHANGES,COPYRIGHT}`.
- `gpgpu-sim/setup_environment`: the leftover "power model removed" marker
  comment.

Commit totals: 51 files changed, 6 insertions, 19,787 deletions.

### `a15ecaa` — scoreboard war power options (config change)

- `gpu-sim-config.cc`: the `-scoreboard_war_static_power` and
  `-scoreboard_war_dynamic_power` registrations. −10 lines.
- `shader.h`: the `scoreboard_war_static_power` / `scoreboard_war_dynamic_power`
  `shader_core_config` members and their `// MOD.` bracket. −4 lines.
- 31 `gpgpusim.config` files: the five-line block carrying both options
  (`# MOD. Begin. Fix WAR at baseline. …`, the two option lines, `# MOD. End`,
  and the trailing blank) — 155 lines total. Reader check before deleting: a
  repo-wide grep found only the two registrations and the two member
  declarations; no code reads either member, matching the finding recorded in
  `2026-07-17-legacy-retire-step5.md`.
- `tests/remodeled_trace/goldens.json`: four config sha256 lines.

Commit totals: 34 files changed, 4 insertions, 173 deletions.

## Sweep coverage

Deleted-name sweep beyond `src/`, per the discipline note at the end of
`2026-07-17-dead-weight.md` (option sweeps must cover `util/tuner/**` and
`util/job_launching` yml `extra_params` emitters):

| Location | Finding | Action |
| --- | --- | --- |
| `util/tuner/config_template/gpgpusim.config` | `-power_simulation_enabled 0` | line + its comment removed |
| `util/tuner/NVIDIA_GeForce_RTX_4090/gpgpusim.config` | `-power_simulation_enabled 0` | line + its comment removed |
| `util/job_launching/common.py` | `-a/--accelwattch_HW` command-line option | removed |
| `util/job_launching/run_simulations.py` | `options.accelwattch_HW` → emits `-hw_perf_bench_name` into the generated config | removed |
| `util/job_launching/get_stats.py` | reads `accelwattch_power_report.log` and the `power_stats` yml key (four blocks) | removed |
| `util/job_launching/stats/example_stats.yml` | `power_stats:` section (23 regexes) | removed |
| `util/job_launching/configs/*.yml` | no power `extra_params` remain (the `PWR_ENABLE` / `Accelwattch_*` emitters and `apps/define-power.yml` went in the earlier review response) | none needed |
| `ISA_Def/accelwattch_component_mapping.h`, `trace-driven/trace_driven.cc` | `OpcodePowerMap` still feeds the live `sp_op` → `get_oprnd_type` operand typing | kept |
| `gpgpu-sim/{CHANGES,COPYRIGHT}`, README citation/authorship | third-party attribution | kept |

Post-sweep grep for `accelwattch|power_simulation|power_trace|steady_power|power_per_cycle|dvfs_enabled|aggregate_power_stats|hw_perf_*|steady_state_definition|power_trace_zlevel|PowerscalingCoefficients|power_config|hw_perf_t|POWER_FLAGS|GPGPUSIM_POWER_MODEL|g_power|incexecstat|active_lanes_in_pipeline|scaling_coeffs|scoreboard_war` over the tracked tree returns nothing outside `docs/`, `validation/`, the kept attribution files, and the kept `OpcodePowerMap` pair.

## Retired-option demo

Scratch config outside the repo tree
(`/tmp/…/retired_option.config`, a copy of `SM89_RTX4090/gpgpusim.config` with
one retired option appended), run against the checked-in HALF fixture trace
extracted to a scratch directory:

```
cd simulator-remodeled
source ./gpu-simulator/setup_environment_no_git.sh
LC_ALL=C OMP_DYNAMIC=FALSE OMP_NUM_THREADS=1 \
  gpu-simulator/bin/release/accel-sim.out \
  -trace <scratch>/half_pipeline_sm89/traces/dynamic_trace.pb \
  -config <scratch>/retired_option.config \
  -config gpu-simulator/configs/tested-cfgs/SM89_RTX4090/trace.config
```

| Appended line | Exit code | Message |
| --- | ---: | --- |
| `-scoreboard_war_static_power 0` | 1 | `GPGPU-Sim ** ERROR: Unknown Option: '-scoreboard_war_static_power'` |
| `-power_simulation_enabled 1` | 1 | `GPGPU-Sim ** ERROR: Unknown Option: '-power_simulation_enabled'` |

The second row is why the bespoke `-power_simulation_enabled` rejection inside
`validate_supported_trace_contract` could be deleted with the option: the
parser now fails such a config loudly and earlier.

## Golden evidence

### `c08b60a` — goldens untouched

`git status` reports no modification to `tests/remodeled_trace/goldens.json` in
this commit, and none of the four hashed files
(`gpgpu-sim/configs/tested-cfgs/{SM86_RTX3080,SM89_RTX4090}/gpgpusim.config`,
`configs/tested-cfgs/{SM86_RTX3080,SM89_RTX4090}/trace.config`) is in the commit
— only the sibling `accelwattch_sass_sim.xml` files in those directories, which
the contract does not hash. `check` against the **existing** goldens: exit 0,
`{"total": 4, "passed": 4, "failed": 0, "observed_not_golden": 0}`, every case
`golden_mismatches: []`.

### `a15ecaa` — re-approval

Procedure per the end of `2026-07-17-dead-weight.md`: `observe` on the changed
tree (exit 0, four cases `observed_not_golden`), then a script asserting that
each case's observed stats are byte-identical to the approved goldens and that
the comparison contract differs in nothing but the config `sha256`:

```
[OK] sm89_shared_lat_rtx4090_observation: stats_identical=True contract_diffs=['configs.gpgpusim.sha256']
[OK] sm89_half_pipeline_rtx4090_semantic: stats_identical=True contract_diffs=['configs.gpgpusim.sha256']
[OK] sm89_fp64_dispatch_rtx4090_semantic: stats_identical=True contract_diffs=['configs.gpgpusim.sha256']
[OK] ampere_pathfinder_sm86_observation: stats_identical=True contract_diffs=['configs.gpgpusim.sha256']
ALL_OK
```

goldens.json diff — four lines, config hashes only:

- SM86_RTX3080 gpgpusim: `3245e2c0…bff71` → `ba2a7c81…d5f66` (1 case)
- SM89_RTX4090 gpgpusim: `3d3f6a43…1ef1d` → `3f0b5c8a…2f9cc` (3 cases)

`source_commit` is left at `0d357ce6…`: observed stats are byte-identical, so
behavioural provenance is unchanged and the re-approval is hashes-only — the
convention of `62fe0cd` and of step 5b in
`2026-07-17-legacy-retire-step5.md`. `check` afterwards: exit 0, 4/4,
`observed_not_golden 0`.

## Reverse-include ledger

Run in `simulator-remodeled/gpu-simulator/gpgpu-sim/src`:
`grep -rln '#include.*remodeling/' --include=*.cc --include=*.h . | grep -v '/remodeling/' | sort`

```
gpgpu-sim/gpu-sim.h
gpgpu-sim/shader_core_wrapper.h
```

Two files, unchanged from the stage-four S5 result. Both are the permitted
L3→L2 downward edges of `docs/plans/2026-07-22-stage4-reorganization-design.md`
§2; no new low-to-high include was introduced.

## Deviations and notes

- **Latent `delete` on an indeterminate pointer found and closed.**
  `SM::m_scaling_coeffs` was never initialised and never assigned (its only
  assignment sat behind the always-false guard), yet `~SM()` deleted it. The
  member and the `delete` are removed together. Reported rather than folded in
  silently because it is a behaviour change on the `~SM` path, not a pure
  deletion — though that path could only ever have been undefined behaviour.
- **Deletion closure beyond the three guards.** Removing the guarded bodies
  leaves `SM::incexecstat` empty and `get_active_lanes_in_pipeline` constant, so
  the functions and everything that reaches zero callers with them are deleted
  in the same commit rather than left as fresh dead weight. Each removed
  function was checked for remaining callers first;
  `inactive_lanes_accesses_nonsfu`, `incsp_stat`, `incsfu_stat`, `incmem_stat`,
  `incregfile_reads/writes`, `incnon_rf_operands` and `inc_simt_to_mem` still
  have live callers and stay.
- **Write-only stat arrays left in place.** The `shader_core_stats` power arrays
  now have neither writer nor reader but keep their declaration, allocation and
  free; collapsing them is a `shader_core_stats` change deliberately kept out of
  this cleanup.
- **Pre-existing callerless helpers left in place.** `SM::incload_stat`,
  `SM::incstore_stat` and the twelve `warp_inst_t` `sp_op` classifiers had no
  callers before this change either; they are not part of the reachability
  argument and are left for a later pass.
- **`.gitignore`** carries an unrelated uncommitted change that is not part of
  this work and was left unstaged in both commits.

## Independent rerun

Not performed as part of this record: an independent read-only rerun of the gate
on the committed HEAD is the coordinator's responsibility per the stage
execution model. Both commits were verified here from a clean object tree
(`make clean` + full rebuild) with the commands in the environment table, and
are reproducible from a clean checkout.
