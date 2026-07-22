# Legacy shader retirement — retire the scoreboard dependency mode

Branch: `refactor/legacy-shader-retirement`. Governing design:
`docs/plans/2026-07-17-legacy-retirement-design.md` §0.2, §6, §7 step 5, §8
(R11/R12), §9, §10 findings 1/3/4/5/6. Predecessor ledger:
`2026-07-17-legacy-retire-step4.md`. Start HEAD `ff3a064`.

Two commits. `5a` makes the issue path control-bit-only in pure code (goldens
byte-identical). `5b` deletes the now-unreferenced scoreboard classes, options,
and configs (config change → golden re-approval, stats byte-identical).

Scoreboard mode was a **deliberately-retired advertised feature**, not dead
legacy: it was README feature #4 ("Configurable dependence handling: scoreboards
or control bits") plus #5/#6, and was exercised by the eight `SM86_RTXA6000_SC_*`
configs (which set `-is_remodeling_scoreboarding_enabled 1`). The user's
control-bit-only scope decision (§0.2) retires it; the retirement is
gate-verifiable because every gate input already ran pure control-bit mode
(R11 ship gate, below).

## Environment

| Fact | Value |
| --- | --- |
| Build | `cd simulator-remodeled && source ./gpu-simulator/setup_environment_no_git.sh && make -j$(nproc) -C ./gpu-simulator/` |
| Gate | `python3 -m unittest discover -s tests` (21 tests) and `OMP_NUM_THREADS=1 python3 tests/remodeled_trace/run_regression.py check` (from repo root) |
| Binary | `simulator-remodeled/gpu-simulator/bin/release/accel-sim.out` |

## Commits

| Step | Hash | Subject |
| --- | --- | --- |
| 5a | `fc7aeedcd96c894527ca28640f006ab8144e7ff3` | refactor: make the issue path control-bit only |
| 5b | `0dd803891caef196e2d5ddd1bdb6a8196f0b9608` | refactor: retire the scoreboard dependency mode |

## Gate table

| Commit | build | unittest (count) | check (passed/total, observed_not_golden) | goldens |
| --- | ---: | --- | --- | --- |
| 5a `fc7aeed` | 0 | 0 (21) | 0 (4/4, 0) | byte-identical, unchanged |
| 5b `0dd8038` | 0 | 0 (21) | 0 (4/4, 0) | re-approved (config sha256 only) |

## Commit 5a — issue path control-bit only (pure code, goldens byte-identical)

Sites changed (the `use_traditional_scoreboarding` flag and its threading, plus
the two unconditional scoreboard reads, normalized to the control-bit path):

- `remodeling/subcore.cc`
  - `Subcore::control_stage` (`m_has_perform_control_stage` block, §10 finding 1):
    removed the guard `is_trace_mode && !((!captured) || is_remodeling_scoreboarding_enabled)`;
    the control-bit wait-barrier increment is now unconditional.
  - `Subcore::issue`: deleted the `bool use_traditional_scoreboarding = ...` local
    and the `if(use_traditional_scoreboarding){ checkCollision_remodeling ... } else { ... }`
    branch, keeping only the control-bit arm (`is_stall_counter_0` / wait barriers /
    yield); dropped `are_traditional_scoreaboards_ready` from
    `are_switch_warp_conditions_ready`; dropped the flag from the `issue_warp(...)` call.
  - `Subcore::issue_warp`: removed the `bool use_traditional_scoreboarding` parameter
    and the argument passed to `SM::issue_warp`.
- `remodeling/subcore.h`: dropped the parameter from the `Subcore::issue_warp` decl.
- `remodeling/sm.cc`
  - `SM::instruction_retirement`: deleted the scoreboard release branch
    (`m_scoreboard[_WAR]->releaseRegisters[_remodeling]`, `getMode()`), kept the
    control-bit else (write wait-barrier decrement + ldgsts accounting).
  - `SM::issue_warp`: removed the `use_traditional_scoreboarding` parameter and the
    scoreboard reserve block (`reserveRegisters[_remodeling]`, `isEnabled()`), kept the
    control-bit else (yield / stall counter / ldgsts).
  - `SM::check_if_warp_has_finished_executing_and_can_be_reclaim`: dropped
    `!m_scoreboard->pendingWrites(warp_id) && !m_scoreboard_WAR->pendingReads(warp_id) &&`.
  - `SM::warp_waiting_at_mem_barrier`: deleted the `use_traditional_scoreboarding` arm;
    `clear_membar = are_all_wait_barrier_ready(warp_id)` is now unconditional.
- `remodeling/sm.h`: dropped the parameter from the `SM::issue_warp` decl.
- `kernel-scheduler.cc` `kernel_scheduler::launch` (§10 finding 3 — the real anchor,
  `add_kernel` does not exist): added a fatal check BEFORE the running-kernel
  state mutation — `if (!kinfo->is_captured_from_binary) { printf(... "remodeled
  trace mode requires captured-from-binary kernels; this kernel is not
  captured." ...); exit(1); }`.

Line delta: 6 files, +50 / -113.

### Golden-neutrality argument (5a)

The scoreboard is populated ONLY by `reserveRegisters*`, which ran only inside the
`use_traditional_scoreboarding`-guarded reserve block. On the captured/control-bit
path that guard is false, so the scoreboard is PROVABLY ALWAYS EMPTY; therefore
every scoreboard read (`pendingWrites` / `pendingReads`) already returned false and
every scoreboard branch was already not taken. Deleting them is behavior-neutral on
the control-bit path. The `control_stage` guard is TRUE on the captured path, so
making the increment unconditional preserves it. Making the non-captured reject
fatal is unreachable for the (all-captured) fixtures.

Evidence: `run_regression.py check` on the 5a tree reported
`passed 4, failed 0, total 4, observed_not_golden 0`, all four cases
`golden_mismatches` empty, against the UNCHANGED approved goldens
(no goldens.json edit in 5a). Unit tests 21/21 OK.

## Commit 5b — delete scoreboard classes, options, configs (golden re-approval)

Sites changed:

- Deleted `scoreboard.h` (101), `scoreboard.cc` (276), `scoreboard_reads.h` (94),
  `scoreboard_reads.cc` (253) — 724 lines of retired class code. Removed the two
  `#include "scoreboard*.h"` from `shader.h`.
- `remodeling/sm.h`: removed the `Scoreboard`/`Scoreboard_reads` forward decls, the
  `get_scoreboard`/`get_scoreboard_WAR` getter decls, and the `m_scoreboard` /
  `m_scoreboard_WAR` members. `remodeling/sm.cc`: removed their construction
  (`make_shared<Scoreboard[_reads]>`), the two args passed to the `ldst_unit_sm`
  ctor, and the two getter definitions.
- `remodeling/ldst_unit_sm.{h,cc}`: removed the `m_scoreboard` / `m_scoreboard_reads`
  dead-store members and the `scoreboard`/`scoreboard_reads` parameters from both
  constructors and `init()` (dead stores — assigned in the ctor, never read).
- `gpu-sim-config.cc`: removed the `-is_remodeling_scoreboarding_enabled`,
  `-scoreboard_war_mode`, `-scoreboard_war_max_uses_per_reg` registrations.
  `gpu-sim.h` `set_custom_options`: removed the `scoreboard_war_mode` →
  `scoreboard_war_reads_mode` parse (§10 finding 1). `shader.h`: removed the
  `scoreboard_war_mode`, `scoreboard_war_reads_mode`, `scoreboard_war_max_uses_per_reg`,
  and `is_remodeling_scoreboarding_enabled` members (the `scoreboard_reads_mode` enum
  dies with `scoreboard_reads.h`, now unreferenced). `validate_supported_trace_contract`
  does not reference `is_remodeling_scoreboarding_enabled` (no change needed there).
- `git rm` the eight `gpgpu-sim/configs/tested-cfgs/SM86_RTXA6000_SC_*` directories
  (gpgpusim.config + xml + icnt) AND their eight paired
  `configs/tested-cfgs/SM86_RTXA6000_SC_*` directories (trace.config) — see Deviations.
  Removed the eight `RTXA6000_SC_*` `base_file` aliases from
  `util/job_launching/configs/define-standard-cfgs.yml` (§10 finding 5).
- Swept `-is_remodeling_scoreboarding_enabled`, `-scoreboard_war_mode`,
  `-scoreboard_war_max_uses_per_reg` from the 31 remaining `gpgpusim.config` files
  (39 total − 8 deleted SC = 31). Verified 0 residual occurrences.
- `README.md`: rewrote feature #4 to control-bit dependency handling only, removed
  the scoreboard features #5/#6, renumbered #7–#15 → #5–#13, and updated the
  "see feature 13" cross-reference to "see feature 11".

Retained (§6 stat-field caveat / R12): the scoreboard-named `shader_core_stats`
fields (`num_scoreboard_reads_check_collision`,
`num_scoreboard_reads_collision_due_to_max_uses_per_reg`,
`num_scheduler_stall_cycle_due_to_war_scoreboard`,
`num_scheduler_stall_cycle_dependencies_other_reasons_not_war_scoreboard`) — they
print into stdout via `shader.cc` and stay `0` (they were already `0` on the
control-bit path since `checkCollision`/`reserveRegisters` never ran), so removing
their only writer (`scoreboard_reads.cc`) is byte-identical.

Line delta: 77 files, +21 / -9246 (the bulk is the eight deleted SC config trees).

### Golden re-approval (5b)

Procedure (per `2026-07-17-dead-weight.md` end / commit `62fe0cd` convention):
`observe` on the changed tree, then assert per-case observed stats byte-identical to
the approved goldens AND the comparison contract differs ONLY in
`configs.gpgpusim.sha256`, then update goldens.json config hashes minimally.

Assertion script output (all four cases):

```
[OK] sm89_shared_lat_rtx4090_observation:   stats_identical=True contract_diffs=['configs.gpgpusim.sha256']
[OK] sm89_half_pipeline_rtx4090_semantic:   stats_identical=True contract_diffs=['configs.gpgpusim.sha256']
[OK] sm89_fp64_dispatch_rtx4090_semantic:   stats_identical=True contract_diffs=['configs.gpgpusim.sha256']
[OK] ampere_pathfinder_sm86_observation:    stats_identical=True contract_diffs=['configs.gpgpusim.sha256']
ALL_OK
```

goldens.json diff — minimal, four config sha256 lines only (three SM89 cases share
one hash, one SM86 case):

- SM89_RTX4090 gpgpusim: `aaacb201…de39` → `a222bd3a…c197f`
- SM86_RTX3080 gpgpusim: `8b483b14…877b` → `0ab42905…f023f`

`source_commit` left unchanged at `735aa37b…` (the last behavior-defining commit):
observed stats are byte-identical, so the behavioral provenance is unchanged, and the
minimal re-approval diff is config-hashes-only — the exact convention of the
analogous prior commit `62fe0cd` ("remove the sm model selection option"). `check`
then reports `passed 4, failed 0, total 4, observed_not_golden 0`.

### R11 ship gate — captured-from-binary proof (all four cases, manifest-driven)

Per §10 finding 6, verified every case archive in `cases.json` (not just the three
`fixtures/`), including the fourth (`rodinia2Ampere` pathfinder):

```
shared_lat_sm89.tar.gz:   "is_captured_from_binary":true
half_pipeline_sm89.tar.gz:"is_captured_from_binary":true
fp64_dispatch_sm89.tar.gz:"is_captured_from_binary":true
rodinia2Ampere.tar.gz:    "is_captured_from_binary":true
```

Both gate configs (`SM89_RTX4090`, `SM86_RTX3080`) set
`-is_remodeling_scoreboarding_enabled 0` (now removed; default was `0`). So the
deleted scoreboard branch was never taken by any gate input — the byte-identical
`check` proves the SURVIVING control-bit path is unchanged, not that the removed
branch was exercised. The three ship-gate facts (all fixtures captured; gate configs
`0`; `reserveRegisters*` never runs on the control-bit path so the normalized reads
were already empty) all hold.

### Non-captured reject

Placed in `kernel_scheduler::launch` before the running-kernel state mutation
(§10 finding 3). A live negative demo was not run: all four case traces are captured,
and no non-captured fixture exists; crafting one would require synthesizing a trace
with `is_captured_from_binary=false`, out of scope here. The guard is a direct
`!kinfo->is_captured_from_binary → printf + exit(1)`, reachable at the same read
point that previously only counted `num_kernel_not_in_binary`.

## Reverse-include ledger 6 → 4

Run in `simulator-remodeled/gpu-simulator/gpgpu-sim/src`:
`grep -rln '#include.*remodeling/' --include=*.cc --include=*.h . | grep -v '/remodeling/' | sort`

Before 5b (6 files): `abstract_hardware_model.cc`, `gpu-sim.h`, `scoreboard.cc`,
`scoreboard_reads.cc`, `shader.cc`, `shader_core_wrapper.h`.

After 5b (4 files):

```
abstract_hardware_model.cc
gpgpu-sim/gpu-sim.h
gpgpu-sim/shader.cc
gpgpu-sim/shader_core_wrapper.h
```

`scoreboard.cc` and `scoreboard_reads.cc` (which reverse-included
`remodeling/sm.h`+`register_file.h`) reach ZERO by file deletion — the drop from 6
to 4 predicted in the design step-5 row.

## Deviations

- **Two out-of-scope power options retained.** `-scoreboard_war_static_power` and
  `-scoreboard_war_dynamic_power` (members `scoreboard_war_static_power`/`…dynamic_power`
  in `shader_core_config`) are NOT in the design's explicit 3-option scope and have
  **no readers anywhere** (dead already, set by the option parser, never consumed —
  they were only ever meant for the removed AccelWattch power backend). Removing them
  belongs to the deferred vestigial-power cleanup (`2026-07-17-dead-weight.md` item 2),
  not scoreboard retirement, so they and their config lines (`… 0`) are kept. They do
  not break the build after the `Scoreboard` deletion (independent `double` members).
- **Both halves of each SC config deleted.** Each SC config spans two directories:
  `gpgpu-sim/configs/tested-cfgs/SM86_RTXA6000_SC_*` (gpgpusim.config + xml + icnt) and
  `configs/tested-cfgs/SM86_RTXA6000_SC_*` (trace.config). The design/task named the
  `gpgpu-sim/configs` path; the paired top-level `trace.config` directories are pure
  orphans after the gpgpusim halves and yml aliases are removed (repo-wide grep for
  `SM86_RTXA6000_SC` finds only the design docs and the removed yml aliases), so both
  halves are deleted to avoid dead-weight orphans.

## Independent rerun

An independent read-only agent did a clean rebuild (`make clean` + full rebuild) and
reran the full gate on the committed HEAD `0dd8038`, modifying/staging/committing
nothing. Reported:

- HEAD `0dd803891caef196e2d5ddd1bdb6a8196f0b9608`; worktree clean (only the untracked
  `.codegraph/`).
- Build exit `0` (no `undefined reference`, no `error:`; `bin/release/accel-sim.out`
  produced). No `--gc-sections`, so any dangling ref would have failed the link.
- Unit tests exit `0` — 21 tests, `OK`.
- Regression `check` exit `0` — `{"total": 4, "passed": 4, "failed": 0,
  "observed_not_golden": 0}`; all four cases `passed`.
- Reverse-include ledger = exactly 4 files (`abstract_hardware_model.cc`,
  `gpu-sim.h`, `shader.cc`, `shader_core_wrapper.h`).
- Retired-symbol grep: no `class Scoreboard`, `use_traditional_scoreboarding`,
  `is_remodeling_scoreboarding_enabled`, `scoreboard_war_mode`, or
  `scoreboard_war_reads_mode` anywhere in the source. (The grep's `Scoreboard_reads`
  alternative matched only the trailing `// MOD. Scoreboard_reads` annotation comments
  on the four retained `num_*scoreboard*` stat fields in `shader.h` — the R12
  retention, not a live symbol.)

All gates green independently from a clean object tree; goldens byte-identical
(re-approved config hashes only).
