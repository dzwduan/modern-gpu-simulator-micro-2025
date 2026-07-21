# Legacy shader retirement — step 6 disposition: test absorption deferred

Branch: `refactor/legacy-shader-retirement`. Governing design:
`docs/plans/2026-07-17-legacy-retirement-design.md` §6, §7 step 6, §3.1.

## Decision

Step 6 (absorb the stalled `archive/true-path-scoreboard-cleanup` branch's C++
unit tests) adds **no code**. The archive branch's test material is superseded
by the control-bit-only retirement completed in steps 1–5, and absorbing it
verbatim would reintroduce dead / mirror code that contradicts the deletion
principle. Meaningful primitive unit tests are deferred to the stage-4
reorganization, per roadmap §3.1 ("阶段四拆分时为每个 primitive 补最小单元测试").

## Evidence

The archive branch (`archive/true-path-scoreboard-cleanup`, tag from the
repo-hygiene stage) added two extracted pure-function headers and their tests:

1. `dependency_path.h` — `uses_control_bit_dependency(trace, captured,
   scoreboarding)` and `uses_trace_mode_scoreboard(...)`. These encode the
   dependency-MODE selection. After step 5 that selection no longer exists:
   the issue path is unconditionally control-bit for captured kernels, and
   non-captured kernels are rejected at `kernel_scheduler::launch`. The
   functions also take `is_remodeling_scoreboarding_enabled`, a config option
   **deleted in step 5**. Absorbing them would re-add an abstraction over
   removed logic. Superseded.

2. `pipeline_routing.h` — `resolve_int_predicate_target(is_unified)` and
   `resolve_sp_op_target(fp32_in_int, int_can_issue, is_imad)`. These mirror
   the inline routing in `Subcore::get_fu` / `functional_unit`. Production does
   not call them:
   `grep -rn 'resolve_int_predicate_target\|resolve_sp_op_target' remodeling/`
   → 0 references. They are a parallel reimplementation that can drift from the
   real routing (design §6 flagged exactly this). Making them meaningful
   requires refactoring the timing-critical issue-path routing to call them —
   that is stage-4 god-file-split work (`ldst_unit_sm.cc` / `subcore.cc`
   decomposition), not part of legacy retirement, and must be golden-verified
   when done. Deferred to stage 4.

The archive branch's other changes are all already superseded by steps 1–5:
its `exec_shader_core_ctx` removal (step 1), its `use_traditional_scoreboarding`
cleanup (step 5), and its `-is_loog_enabled` registration (moot — LOOG deleted
in step 3). Nothing remains to absorb.

## Consequence

Stage 3 (legacy shader retirement) is functionally complete at step 5. The
GoogleTest harness and per-primitive tests are wired in stage 4, where the
remodeling components are split into independently-testable units and the
routing logic can be extracted into functions that production actually calls
(so the tests exercise real code, not a mirror).

## Gate at stage-3 end (HEAD after step 5)

- `python3 -m unittest discover -s tests` — exit 0, 21 tests OK.
- `python3 tests/remodeled_trace/run_regression.py check` — exit 0, 4/4 passed,
  observed_not_golden 0.
- Reverse-include ledger: 7 (stage-2 baseline) → 4. Remaining inbound files:
  `abstract_hardware_model.cc`, `gpu-sim.h`, `shader.cc`, `shader_core_wrapper.h`
  — all deferred to stage 4 (warp_inst_t / shd_warp_t ownership + the sanctioned
  `new_stats.h` seam).
