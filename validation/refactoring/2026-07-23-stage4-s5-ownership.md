# Stage 4 S5 — close the last layering violations

Branch: `dev_dzw`. Governing design:
`docs/plans/2026-07-22-stage4-reorganization-design.md` §2, §4 S5.
Three commits, each build-green with byte-identical goldens.

## Result

The reverse-include ledger reaches its target: **4 → 2**, and both survivors are
**allowed downward edges** (L3 orchestration depending on L2), not violations:

```
gpgpu-sim/gpu-sim.h            # L3 -> L2, allowed (Element_stats, coalescingStats)
gpgpu-sim/shader_core_wrapper.h # L3 <-> L2 sanctioned contract seam (Element_stats)
```

`abstract_hardware_model` (L0) and `shader.cc` (L1) no longer reference the
remodeling layer at all — by include *or* by type name. The bidirectional
dependency reported in the original structure review is closed.

## Commit 1 (`233a0f0`) — dead include in the hardware model

`abstract_hardware_model.cc` included `remodeling/register_file.h`. The only
symbol that appeared to come from it, `get_number_of_uses_per_operand`, is
declared in `util/traces_enhanced/src/traced_instruction.h`, which
`abstract_hardware_model.h` already includes. The include carried nothing, so
it is deleted. Ledger 4 → 3.

## Commit 2 (`663da0a`) — issue routing as a unit slot

`warp_inst_t` held `remodel::functional_unit *m_fu_assigned`, making the L0
instruction type name an L2 type (forward-declared pointer: no include, but a
real conceptual coupling).

Re-deriving the unit later is **not** possible: `Subcore::get_fu()` routes
non-IMAD SP ops to the INT pipeline based on `m_int_pipeline->can_issue(pI)`,
which is dynamic issue-time state. The decision must be remembered.

Evidence that a slot suffices: `get_fu()` returns only units owned by the
issuing subcore (all members of `m_all_subcore_ex_pipelines`), and all three
readers (`Subcore::read_rf`, and the CONTROL/ALLOCATE and ISSUE/CONTROL stages)
are subcore methods with `this` in hand.

Implementation: each unit is numbered after the pipeline table is populated
(`functional_unit::set_subcore_slot`), instructions carry a plain
`int m_fu_slot` (-1 until routed), and `Subcore::fu_at_slot()` resolves it with
a bounds assert. Numbering by a loop over the final table keeps it correct if
the construction order ever changes. `abstract_hardware_model.h` now contains
zero `remodel::` references.

## Commit 3 — per-warp remodeling state owned by the SM

`shd_warp_t`'s constructor/destructor allocated and freed
`remodel::IBuffer_Remodeled` and `remodel::Dependency_State`, which forced
`shader.cc` (L1) to include both remodeling headers.

Ownership moves to the SM, which already creates the warps: `SM::create_shd_warp`
constructs the warp and injects the state via `shd_warp_t::set_remodel_state`,
and `SM::~SM` frees the state before the warp. `shd_warp_t` now only observes
the pointers (null until injected), its destructor is empty, and `shader.cc`
drops both includes. Ledger 3 → 2.

Preconditions verified before the move:
- `shd_warp_t::reset()` (called from the constructor) does not touch either
  pointer, so post-construction injection is safe.
- `create_shd_warp()` has exactly one caller (`SM::init`), so no repeated
  allocation.
- `delete warp` appears in exactly one place (`SM::~SM`), so there is one
  balanced deletion site.

Ownership balance is by construction: one creation loop and one deletion loop
over the same `m_physical_warp` bounds, and the warp destructor no longer frees
anything, so neither a leak nor a double free is introduced. The four golden
runs complete cleanly.

## Gate (each commit)

- Build: exit 0.
- `python3 -m unittest discover -s tests`: exit 0, 21 tests OK.
- `python3 tests/remodeled_trace/run_regression.py check`: exit 0, 4/4 passed,
  observed_not_golden 0 — goldens untouched throughout (no config changed, so
  no re-approval was needed).

## Follow-up noted, not fixed here

`abstract_hardware_model.cc` reports a pre-existing `-Wuninitialized` warning
for `index` (upstream code, unrelated to these commits).
