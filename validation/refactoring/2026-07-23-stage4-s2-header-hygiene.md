# Stage 4 S2 — header hygiene in the remodeling headers

Branch: `dev_dzw`. Governing design:
`docs/plans/2026-07-22-stage4-reorganization-design.md` §4 S2. Pure
structure-preserving cleanup; goldens byte-identical.

## Changes

1. **Dead debug-decode array deleted** (`sm.h`):
   `const char *const subcore_dispatch_latch_decode[]` had ZERO users repo-wide
   (grep over all `.cc`), so per the deletion principle it is removed outright
   rather than moved out of the header. It also stops duplicating a copy of the
   array into every translation unit that includes `sm.h`.
2. **`#define STRSIZE 1024` triplication unified** (`sm.cc`, `subcore.cc`,
   `ldst_unit_sm.cc` — the `subcore.cc` one sat mid-file): replaced by one
   `constexpr unsigned STRSIZE = 1024;` in `namespace remodel` (`sm.h`), which
   all three sources include. Usage sites are `char buf[STRSIZE]` — value and
   semantics identical.
3. **Latency macros converted to `constexpr`** (`sm.h`): the five
   `NO_TENSOR_OP_…` / `MULTIPLIER_…` / `MAXIMUM_…` / `NUM_INTERMEDIATE_…`
   defines moved inside `namespace remodel` as `constexpr unsigned`. Safety
   check performed before conversion: every usage site (function arguments,
   `resize()`, comparisons, ternary operands in `subcore.cc`, `sm.cc`,
   `register_file.cc`, `ldst_unit_sm.cc`) was verified to have no adjacent
   operator that textual macro expansion would bind differently, so the values
   (3, 2, 6, 5, 8) are unchanged.
4. **`new_stats.h` dead surface removed**: the never-populated, never-read
   `std::map<std::string, std::type_index> m_stats_type_map` member and the
   three includes it justified (`<typeindex>`, `<type_traits>`, `<variant>` —
   the latter two had zero uses anywhere in the header).

Deliberately NOT moved: the small `Waiting_Dep_Counters_per_Warp` /
`InterWarp_Coalescing_Waiting_Dep_Counters` helper classes keep their in-header
bodies — they are cohesive value-type helpers a few lines long; relocating them
buys no compile-time or clarity benefit. The originally-flagged fully-inline
`functional_unit_with_queue_stage` no longer exists (removed with the
retirement).

## Gate

- Build (incremental after edit): exit 0, zero errors.
- `python3 -m unittest discover -s tests`: exit 0, 21 tests OK.
- `python3 tests/remodeled_trace/run_regression.py check`: exit 0, 4/4 passed,
  observed_not_golden 0 — goldens untouched, byte-identical.
- Reverse-include ledger: unchanged at 4 (no include topology change).
