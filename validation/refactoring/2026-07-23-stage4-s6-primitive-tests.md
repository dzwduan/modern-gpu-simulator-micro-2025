# Stage 4 S6 — primitive extraction and GoogleTest unit tests

Branch: `dev_dzw`. Governing design:
`docs/plans/2026-07-22-stage4-reorganization-design.md` §3.1, §4 S6.
Prior disposition this closes: `validation/refactoring/2026-07-17-legacy-retire-step6.md`.

Two code commits, both build-green with byte-identical goldens, plus this
record.

| commit | subject |
| --- | --- |
| `a4c8f05` | `refactor: extract the register encoding helpers into their own unit` |
| `957d323` | `test: cover the register encoding and access queue primitives` |

## Governing principle

The archived branch's tests were rejected because they exercised a *mirror* of
the production logic (`pipeline_routing.h`, zero production callers). Every case
here links the object files the simulator build emits. Nothing under `tests/cpp`
recompiles or restates production logic; the evidence is recorded below.

## Commit 1 — `remodeling/register_encoding.{h,cc}`

### What moved

Out of `remodeling/sm.h` (declarations, constants) and `remodeling/sm.cc`
(definitions), into the new unit, **verbatim** — no logic edit, no rename, no
signature change:

| symbol | kind |
| --- | --- |
| `translate_warp_id_of_sm_to_subcore` | function |
| `get_reg_type_eval` | function |
| `check_is_reserved_regs_remodeling` | function |
| `translate_reg_to_global_id` | function |
| `RESERVED_REG_NUMBER` / `RESERVED_UREG_NUMBER` / `RESERVED_PRED_NUMBER` / `RESERVED_UPRED_NUMBER` | constant |
| `GLOBAL_ID_BASE_UREG` / `GLOBAL_ID_BASE_PRED` / `GLOBAL_ID_BASE_UPRED` | constant |

`sm.h` gains `#include "register_encoding.h"`, so every existing caller
(`subcore.cc`, `register_file.cc`) resolves unchanged: same `namespace remodel`,
same names, no call-site edit anywhere in the tree. `sm.cc` keeps its own
`traced_operand.h` include because it still names `TraceEnhancedOperandType`
directly.

### Rationale

These four are pure encoding helpers between the trace operand representation
and the simulator's flat register/warp numbering. Their entire dependency set is
`TraceEnhancedOperandType` / `traced_operand` (util layer) plus the numbering
constants — no SM, no `shader_core_config`, no simulator state. Hosting them in
the ~1900-line SM translation unit was incidental to how the SM grew, not a
design choice. Splitting them out both raises cohesion and makes them linkable
by a small binary: `register_encoding.o` pulls in only `traced_operand.o` and
`string_utilities.o`, whereas `sm.o` pulls in essentially the whole simulator.
This is the §3.1 "independently instantiable and verifiable" requirement applied
to the smallest unit that had it available.

The new header includes `util/traces_enhanced/src/traced_operand.h` directly
rather than relying on the transitive path through
`abstract_hardware_model.h`, so it compiles standalone. That include is a
downward (L2 → util) edge and adds nothing new to any existing translation
unit, which already reached the same header transitively.

## Commit 2 — `tests/cpp`

### Layout

```
tests/cpp/Makefile                    out-of-tree GoogleTest build
tests/cpp/test_register_encoding.cc   21 cases
tests/cpp/test_access_queue.cc         9 cases (3 of them death tests)
tests/cpp/.gitignore                  build/
```

### Build and run

```bash
make -C tests/cpp test
```

Requires the simulator to have been built (the Makefile links its objects and
never rebuilds them) and GoogleTest present (`pkg-config --exists gtest`;
`/usr/include/gtest/gtest.h`, `/usr/lib/x86_64-linux-gnu/libgtest{,_main}.a`).
`make -C tests/cpp clean` removes only `tests/cpp/build`.

Observed output tail:

```
[==========] 30 tests from 6 test suites ran. (212 ms total)
[  PASSED  ] 30 tests.
```

Exit code 0.

Isolation from the rest of the harness:

- Nothing in the simulator Makefiles references `tests/cpp`; a `make clean` plus
  full rebuild of the simulator was run after the tests were added and is green
  (see the gate table).
- `python3 -m unittest discover -s tests` still reports exactly 21 tests, so the
  C++ cases are not in the Python discovery path.
- Without the simulator objects present the Makefile fails fast with
  `error: no simulator object directory under .../gpgpu-sim/build; build the
  simulator first` rather than silently compiling a substitute. Verified by
  running the target immediately after `make clean` of the simulator
  (exit code 2).

### Evidence that the tests link production code, not a copy

1. The test objects *declare but do not define* every symbol under test — they
   are undefined references resolved at link time:

   ```
   $ nm -C tests/cpp/build/test_register_encoding.o | grep 'remodel::'
                    U remodel::get_reg_type_eval(traced_operand&)
                    U remodel::translate_reg_to_global_id(int, TraceEnhancedOperandType)
                    U remodel::check_is_reserved_regs_remodeling(int, TraceEnhancedOperandType, bool)
                    U remodel::translate_warp_id_of_sm_to_subcore(unsigned int, unsigned int)
   $ nm -C tests/cpp/build/test_access_queue.o | grep 'remodel::'
                    U remodel::AccessQueue::{ctor,dtor,push,pop,front,empty,full,size}
   ```

   (The same `nm` output also lists `r remodel::RESERVED_*` /
   `r remodel::GLOBAL_ID_BASE_*`. Namespace-scope `constexpr` has internal
   linkage, so an unoptimised build materialises a private copy in every
   translation unit that includes the header; the assertions do not read them.
   Expected values are written as literals — 255, 512, 520 and so on — so a
   change to one of the constants fails a test instead of being mirrored by it.
   Every function and every `AccessQueue` member is `U`.)

2. The definitions come from the objects the simulator build produced:

   ```
   $ nm -C simulator-remodeled/gpu-simulator/gpgpu-sim/build/gcc-/cuda-12060/release/remodeling/register_encoding.o | grep ' T .*remodel::'
   0000000000000000 T remodel::translate_warp_id_of_sm_to_subcore(unsigned int, unsigned int)
   0000000000000010 T remodel::get_reg_type_eval(traced_operand&)
   0000000000000120 T remodel::check_is_reserved_regs_remodeling(int, TraceEnhancedOperandType, bool)
   0000000000000170 T remodel::translate_reg_to_global_id(int, TraceEnhancedOperandType)
   ```

   The same command on `access_queue.o` lists all eight `AccessQueue` members.

3. Removing the two production objects from the link line makes the link fail
   with 93 undefined references, all of them `remodel::` symbols — there is no
   fallback definition anywhere in the test sources.

The link set is exactly `register_encoding.o`, `access_queue.o` (production
remodeling objects) plus `traced_operand.o` and `string_utilities.o` (the util
objects `get_reg_type_eval` calls into), and GoogleTest. No production `.cc` is
compiled by `tests/cpp/Makefile`.

### Test inventory

`translate_warp_id_of_sm_to_subcore` — warps are dealt to subcores by
`warp_id % num_subcores`; the helper returns the slot inside that subcore's
slice.

| test | contract pinned |
| --- | --- |
| `TranslateWarpIdOfSmToSubcore.IndexesTheWarpWithinItsSubcoreSlice` | 4 subcores: warps 0–3 → slot 0, 4–7 → slot 1, 47 → slot 11 |
| `TranslateWarpIdOfSmToSubcore.IsIdentityForASingleSubcore` | `num_subcores == 1` returns the warp id unchanged |
| `TranslateWarpIdOfSmToSubcore.TruncatesWhenTheWarpCountIsNotDivisible` | 6 warps over 4 subcores → `{0,0,0,0,1,1}`; 3 subcores over warps 7/8/9 → 2/2/3 |

`check_is_reserved_regs_remodeling`

| test | contract pinned |
| --- | --- |
| `CheckIsReservedRegsRemodeling.DetectsTheDiscardRegisterOfEachFile` | RZ 255 / URZ 63 / PT 7 / UPT 7 report reserved for REG / UREG / PRED / UPRED |
| `CheckIsReservedRegsRemodeling.AcceptsOrdinaryRegistersOfEachFile` | the neighbouring numbers (0, 254, 62, 6) do not |
| `CheckIsReservedRegsRemodeling.ReservedNumbersAreScopedToTheirFile` | a reserved number of another file (63 as REG, 255 as UREG, …) is not reserved |
| `CheckIsReservedRegsRemodeling.OperandTypesWithoutARegisterFileAreNever` | all 11 non-register operand types × {255, 63, 7} → false |
| `CheckIsReservedRegsRemodeling.OutsideTraceModeNothingIsReserved` | `is_trace_mode == false` → false for all four files and for every REG number 0–255 |

`translate_reg_to_global_id`

| test | contract pinned |
| --- | --- |
| `TranslateRegToGlobalId.RegularRegistersKeepTheirNumber` | REG *n* → *n*; RZ → 255 |
| `TranslateRegToGlobalId.UniformRegistersStartAtTheUniformBase` | UREG *n* → 256 + *n*; URZ → 319 |
| `TranslateRegToGlobalId.PredicatesStartAtThePredicateBase` | PRED *n* → 512 + *n*; PT → 519 |
| `TranslateRegToGlobalId.UniformPredicatesStartAtTheUniformPredicateBase` | UPRED *n* → 520 + *n*; UPT → 527 |
| `TranslateRegToGlobalId.WholePredicateRegisterCollidesWithPredicateZero` | PR → 512 and UPR → 520 (see surprises) |
| `TranslateRegToGlobalId.OperandTypesWithoutARegisterFileMapToZero` | all 11 non-register operand types → 0 (see surprises) |
| `TranslateRegToGlobalId.TheFourFileRangesDoNotOverlap` | the top id of each file is below the base of the next |

`get_reg_type_eval` (inputs are real `traced_operand`s built from SASS operand
strings, so the trace parser is in the loop)

| test | contract pinned |
| --- | --- |
| `GetRegTypeEval.PassesPlainRegisterOperandsThrough` | `R5`/`RZ`→REG, `UR4`/`URZ`→UREG, `P0`/`PT`→PRED, `UP1`→UPRED |
| `GetRegTypeEval.PassesOperandsWithoutARegisterFileThrough` | `0x1234`→IMM_UINT64, `SR_TID.X`→SR, `B0`→BREG |
| `GetRegTypeEval.ResolvesMemoryReferencesToTheirAddressRegisterFile` | `[R2]`, `[R2+0x10]`→REG; `[UR4]`→UREG |
| `GetRegTypeEval.ResolvesConstantBankOperandsToTheirIndexRegisterFile` | `c[0x0][R4]`→REG, `c[0x0][UR4]`→UREG |
| `GetRegTypeEval.LeavesRegisterFreeConstantBankOperandsUnchanged` | `c[0x0][0x160]` stays CBANK |
| `GetRegTypeEval.UniformFileWinsWhenAnOperandNamesBothFiles` | `[R2+UR4]` and `desc[UR4][R2.64]` → UREG (see surprises) |

`AccessQueue`

| test | contract pinned |
| --- | --- |
| `AccessQueue.StartsEmpty` | fresh queue: `empty()` true, `full()` false, `size()` 0 |
| `AccessQueue.ServesAccessesInPushOrder` | FIFO: `front()` returns the pushes in order across `pop()`s |
| `AccessQueue.FrontDoesNotConsume` | repeated `front()` leaves `size()` unchanged |
| `AccessQueue.SizeAndEmptyTrackPushAndPop` | `size()` follows every push and pop; `empty()` flips only at 0 |
| `AccessQueue.BecomesFullAtItsCapacityAndFreesUpOnPop` | `full()` at exactly `max_size`, clears on `pop()`, re-arms on the next push |
| `AccessQueue.ZeroCapacityIsBothEmptyAndFull` | capacity 0 → `empty()` and `full()` both true (see surprises) |
| `AccessQueueDeathTest.PushingPastCapacityAborts` | the capacity boundary is an `assert`, live in the release objects |
| `AccessQueueDeathTest.PoppingAnEmptyQueueAborts` | `pop()` asserts on an empty queue |
| `AccessQueueDeathTest.ReadingTheFrontOfAnEmptyQueueAborts` | `front()` asserts on an empty queue |

The `AccessQueue` cases use distinct non-null sentinel pointers, which the class
never dereferences. Its destructor `delete`s whatever is still queued — it owns
what it holds — so every case drains the queue before it leaves scope. That
ownership path is therefore *not* covered: exercising it needs real
`mem_access_t` instances, which would drag the whole simulator into the link.
Noted as an uncovered edge, not worked around.

## Behaviour pinned as observed rather than corrected

None of these were changed. S6 is golden-neutral by construction; each is
recorded for a later owner to decide on.

1. **`PR` / `UPR` collide with predicate 0.** `PR` (8) is the trace encoding for
   *the whole predicate register file*, not for one predicate.
   `translate_reg_to_global_id` drops the offset for it, so `PR` → 512, the same
   id as `P0`; likewise `UPR` → 520 = `UP0`. Any dependence tracking keyed on
   this id cannot distinguish "the whole file" from "predicate 0". Pinned in
   `WholePredicateRegisterCollidesWithPredicateZero`.

2. **Operand types with no register file map to global id 0.**
   `translate_reg_to_global_id` initialises `global_id = 0` and falls through for
   CBANK / MREF / IMM / SR / SB / BREG / DESC / GENERIC / CALL_TARGET / NONE, so
   they alias regular register `R0`. Callers must gate on the type before
   trusting the id. Pinned in `OperandTypesWithoutARegisterFileMapToZero`.

3. **`get_reg_type_eval` prefers the uniform file when an operand names both.**
   It tests `find("UR")` before `find("R")`, so `desc[UR4][R2.64]` — the ordinary
   descriptor form of a global access whose *address* register is `R2` — reports
   UREG, and so does `[R2+UR4]`. Pinned in
   `UniformFileWinsWhenAnOperandNamesBothFiles`.

4. **A zero-capacity `AccessQueue` is simultaneously empty and full.** `full()`
   is `size() == m_max_size` with no lower bound, so `empty()` and `full()` are
   both true at capacity 0, and any `push` immediately trips the assert. Pinned
   in `ZeroCapacityIsBothEmptyAndFull`.

5. **Asserts are live in the release build.** The remodeling Makefile never
   defines `NDEBUG`, so the `AccessQueue` preconditions abort in the shipped
   objects rather than silently overflowing. The three death tests depend on
   this; if `NDEBUG` is ever added to the release flags they will start failing,
   which is the intended signal.

6. **Two of the four extracted functions have no production caller.**
   `check_is_reserved_regs_remodeling` and `translate_reg_to_global_id` are
   referenced nowhere outside their own declaration and definition
   (`grep -rIn 'check_is_reserved_regs_remodeling\|translate_reg_to_global_id'`
   over the tree returns only `register_encoding.{h,cc}` and documentation).
   `get_reg_type_eval` (`register_file.cc`, `subcore.cc`) and
   `translate_warp_id_of_sm_to_subcore` (`subcore.cc`, five sites) are live.
   The tests still exercise the real definitions rather than a copy, so they are
   not a mirror — but they pin an API that no simulation path reaches today.
   `util/traces_enhanced/src/traced_instruction.h` already offers
   `is_reserved_reg(reg_id, reg_type)`, which `register_file.cc` and `subcore.cc`
   call instead, and which duplicates the intent of
   `check_is_reserved_regs_remodeling`. Deciding between the two is a deletion /
   consolidation question, out of scope for a golden-neutral step; flagged for
   the coordinator.

## Gate

Run from the repository root. Build:
`cd simulator-remodeled && source ./gpu-simulator/setup_environment_no_git.sh && make -j$(nproc) -C ./gpu-simulator/`.

| step | build | `unittest discover -s tests` | `run_regression.py check` | `make -C tests/cpp test` |
| --- | --- | --- | --- | --- |
| `a4c8f05` extraction | exit 0 | exit 0, 21 tests OK | exit 0, 4/4, `observed_not_golden` 0 | n/a |
| `957d323` tests (after `make clean` + full rebuild) | exit 0 | exit 0, 21 tests OK | exit 0, 4/4, `observed_not_golden` 0 | exit 0, 30/30 passed |

The final row is a from-scratch build: the simulator tree was cleaned
(`make clean -C ./gpu-simulator/`, exit 0) and rebuilt in full before the gate,
which also re-derives the `makedepend` fragment that now covers
`register_encoding.cc`. No compiler warning is emitted for either new file.

All four commands were re-run once more with `957d323` as the tree state and
all four exit 0 with the same results: build 0,
21 Python tests OK, 30 C++ tests passed, regression 4/4 with
`observed_not_golden` 0. An independent-agent re-run is left to the coordinator's
review pass; the commands above are the whole reproduction.

## Goldens

Byte-identical throughout. `git status --short tests/` reports no modification
to any tracked file under `tests/` across both commits (the only entry is the
new untracked `tests/cpp/`), and both `check` runs report
`"observed_not_golden": 0` with `"passed": 4` of `"total": 4`. Neither commit
touches simulated behaviour: commit 1 moves code without editing it, commit 2
adds no production code at all.

## Reverse-include ledger

Unchanged at **2**, the S5 target, both allowed downward edges:

```
$ grep -rl --include=*.h --include=*.cc \
    '#include *"[^"]*remodeling/' \
    simulator-remodeled/gpu-simulator/gpgpu-sim/src | grep -v '/remodeling/'
simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.h
simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader_core_wrapper.h
```

(`gpu-simulator/trace-driven/trace_driven.cc` also includes `remodeling/sm.h`
and `remodeling/ldst_unit_sm.h`, unchanged by this step; it sits above the
remodeling layer, so those are downward edges and outside the ledger's scope as
defined in design §2.)

Neither commit adds a cross-layer include: `register_encoding.h` lives inside
`remodeling/` and depends only on the util layer below it, and `tests/cpp` is
not part of the simulator build graph at all.

## Review response

The independent review found the production extraction behavior-preserving and
raised two P2 defects in the new test build, both confirmed and fixed:

1. **Wrong object tree.** The Makefile globbed
   `build/*/*/release/remodeling` and took `firstword`, so a debug build was
   never found and, with several compiler/CUDA trees present, an arbitrary
   (possibly stale) one was linked. Fixed: the object directory is now derived
   from `GPGPUSIM_CONFIG`, which `setup_environment_no_git.sh` exports and which
   names the build tree exactly. Without that variable the build discovers the
   candidate trees and refuses to guess when more than one exists, with an error
   pointing at the setup script.
2. **No header dependency tracking.** The test objects depended only on their
   `.cc`, so a change to `register_encoding.h`, `access_queue.h` or any
   transitive header reused a stale object — the tests could then report on
   declarations that no longer exist. Fixed with `-MMD -MP` and `-include` of
   the generated `.d` files.

Verification of both: `make -C tests/cpp test` passes 30/30 with `GPGPUSIM_CONFIG`
set and, after `make clean`, also via the single-candidate discovery path;
`build/*.d` files are generated; and `touch`ing `register_encoding.h` now
recompiles both test objects (it previously would not).
