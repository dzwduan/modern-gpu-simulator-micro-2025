# Legacy shader retirement — sever shader.h from remodeling headers

Branch: `refactor/legacy-shader-retirement`. Governing design:
`docs/plans/2026-07-17-legacy-retirement-design.md` §3, §7 step 2, §10 finding 5.

## Commit

`96ff442` — move `shd_warp_t` constructor/destructor out of `shader.h` into
`shader.cc`; replace the three remodeling includes with forward declarations;
add direct includes to the files that actually use the leaked types.

Build 0, unit tests 0 (21 OK), regression check 0 (4/4) — goldens unchanged, so
the change is behavior-neutral (pure include topology + out-of-line definition).

## Design deviation and why (improvement over §7 step 2)

The design proposed relocating `shd_warp_t` into a new L1 header `shd_warp.cc`,
which finding 5 correctly noted would only relocate the reverse include (a new
`shd_warp.cc -> remodeling` edge). Instead the constructor/destructor were moved
into `shader.cc`, which was **already** on the reverse-include ledger (it
includes `remodeling/sm.h`). Net effect: `shader.h` leaves the ledger with **no
new entry added** — strictly better than the design, which would have kept the
count flat. Physical relocation of the KEEP facilities into dedicated L1 headers
is organizational and is deferred to the stage-4 reorganization, where the
god-file split lives; it is not needed to retire the legacy core.

## Transitive-dependency cascade (not fully mapped by the design)

Removing the three includes from `shader.h` surfaced consumers that had been
getting remodeling types transitively through it. Each was given a direct
include of the header it actually uses (include-what-you-use):

| Leaked type | Defining header | Consumer that relied on transitivity | Fix |
| --- | --- | --- | --- |
| `IBuffer_Remodeled`, `Dependency_State` (ctor/dtor new/delete) | ibuffer_remodeled.h, warp_dependency_state.h | shader.cc (moved bodies) | includes added to shader.cc |
| `scheduler_unit` (fwd-decl used by shd_warp_t before its definition) | (was fwd-declared in ibuffer_remodeled.h) | shader.h itself | explicit `class scheduler_unit;` fwd-decl in shader.h |
| `IBuffer_Entry` | ibuffer_remodeled.h | subcore.h (line 163 signature) | include added to subcore.h |
| `Wait_Barrier_Checking` | warp_dependency_state.h | subcore.h (line 187) | include added to subcore.h |
| `L0_icnt` | l0_icnt.h | shader.cc (create_front_pipeline) | include added to shader.cc |
| `num_bytes_cache_req` (free fn) | l0_icnt.h | subcore.cc (line 966) | include added to subcore.cc |

The `scheduler_unit` member of `shd_warp_t` is legacy (dies with the scheduler
family in step 3); it is only forward-declared here to keep the build green, not
kept as a dependency.

## Ledger

7 -> 6 inbound files. `shader.h` removed:

```
abstract_hardware_model.cc
gpgpu-sim/gpu-sim.h
gpgpu-sim/scoreboard.cc
gpgpu-sim/scoreboard_reads.cc
gpgpu-sim/shader.cc
gpgpu-sim/shader_core_wrapper.h
```

`shader.cc` remains (it includes `remodeling/sm.h` for `SM`/`L0_icnt`); it drops
in step 3 when the legacy `create_front_pipeline`/`shader_core_ctx` body that
uses those types is deleted.
