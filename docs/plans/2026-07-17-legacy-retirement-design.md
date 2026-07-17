# 阶段三动刀前置：Legacy Shader 时序路径退役 — 目标架构细化设计

Status: DRAFT for independent review. READ-ONLY investigation, no repo file modified.
Branch: `dev_dzw`. Repo root: `/home/duanzhenwei/modern-gpu-simulator-micro-2025`.
Contract inputs read: `docs/plans/2026-07-17-structure-refactor-roadmap.md` (§2, §3.1, 阶段三),
`docs/plans/2026-07-15-remodeled-trace-p0-p1-semantic-repair.md`,
`validation/refactoring/2026-07-17-safety-net.md` (Dependency-direction baseline),
`validation/refactoring/2026-07-17-dead-weight.md`.

## 0. 约定与证据口径

Path shorthand used below:

- `SRC` = `/home/duanzhenwei/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/src`
- `CORE` = `SRC/gpgpu-sim` (contains `shader.{h,cc}`, `shader_core_wrapper.h`, `scoreboard*.{cc,h}`, `gpu-sim.*`)
- `REM` = `SRC/gpgpu-sim/remodeling` (the L2 SM model)
- `TRACE` = `/home/duanzhenwei/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/trace-driven`
- `MAIN` = `/home/duanzhenwei/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/main.cc`

Every claim is anchored by symbol name (class/method/member) plus, where load-bearing, the grep
command that produced the count. Line numbers, when cited, are paired with a symbol and are
supplementary only. Gate = `python3 -m unittest discover -s tests` (21 tests) +
`OMP_NUM_THREADS=1 python3 tests/remodeled_trace/run_regression.py check` (4/4), per the two
validation records.

File sizes (`wc -l`): `shader.h` 3716, `shader.cc` 4870, `shader_core_wrapper.h` 163,
`scoreboard.cc` 275, `scoreboard_reads.cc` 252, `abstract_hardware_model.h` 2172,
`abstract_hardware_model.cc` 1471.

## 0.1 架构地基（先讲清楚，否则整份清单会读错）

退役设计的全部结论都依赖一个此前文档未点破的事实：**`SM` 与 legacy `shader_core_ctx` 是两个并列
的 core 实现，二者都实现同一个 `shader_core_ctx_wrapper` 接口；trace 运行时只实例化 `SM`。**

证据（继承关系）：

- `class SM : public core_t, public shader_core_ctx_wrapper` — `REM/sm.h` (`class SM`).
- `class shader_core_ctx : public core_t, public shader_core_ctx_wrapper` — `CORE/shader.h`.
- `class trace_shader_core_ctx : public shader_core_ctx` — `TRACE/trace_driven.h`.
- `class exec_shader_core_ctx : public shader_core_ctx` — `CORE/shader.h`.

证据（工厂分支，`m_core[]` 由谁填充）：`trace_simt_core_cluster::create_shader_core_ctx()` in
`TRACE/trace_driven.cc`：

```
if (m_config->is_SM_remodeling_enabled) {
    m_core[i] = new SM(m_config->num_subcores_in_SM, ...);   // supported path
    m_core[i]->init();
} else {
    m_core[i] = new trace_shader_core_ctx(...);              // legacy trace core -> DELETE
}
```

`exec_simt_core_cluster::create_shader_core_ctx()` in `CORE/shader.cc` 有同构分支
（`new SM` 或 `new exec_shader_core_ctx`）。因此：

- 支持路径的对象图是 `trace_gpgpu_sim -> trace_simt_core_cluster -> SM`（`MAIN` 只造 `trace_gpgpu_sim`，见 §4）。
- `shader_core_ctx`（连同其派生 `trace_shader_core_ctx`、`exec_shader_core_ctx`）在支持路径上**从不被实例化**——只要 `is_SM_remodeling_enabled` 为真，工厂只造 `SM`。
- `SM` 自己重建了整条时序流水线：`SM::create_shd_warp` (`REM/sm.cc`), `SM::func_exec_inst`,
  `SM::checkExecutionStatusAndUpdate`, `Subcore::get_next_inst` (`REM/subcore.cc`) 等，均与
  `shader_core_ctx` 同名方法一一对应（复制粘贴来源，见 §4.3）。

推论：退役 legacy 时序路径 = **删除整个 `shader_core_ctx` 类族**（base + `trace_shader_core_ctx`
\+ `exec_shader_core_ctx`）及其独占的流水线机器，让 `SM` 成为 `m_core[]` 的唯一居民。这比“删掉
一些方法”激进得多，也更干净。

### 一个新发现的反向依赖（未在现有 ledger 中）

`REM/sm.cc` 直接 `#include "../../../../trace-driven/trace_driven.h"`，并在 `SM::create_shd_warp`
里 `new trace_shd_warp_t(...)`，在多处 `static_cast<trace_shd_warp_t *>(m_physical_warp[...])`
（`SM::func_exec_inst`, `SM::init_warps` 等）。这是 **L2 → L4 的反向依赖**（remodeling 反向依赖
trace-driven），现有 dependency-direction ledger 只统计“外部反向 include remodeling/”，没有捕获
这一条。它直接决定 `shd_warp_t` / `trace_shd_warp_t` 的归属裁定（见 §3），必须在本阶段处理或显式
记账。

Grep evidence:
`grep -n '#include' REM/sm.cc` → line 34 `../../../../trace-driven/trace_driven.h`;
`grep -n 'trace_shd_warp_t' REM/sm.cc` → `new trace_shd_warp_t` in `SM::create_shd_warp`, three
`static_cast<trace_shd_warp_t *>` sites.

---

## 1. shader.{h,cc} 解剖清单（DELETE / KEEP / TRANSFORM）

`shader.h` top-level 类/结构/枚举（`grep -nE '^(class|struct|enum)' shader.h`）共 30 个符号定义 +
若干前置声明。`shader.cc` 方法定义按类分布（`grep -oE '<Class>::<method>' shader.cc | 归类`）：
`shader_core_ctx` 52, `ldst_unit` 30, `simt_core_cluster` 25, `pipelined_simd_unit` 19,
`opndcoll_rfu_t` 19, `shader_core_stats` 12, `scheduler_unit` 11, `exec_shader_core_ctx` 7,
`barrier_set_t` 7, `shd_warp_t` 6, `simd_function_unit` 4, `tensor_core`/`sfu` 3 each,
`two_level_active_scheduler`/`swl_scheduler`/`sp_unit`/`specialized_unit`/`shader_core_mem_fetch_allocator`/`shader_core_config`/`int_unit`/`dp_unit` 2 each,
`rrr_scheduler`/`oldest_scheduler`/`lrr_scheduler`/`gto_scheduler`/`exec_simt_core_cluster` 1 each.

### 1a. DELETE — legacy pipeline machinery, no remodeled-path use

Reference test for each: `grep -rn '<symbol>' SRC TRACE --include=*.cc --include=*.h` filtered to
exclude `shader.{cc,h}` themselves. "0 outside" means the only referrers are legacy code inside
`shader.{cc,h}`.

| Symbol (shader.h anchor) | Kind | Proof it is legacy-only |
| --- | --- | --- |
| `scheduler_unit` + `lrr_scheduler`,`rrr_scheduler`,`gto_scheduler`,`oldest_scheduler`,`two_level_active_scheduler`,`swl_scheduler` | scheduler family | `lrr/gto/rrr/oldest/two_level/swl` = 0 refs outside `shader.{cc,h}`. `scheduler_unit` outside = 2 refs, both inert: a comment in `CORE/shader_trace.h` ("Intended to be called from inside a scheduler_unit") and an **unused forward decl** `class scheduler_unit;` in `REM/ibuffer_remodeled.h` ("Definition to be allowed to compile"). No remodeling `.cc` calls it. |
| `enum scheduler_prioritization_type`, `enum concrete_scheduler` | scheduler config enums | Consumed only by the scheduler family + legacy `create_schedulers`. |
| `opndcoll_rfu_t` (operand collector, incl. nested `op_t`,`allocation_t`,`arbiter_t`,`input_port_t`,`collector_unit_t`,`dispatch_unit_t`) | register-file/operand collector | 4 refs outside `shader.{cc,h}`, all in `CORE/result_bus.{h,cc}` (`RRS::init(unsigned,unsigned,opndcoll_rfu_t*)` + `m_rf` member). That coupling is itself legacy (see TRANSFORM `result_bus`). Remodeling uses its own `Register_file`/`Register_file_cache` (`REM/register_file.h`), never `opndcoll_rfu_t`. |
| `simd_function_unit`, `pipelined_simd_unit`, `sfu`, `dp_unit`, `tensor_core`, `int_unit`, `sp_unit`, `specialized_unit` | legacy EX pipeline units | `simd_function_unit`,`pipelined_simd_unit`,`class sfu`,`class dp_unit` = 0 refs outside `shader.{cc,h}`. `sp_unit`/`int_unit`/`tensor_core`/`specialized_unit` outside-hits are all substrings of config fields (`m_config->...`, `gpgpu_num_sp_units`, `specialized_unit_params`, `tensor_core_avail`, `OP_*`) or option strings in `cuda-sim.cc`/`trace_driven.cc` — no use of the **classes**. Remodeling EX pipeline is `REM/functional_unit.{h,cc}`. |
| `ldst_unit` (legacy, `class ldst_unit : public pipelined_simd_unit`) | legacy LD/ST | 0 refs outside `shader.{cc,h}` after excluding `ldst_unit_sm` and the forward decl `ldst_unit_remake`. Remodeling LD/ST is `REM/ldst_unit_sm.{h,cc}`. |
| `enum pipeline_stage_name_t` | legacy pipeline reg names | 0 refs in `REM` (`grep -n 'pipeline_stage_name_t\|ID_OC_SP\|N_PIPELINE_STAGES' REM` empty). |
| `struct insn_latency_info` | legacy latency probe | 0 refs anywhere except its definition. Dead. |
| `shader_core_ctx` (52 methods) + `exec_shader_core_ctx` (7) | legacy timing core | Instantiated only via the legacy `else`-branch of the two `create_shader_core_ctx` factories; never when `is_SM_remodeling_enabled` (§0.1). SM re-implements every timing method. |
| `exec_simt_core_cluster` | legacy cluster subclass | Constructed only by `exec_gpgpu_sim::createSIMTCluster` (`gpu-sim.cc`), i.e. the PTX/CUDA entrypoint; unreachable from `MAIN` (§4). |
| free fn `register_bank(...)` (`shader.cc`) | operand-collector bank map | 0 refs in `REM`; used only by `opndcoll_rfu_t`. |
| free fn `coalesced_segment(...)` (`shader.cc`) | legacy coalescer helper | 0 refs in `REM`/`TRACE`. |
| free fn `check_kernel_launch_limitation(...)` (`shader.h`/`shader.cc`) | launch guard | Sole caller is legacy `shader_core_ctx::issue_block2core` (`shader.cc`); no `SM` caller. If the guard is still wanted, re-host inside `SM::issue_block2core` (small, see Risk R7); otherwise DELETE. |

Legacy timing method set on `shader_core_ctx` (the 52) that dies wholesale (grep-derived):
`create_front_pipeline`,`create_exec_pipeline`,`create_schedulers`,`create_shd_warp`,`fetch`,`decode`,
`issue`,`issue_warp`,`read_operands`,`execute`,`writeback`,`next_pc`,`print_stage`,`test_res_bus`,
`cycle`,`func_exec_inst`,`checkExecutionStatusAndUpdate`,`get_next_inst`,`decrement_trace_pc`,
`get_active_mask`,`get_pdom_stack_top_info`,`init_warps`,`translate_local_memaddr`,`warp_exit`,
`warp_inst_complete`,`register_cta_thread_exit`,`set_max_cta`,`reinit`,`store_ack`,
`accept_fetch_response`,`accept_ldst_unit_response`,`broadcast_barrier_reduction`,`decrement_atomic_count`,
`is_subcore_active`,`incexecstat`,`customStatsWarpActiveLanes`,`get_Scoreboard_reads`,
`check_if_non_released_reduction_barrier`,`get_current_gpu_cycle`,`get_current_occupancy`,`get_regs_written`,
`get_cache_stats`,`get_L{0I,1I,1C,1D,1T}_sub_stats`,`get_icnt_power_stats`,`print_cache_stats`,
`display_pipeline`,`display_simt_state`,`fetch_unit_response_buffer_full`,`ldst_unit_response_buffer_full`.
All have a same-named `SM` override (§0.1) — deleting the base leaves `SM`'s copy as sole implementation.

### 1b. KEEP — shared facilities the SM path uses (with target L-layer + target home file)

Per roadmap §3.1 L0–L4. Target home files are proposals for the reorganization; the stage-3
requirement is only that these classes survive and stop living inside the legacy `shader_core_ctx`
translation unit.

| Symbol | Proof of remodeled-path use (grep hit) | Target layer | Proposed home |
| --- | --- | --- | --- |
| `shd_warp_t` (+ inner `ibuffer_entry`) | `SM::m_physical_warp` is `std::vector<shd_warp_t *>` (`REM/sm.h`); `SM::create_shd_warp` fills it (`REM/sm.cc`); pervasive `m_physical_warp[warp_id]->...` in `REM/sm.cc`,`REM/subcore.cc` | L1 | `CORE/shd_warp.h` (see §3) |
| `barrier_set_t` | `SM::m_barriers` is a `barrier_set_t` member (`REM/sm.h`), constructed `m_barriers(this, ...)` (`REM/sm.cc`), called `m_barriers.warp_reaches_barrier/allocate_barrier/warp_exit/warp_waiting_at_barrier/deallocate_barrier` (`REM/sm.cc`) | L1 | `CORE/barrier_set.h` |
| `shader_core_config` (+ `struct specialized_unit_params` it embeds) | Holds `is_SM_remodeling_enabled` (`shader.h`); read everywhere in `REM` via `m_config->...`; SM ctor takes `const shader_core_config*` | L1 (shared config; extends L0 `core_config`) | `CORE/shader_core_config.h` |
| `shader_core_stats` (+ POD base `shader_core_stats_pod`) | `SM::m_stats`; `SM::m_stats->shader_cycles[...]`, `shader_cycles_per_kernel`, `shader_active_warps_per_kernel` (`REM/sm.cc`); referenced by every `REM/*.{cc,h}` file | L1 | `CORE/shader_core_stats.h` |
| `shader_core_mem_fetch_allocator` | `SM::m_mem_fetch_allocator` is `shared_ptr<shader_core_mem_fetch_allocator>` (`REM/sm.h`); `ldst_unit_sm` ctor takes it (`REM/ldst_unit_sm.{h,cc}`) | L1 | `CORE/shader_core_mem_fetch_allocator.h` (see §1c — mild TRANSFORM) |
| `shader_memory_interface`, `perfect_memory_interface` | `SM::m_icnt = new perfect_memory_interface(this,m_cluster)` / `new shader_memory_interface(...)` (`REM/sm.cc`) | L1/L3 seam (bridges core↔cluster icnt) | `CORE/shader_memory_interface.h` |
| `thread_ctx_t` | `SM::m_threadState` is `thread_ctx_t *` (`REM/sm.h`) | L1 | co-locate with `shd_warp.h` or `core_t` support header |
| `struct function_call_entry_info` | member of `shd_warp_t::m_function_call_stack`; used by `SM::func_exec_inst` call-stack push/pop | L1 | with `shd_warp.h` |
| `struct ifetch_buffer_t` | `Subcore::m_inst_fetch_decode_latch` is `ifetch_buffer_t` (`REM/subcore.h`), assigned in `Subcore::fetch` (`REM/subcore.cc`) | L1 | `CORE/ifetch_buffer.h` (or `REM` if only SM uses it) |

Note on `shader_core_config` layer: it subclasses L0 `core_config` (`abstract_hardware_model.h`) but
adds shader/SM-specific fields; placing it at L1 keeps L0 free of SM assumptions. Its option
registration (`shader_core_config::reg_options`) is the L0 "选项解析" surface and can stay callable
from `MAIN`.

### 1c. TRANSFORM — glue to shrink or re-point

| Symbol | Current shape | Transform |
| --- | --- | --- |
| `shader_core_ctx_wrapper` (`shader_core_wrapper.h`) | 90 pure-virtual methods (`grep -cE '= 0;'` = 90; `grep -cE '^\s*virtual '` = 91 incl. dtor). Includes `remodeling/new_stats.h` for `Element_stats`. | Shrink to the ~30-method surface actually invoked post-retirement (§2); `SM` becomes sole implementor. Keep as the formal L3↔L2 contract (roadmap §3.1). |
| `simt_core_cluster` (25 methods, `create_shader_core_ctx()=0`) | Holds `std::vector<shader_core_ctx_wrapper *> m_core`; abstract via one pure virtual factory. Only concrete subclasses = exec/trace cluster. | KEEP as L3 orchestration. After exec removal, only `trace_simt_core_cluster` remains; its factory `else` branch dies (unconditional `new SM`). Decide `m_core` element type per §2. |
| `shader_core_mem_fetch_allocator` | Defined in `shader.h`, subclass of L0 `mem_fetch_allocator`. `create_inst_memory_access` etc. | KEEP the class; only re-home out of the legacy TU. Verify no method touches a deleted legacy type. |
| `result_bus.h` `RRS` / `m_res_bus_improved` | `RRS::init(unsigned,unsigned,opndcoll_rfu_t*)` (`result_bus.h`), called at `shader_core_ctx` ctor `m_res_bus_improved.init(..., &m_operand_collector)` (`shader.cc`). `SM::get_loog_rrs()` **throws** `std::logic_error("LOOG is not compatible with this new accurate remodeling")` (`REM/sm.cc`). | LOOG/RRS is legacy-only on the remodeled path (the getter throws; only `shader_core_ctx` calls `RRS::init` with the operand collector). Recommend DELETE `result_bus.{h,cc}` with the operand collector, and drop the `get_loog_rrs`/`get_is_loog_enabled` wrapper virtuals. Flag for verification (Risk R6): confirm no `REM` code path calls a live `RRS`. |

---

## 2. Cluster 实际调用面 = 收缩后接口

**Now (measured).** The wrapper `shader_core_ctx_wrapper` declares **90 pure virtuals**
(`grep -cE '= 0;' shader_core_wrapper.h` = 90). The roadmap's "约 110" is an approximation; the
current measured count after stage-0..2 pruning is 90.

**What the cluster actually calls through `m_core[]`.** `simt_core_cluster` touches cores only via
`std::vector<shader_core_ctx_wrapper *> m_core`. Distinct methods invoked
(`grep -oE 'm_core\[[^]]*\]->[A-Za-z_]+' shader.cc | sort -u`, minus one commented line
`// m_core[0]->m_stats_map...`) = **28**:

```
accept_fetch_response, accept_ldst_unit_response, cache_flush, cache_invalidate,
can_issue_1block, cycle, display_pipeline, fetch_unit_response_buffer_full, get_cache_stats,
get_current_occupancy, get_icnt_power_stats, get_kernel, get_L0I_sub_stats, get_L1C_sub_stats,
get_L1D_sub_stats, get_L1I_sub_stats, get_L1T_sub_stats, get_n_active_cta, get_not_completed,
get_pdom_stack_top_info, increment_sm_stat_by_integer, init, isactive, issue_block2core,
ldst_unit_response_buffer_full, print_cache_stats, reinit, set_kernel
```

`trace_simt_core_cluster` (`trace_driven.cc`) adds through `m_core[]` only
`create_gpu_per_sm_stats` (and `init`, already counted) → union ≈ **29**.

**Other surviving polymorphic callers** (the wrapper is also held by non-cluster code that survives
retirement — these define the rest of the shrunken surface):

- `L0_icnt::m_shader` (`REM/l0_icnt.h`, remodeling L2) calls exactly 2:
  `get_num_subcores`, `set_subcore_req_fetch_L1I_priority`
  (`grep -oE 'm_shader->[A-Za-z_]+' REM/l0_icnt.cc`).
- `shd_warp_t`/`barrier_set_t`/`trace_shd_warp_t` (L1) call via `m_shader`/`get_shader()`:
  `get_config`(×12), `get_stats`, `get_gpu`, `get_sid`, `get_kernel_info`, `get_num_subcores`,
  `ptx_thread_done`, `num_cycles_to_stall_SM`, `incregfile_reads`, `incnon_rf_operands`,
  `broadcast_barrier_reduction`, `warp_waiting_at_barrier`, `warp_waiting_at_mem_barrier`,
  `warp_waiting_grid_barrier` (`grep -oE '(m_shader|get_shader\(\))->[A-Za-z_]+'`).
- `shader_memory_interface`/`perfect_memory_interface::m_core` call `inc_simt_to_mem`.
- `simt_core_cluster::gather_stats`/`gather_single_stat`/`reset_cycless_access_history` call the 4
  `Element_stats` methods (`create_gpu_per_sm_stats`, `gather_gpu_per_sm_stats`,
  `gather_gpu_per_sm_single_stat`, `reset_cycless_access_history`).

Union of the above ≈ **32–36 methods** — i.e. the interface shrinks from 90 to roughly a third.
The ~55 unused virtuals are the per-op stat incrementers (`incialu_stat`,`incimul_stat`,…,`inctensor_stat`,
`incsp_stat`,`incmem_stat`, ~30 of them) and legacy accessors that only the deleted
`simd_function_unit`/`pipelined_simd_unit`/`scheduler_unit`/`opndcoll_rfu_t` ever called — confirmed
by the fact that the remodeling EX units hold `SM*` directly (below) and increment stats on `SM`
concretely, not through the wrapper.

**Does polymorphism still buy anything?** No. Once the `shader_core_ctx` family is deleted, `SM` is
the **sole** implementor of `shader_core_ctx_wrapper`. Moreover the remodeling internals already
bypass the wrapper and hold the concrete type:

- `functional_unit::m_sm` is `SM *` (`REM/functional_unit.h`).
- `Subcore::m_sm` is `SM *` (`REM/subcore.h`).
- `ldst_unit_sm::m_core` is `SM *` (`REM/ldst_unit_sm.h`).

Only `L0_icnt` and the cluster still use `shader_core_ctx_wrapper *`. So the pure-virtual dispatch
is pure overhead with a single implementor.

**Options.**

- Option A — dissolve the wrapper; `simt_core_cluster::m_core` becomes `std::vector<SM *>`; L0_icnt
  holds `SM *`. Removes the 90-method interface and the vtable. Cost: L3 (`gpu-sim`/cluster) must
  `#include remodeling/sm.h`. That is the **allowed** L3→L2 direction and is *already* true — `gpu-sim.h`
  includes `remodeling/new_stats.h` and `remodeling/fusedMemory/coalescingStats.h` today — so it is
  not a layering regression, only a heavier compile include and loss of an explicit seam.
- Option B — keep `shader_core_wrapper.h` as the formal L3↔L2 contract but **shrink** it to the
  ~30–36 methods above; `SM` remains sole implementor; cluster/L0_icnt keep `shader_core_ctx_wrapper *`.
  Keeps L3 free of `remodeling/sm.h`, costs one vtable dispatch/call (negligible on a per-SM-cycle
  granularity), and is the smaller, lower-risk diff (delete unused virtuals, touch no call sites).

**Recommendation: Option B for stage 3.** It is exactly what roadmap §3.1 mandates
("L3↔L2 之间以收缩后的 core 接口为唯一通道（阶段三产物）") and what the safety-net ledger predicts
("`shader_core_wrapper.h` becomes the formal L3-L2 contract"). It also removes `new_stats.h`'s reach
into the contract only if the 4 `Element_stats` methods are re-typed; otherwise keep that one L2
include documented as the sanctioned seam. Option A (full `SM*`, dissolve) is a clean **stage-4**
follow-up once the shrunk interface has proven stable, and should not be forced in the high-risk
retirement stage.

---

## 3. `shd_warp_t` 归属裁定

**Today.** `shd_warp_t` is defined in `CORE/shader.h` and carries a mix of legacy and remodeling
members (`shd_warp_t` private section):

- Legacy: `scheduler_unit *m_scheduler` with `set_scheduler`/`get_scheduler` ("MOD. Added L0I").
- Remodeling: `IBuffer_Remodeled *m_IBuffer_remodeled`, `Dependency_State *m_dependency_state`
  (both `new`-ed in the **inline** constructor), `Subcore *m_subcore`.
- Neutral: warp id/pc/active-mask/ibuffer/atomic/membar/gridbar state, function-call stack.

Because the constructor is inline and calls `new IBuffer_Remodeled(...)` / `new Dependency_State(...)`,
`shader.h` **reverse-includes** `remodeling/ibuffer_remodeled.h` and `remodeling/warp_dependency_state.h`
(`grep -n '#include.*remodeling' shader.h`). The third include, `remodeling/l0_icnt.h`, is **stray/unused**
in `shader.h` — no `L0_icnt`/`L0I` symbol is referenced anywhere in the header
(`grep -nE 'L0_icnt|L0I' shader.h` matches only the include line), so it can be deleted outright,
independent of the `shd_warp_t` move. `trace_shd_warp_t : public shd_warp_t`
(`TRACE/trace_driven.h`) extends it with the actual trace instruction stream (`map_warp_traces`,
`get_next_trace_inst`), and **`SM` instantiates `trace_shd_warp_t` directly** (`REM/sm.cc`
`new trace_shd_warp_t`, `static_cast<trace_shd_warp_t *>`), which is why `REM/sm.cc` includes
`trace_driven.h` (the L2→L4 edge from §0.1).

The legacy member is dead on the remodeled path: `scheduler_unit` has no `REM` user
(`set_scheduler`'s only caller is `scheduler_unit::add_supervised_warp_id` at `shader.h`, and the
`REM/ibuffer_remodeled.h` mention is an unused forward decl). So `m_scheduler`/`set_scheduler`/
`get_scheduler` are removed with the scheduler family — `shd_warp_t` loses its last legacy tie.

**Placement options under L0–L4.**

- Option 1 — **`shd_warp_t` → L1 (`CORE/shd_warp.h`), `trace_shd_warp_t` stays L4, and SM stops
  instantiating the L4 subclass by construction.** Move the trace-stream ownership so that L2 never
  names an L4 type: give `SM` a factory hook (or a `create_shd_warp` provided by the L4 driver that
  hands SM ready-made `shd_warp_t*`), or push `map_warp_traces` down into an L1/L2-visible interface
  that `trace_shd_warp_t` fills. Consequence: breaks the new L2→L4 edge (`REM/sm.cc` no longer
  includes `trace_driven.h`); `shd_warp_t`'s remodeling members stay (L1 may depend on L2 types only
  via forward-decl + out-of-line ctor, so move the ctor body to a `.cc` to kill the `shader.h`→`REM`
  include). Highest architectural payoff, largest surface.
- Option 2 — **`shd_warp_t` → L1, keep the L2→L4 downcast as a documented, single-point exception.**
  Move the class + de-inline the constructor (kills three `shader.h`→`remodeling/` includes), but let
  `SM::create_shd_warp` keep `new trace_shd_warp_t` and the casts. Consequence: the L2→L4 edge
  survives into stage 4; ledger records it explicitly as deferred. Medium payoff, small surface,
  lowest risk. Does **not** by itself let `abstract_hardware_model`/`scoreboard` reverse-includes
  reach zero, but those are independent of this choice.
- Option 3 — **`shd_warp_t` → L2 (into `remodeling/`).** Since its only live owner is `SM` and it
  carries three remodeling members, co-locating with the SM model removes all `shader.h`→`REM`
  edges by absorption. Consequence: L1 no longer owns the warp abstraction; `barrier_set_t` and the
  memory interfaces (which reference `shd_warp_t` indirectly) would then sit above or beside it. But
  `shd_warp_t` also holds generic warp state used by L1 `barrier_set_t`/scoreboard-style facilities,
  and the roadmap explicitly lists `shd_warp_t`归属 as an L1-candidate decision — pushing it into L2
  contradicts the "L1 共享部件层" intent and would make any future non-SM consumer depend on L2.

**Recommendation: Option 2 for stage 3, converging to Option 1 in stage 4.** Rationale: the
mandatory, low-risk win this stage is *de-inlining the constructor and moving `shd_warp_t` to a
dedicated L1 header* — that alone deletes the three `shader.h`→`remodeling/` includes and lets
`shd_warp_t` survive the deletion of `shader.h`'s legacy body. Re-homing the trace-stream ownership
(Option 1) is a genuine ownership redesign that touches `SM::create_shd_warp`, `SM::func_exec_inst`,
and the trace driver together; per the roadmap it belongs with the stage-4 "warp_inst_t/shd_warp_t
remodeling 成员归属重整" work. Record the residual L2→L4 edge in the stage-3 ledger as an explicit
deferral (not a regression — it does not grow the tracked reverse-include set, which counts
`remodeling/`-inbound edges only). Keep `trace_shd_warp_t` in L4.

---

## 4. exec-path 移除清单 + cuda-sim 裁线

### 4.1 exec (PTX functional) 分支符号与文件

Selection is by **entrypoint**, not a runtime `if`. There is no `if(trace)…else(exec)` anywhere;
each `gpgpu_sim`/cluster subclass constructs its own sibling type.

- Trace entrypoint: `MAIN::main` → `gpgpu_trace_sim_init_perf_model` → `new trace_gpgpu_sim(...)`
  (`MAIN`). The trace binary never names any `exec_*` symbol
  (`grep -rn 'exec_' MAIN TRACE` = none).
- Exec entrypoint: `gpgpu_context::gpgpu_ptx_sim_init_perf` → `new exec_gpgpu_sim(...)`
  (`CORE/gpgpusim_entrypoint.cc`), whose only caller is the CUDA runtime shim
  `gpgpu-sim/libcuda/cuda_runtime_api.cc`. Unreachable from `MAIN`.

| exec symbol | Definition site | Referrers (all are defn or `new`, no call sites — dispatch is via base ptr) |
| --- | --- | --- |
| `exec_gpgpu_sim` | `CORE/gpu-sim.h` | ctor + `createSIMTCluster` body (`gpu-sim.cc`); `new exec_gpgpu_sim` in `gpgpusim_entrypoint.cc` |
| `exec_simt_core_cluster` | `CORE/shader.h` | `create_shader_core_ctx` body (`shader.cc`); `new exec_simt_core_cluster` in `exec_gpgpu_sim::createSIMTCluster` (`gpu-sim.cc`) |
| `exec_shader_core_ctx` | `CORE/shader.h` | 7 method bodies in `shader.cc` (`create_shd_warp`,`get_next_inst`,`decrement_trace_pc`,`get_pdom_stack_top_info`,`get_active_mask`,`func_exec_inst`,`checkExecutionStatusAndUpdate`) + **cross-file** `exec_shader_core_ctx::sim_init_thread` in `gpu-sim.cc`; `new exec_shader_core_ctx` in `shader.cc` |

Factory / registration points that shrink or die:

- `gpgpu_sim::createSIMTCluster()=0` (`gpu-sim.h`): exec override dies; trace override
  (`trace_gpgpu_sim::createSIMTCluster`, `trace_driven.cc`) stays. Keep the pure virtual (one
  subclass left) or de-virtualize in stage 4.
- `simt_core_cluster::create_shader_core_ctx()=0` (`shader.h`): exec override
  (`shader.cc`) dies; trace override (`trace_driven.cc`) stays and drops its `else` branch (§5).

What dies **with** exec (does not survive as trace-needed):

- `exec_gpgpu_sim::createSIMTCluster` construction path → `new exec_simt_core_cluster` → `new exec_shader_core_ctx`.
- Base default bodies that only exec objects dispatched to (trace overrides them, and trace_ dies too
  when SM is unconditional): `shader_core_ctx::init_warps`, `shader_core_ctx::issue_warp` base bodies.
- Functional-sim surface reachable only via the CUDA entrypoint (never from `MAIN`):
  `functionalCoreSim` (`cuda-sim.h`, impl `cuda-sim.cc`), `cuda_sim::gpgpu_cuda_ptx_sim_main_func`
  (callers `gpgpusim_entrypoint.cc` only). `gpgpu_sim::set_functional_sim` has **zero callers**
  repo-wide → the functional-sim flag is never enabled; `is_functional_sim()` is effectively always
  false on the trace path. (Keep the inherited `gpgpu_functional_sim_config` — shared config.)

Must-KEEP virtuals (trace path still overrides them; deletion of exec must not touch the base
**declarations**). Note: after §5 makes `SM` unconditional, `trace_shader_core_ctx` is also deleted,
so these matter only transiently — but if the retirement is staged (delete exec first, then
`shader_core_ctx`), keep them until the `shader_core_ctx` family is removed as a unit:
`checkExecutionStatusAndUpdate`, `func_exec_inst`, `sim_init_thread`, `create_shd_warp`,
`get_next_inst`, `decrement_trace_pc`, `get_pdom_stack_top_info`, `get_active_mask` (pure virtuals),
plus non-pure `init_warps`, `issue_warp`, and `core_t::updateSIMTStack`.

**Cross-check — `core_t::updateSIMTStack` is NOT dead.** `SM` derives from `core_t` and calls the
inherited 2-arg `updateSIMTStack(warp_id, pipe_reg)` (`REM/sm.cc`); `SM` does **not** override it
(`grep -n updateSIMTStack REM/sm.h` = empty). So `core_t::updateSIMTStack` (base body in
`abstract_hardware_model.cc`) must be KEPT. Only the 3-arg `trace_shader_core_ctx::updateSIMTStack`
overload dies with `trace_shader_core_ctx`.

### 4.2 cuda-sim/ 裁线（保守：证明不可达才删）

Decisive constraint: the simulator is built as `libcudart.so` **without** `-ffunction-sections`/
`--gc-sections` (`gpgpu-sim/Makefile`), so **the entire `cuda-sim/` subtree is link-present in the
trace binary today** — verified by `nm -C` on the shipped `.so` (full PTX parser `ptx_recognizer`,
`instructions.cc` `*_impl`, `ptx_thread_info::ptx_exec_inst` all defined). The PTX-vs-trace split is
therefore a **runtime** distinction, not a current-link one. No `cuda-sim/` file can be `rm`-ed in
isolation without first cutting one of three seams:

1. `gpgpu_context` ctor (`libcuda/gpgpu_context.h`) unconditionally `new`s `ptx_recognizer`,
   `ptxinfo_data`, `cuda_runtime_api`, `cuda_sim`, `cuda_device_runtime`, `ptx_stats` — even in trace mode.
2. Non-trace `else` branches still compiled: `SM::sim_init_thread` else → `ptx_sim_init_thread`
   (`REM/sm.cc`); `SM::func_exec_inst` else → `execute_warp_inst_t`→`ptx_exec_inst` (`REM/sm.cc`);
   the exec core.
3. Trace-needed symbols co-located inside PTX `.cc` files: `function_info` base ctor +
   `symbol_table::get_ptx_version` (`ptx_ir.cc`), `ptx_sim_kernel_info` +
   `cuda_sim::ptx_opcocde_latency_options` (`cuda-sim.cc`), `gpgpu_context::ptx_reg_options`
   (`ptx_loader.cc`) — these are direct undefined refs of `main.o`/`trace_driven.o`
   (`nm -C -u bin/release/accel-sim.out`).

**Ruling (conservative).** Stage 3 **KEEPS all of `cuda-sim/`.** The trace path genuinely links and
runtime-uses this KEEP set:

| KEEP (trace-linked, evidence) |
| --- |
| `cuda-sim.cc`/`cuda-sim.h` — `ptx_opcocde_latency_options` (`MAIN`), `ptx_sim_kernel_info` (resource alloc, `REM/sm.cc`,`gpu-sim.cc`), `cuda_sim` object (ctor) |
| `ptx_ir.{cc,h}` — `function_info` is the base of `trace_function_info` (`trace_driven.h`); `symbol_table::get_ptx_version` is an undefined-ref of the trace binary |
| `ptx_loader.{cc,h}` — `gpgpu_context::ptx_reg_options` DEFINED here, called by `MAIN`; `ptxinfo_data` constructed in gpgpu_context ctor |
| `ptx-stats.{cc,h}` — `ptx_file_line_stats_*` called at runtime by `shader.cc`,`gpu-sim.cc` |
| `memory.{cc,h}` — `memory_space_impl<8192>` allocated in `gpgpu_t` ctor (base of `trace_gpgpu_sim`), runs every launch |
| `cuda_device_runtime.{cc,h}` — launch-latency config `g_kernel_launch_latency`/`g_TB_launch_latency` read by `abstract_hardware_model.cc`, registered in `gpu-sim-config.cc` |
| `ptx_sim.{cc,h}` — `ptx_thread_info`/`ptx_reg_t` types threaded through `core_t`; `ptx_thread_info::get_pc` link-referenced by `SM::next_pc` (`REM/sm.cc`) even though `m_thread[tid]` stays NULL in trace |
| `opcodes.{h,def}`, `half.h` — compile deps pulled via `ptx_sim.h` |

DELETE-CANDIDATE, but **out of stage-3 scope** — runtime-dead yet link-present; removable only as a
coordinated multi-seam cut (a separate "PTX functional retirement" effort, ≈20k+ lines):
`instructions.cc` (+ its exclusive includes `half.hpp`, `cuda-math.h`), `cuda_device_printf.{cc,h}`,
the PTX parse subsystem `ptx_parser.cc`+`ptx.y`+`ptx.l`+`ptxinfo.y`+`ptxinfo.l`+`decuda_pred_table/`,
and (outside cuda-sim) the 238 KB `libcuda/cuda_runtime_api.cc`. Stage 2 already recorded two PTX-only
couplings deliberately retained (`ptx_loader.cc` cuobjdump invocation; `cuda-sim.cc` Watch-Your-Step
path) — consistent with keeping cuda-sim intact now.

**What stage 3 may legitimately simplify inside this area** (because it dies with the legacy core,
not with cuda-sim): the exec `else`-branches of `SM::func_exec_inst`/`SM::sim_init_thread` become the
only branch once trace mode + captured-binary is enforced — but since those call into KEEP cuda-sim
types and touch functional-sim semantics, treat any such simplification as optional and gate it
behind byte-identical goldens (Risk R5). The safe stage-3 position is: delete the exec **core**
classes; leave the cuda-sim functional subsystem KEEP.

### 4.3 SM 复制粘贴对账（roadmap 阶段三 item 4）

`SM` re-implements the legacy timing methods (§1a list) under the same names
(`SM::cycle`/`fetch`/`issue`/`decode`/`writeback` via `Subcore`, `SM::func_exec_inst`,
`SM::checkExecutionStatusAndUpdate`, `SM::warp_inst_complete`, `SM::create_shd_warp`, …). Once
`shader_core_ctx` is deleted, each `SM` copy is automatically the sole implementation — there is no
"reconcile two live copies" work, only a check for shared **free-function** helpers that both TUs
reference (e.g. `get_oprnd_type` in `TRACE`, `ptx_sim_kernel_info` in cuda-sim). Action: after
deletion, grep for any now-single-caller free function that was extracted to serve both cores and
inline/relocate it (low priority, stage-4-eligible).

---

## 5. `-is_SM_remodeling_enabled` 移除计划

- **Registration site:** `shader_core_config::reg_options` in `CORE/gpu-sim-config.cc`
  (`reg_option(opp, "-is_SM_remodeling_enabled", OPT_BOOL, &is_SM_remodeling_enabled, ...,
  "is_SM_remodeling_enabled (default = enabled)")`).
- **Member:** `bool is_SM_remodeling_enabled` on `shader_core_config` (`CORE/shader.h`).
- **Every branch testing it** (`grep -rn is_SM_remodeling_enabled SRC TRACE`):
  1. `trace_simt_core_cluster::create_shader_core_ctx` (`TRACE/trace_driven.cc`) — the live
     `if(...) new SM else new trace_shader_core_ctx`. Remove the `else`; `new SM` unconditional;
     delete `trace_shader_core_ctx`.
  2. `exec_simt_core_cluster::create_shader_core_ctx` (`CORE/shader.cc`) — same shape; dies entirely
     with `exec_simt_core_cluster` (§4).
  3. `gpgpu_sim_config::validate_supported_trace_contract` (`CORE/gpu-sim-config.cc`) — the guard
     `if (!sc.is_SM_remodeling_enabled) { return; }`. **Important:** today validation *early-returns*
     (does not reject) when the flag is off — the legacy path is "outside the contract, unconstrained".
     After removal, the flag is gone and `SM` is always the core, so this early-return must be deleted
     and the remaining checks (trace mode, sub-core, ibuffer, pipeline depth) run unconditionally.
- **Config files containing it:** **39** `*.config` files
  (`grep -rl is_SM_remodeling_enabled --include=*.config` = 39; `is_ibuffer_remodeled_enabled` and
  `gpgpu_sub_core_model` likewise 39). Per the roadmap 删除原则, the removed option **line must be
  deleted from every shipped config** in the same change; and per the supported contract, a config
  that still sets `-is_SM_remodeling_enabled 0` must fail fast (unknown-option error) rather than be
  silently ignored — the option parser already errors on unknown options once the registration is
  removed, satisfying the acceptance criterion "被删配置项在配置文件中出现时给出明确错误而非静默忽略".

**Golden re-approval:** editing the 39 tracked `gpgpusim.config` files changes the SHA-256 that
`goldens.json` records as the `configs` provenance contract — exactly the hash-lock that blocked the
stage-2 `-network_mode` sweep (`2026-07-17-dead-weight.md`, Item 3 / "Deferred-item completion").
This step therefore **requires the documented golden re-approval procedure** (end of
`2026-07-17-dead-weight.md`): run `observe`; assert per-case observed stats are byte-identical to the
approved goldens and that the comparison contract differs **only** in config `sha256` fields; commit
the resulting minimal goldens diff (config hashes + `source_commit`); confirm `check` passes 4/4.
Stats identity proves the option removal changed no simulated behavior — the flag was already
`1`-effective in every tested config.

---

## 6. 停滞分支吸收（`archive/true-path-scoreboard-cleanup`）

**Identity & relationship.** `archive/true-path-scoreboard-cleanup` is a **tag** →
`6ded32333c3728cb84f7f930730f650e277acd2f`, identical to `origin/refactor/true-path-scoreboard-cleanup`
(`git rev-list --left-right --count` = `0 0`). Merge-base with `dev_dzw` = `f07e4ae`. The tag is
**2 commits ahead** of the merge-base (`465af43` "refactor: remove scoreboard legacy path, keep only
control-bit dependency"; `1969355` docs), `dev_dzw` is **35 ahead** (stages 0–2). The 2 tag commits
are **not** on `dev_dzw`. `git diff --stat dev_dzw...archive/...` = 20 files, +656/−254 (essentially
all in `465af43`). **Do not merge/cherry-pick the commit** (it also rewrites `arch.md`/`CLAUDE.md`/
`.gitignore` that the 35 `dev_dzw` doc commits reworked) — **re-apply the source changes** against
current `dev_dzw`.

**What `465af43` did (still valid, targets present verbatim on `dev_dzw`):**

1. Deleted `exec_shader_core_ctx` (class + 7 bodies in `shader.cc`, `sim_init_thread` in `gpu-sim.cc`)
   and made `create_shader_core_ctx` construct `SM` unconditionally — **the same work as §4/§5**.
2. Removed the `use_traditional_scoreboarding` guarded blocks in `REM/sm.cc`, `REM/sm.h`,
   `REM/subcore.cc`, `REM/subcore.h`, `REM/functional_unit.cc`: the branches calling
   `m_scoreboard->reserveRegisters[_remodeling]`, `releaseRegisters[_remodeling]`,
   `checkCollision_remodeling`, `pendingWrites`, and `m_scoreboard_WAR->...` (the traditional-
   scoreboarding retrocompat path). Kept only the control-bit / `dependency_state` (True-Path) path.
3. Registered `-is_loog_enabled` in `gpu-sim-config.cc` (a real gap on `dev_dzw`: the member is read
   in ≥6 places but never registered).
4. Added standalone GoogleTest scaffolding.

**Critical caveat — the tag does NOT fix the scoreboard reverse-include.** `465af43` never touches
`scoreboard.cc`/`scoreboard_reads.cc`; on the tag those files still `#include "remodeling/sm.h"` (and
`register_file.h`), and `SM` still holds `std::shared_ptr<Scoreboard> m_scoreboard` /
`m_scoreboard_WAR`. The tag removes the hot-path **callers** of the scoreboard — a **prerequisite**
for deleting the `_remodeling` scoreboard methods and the `m_scoreboard*` members, which is the step
that actually breaks the L1→L2 cycle (see §7 step 5). So the tag is a *partial precursor*, not the
finish line.

**C++ unit tests it added.** Location on the tag: `tests/CMakeLists.txt`,
`tests/test_dependency_path.cc`, `tests/test_pipeline_routing.cc`. Framework = **GoogleTest**
(`find_package(GTest REQUIRED)`, `gtest_discover_tests`). Build = a **standalone CMake project**
(`project(gpu_simulator_tests)`, C++17) producing one `run_tests` binary; it links only
`GTest::gtest gtest_main pthread` and compiles **no simulator object** — it merely adds the
`gpgpu-sim/src/...` include dirs so two header-only pure functions resolve. **Not wired into the
Makefile build.** 17 tests total:

- `test_dependency_path.cc` — 10 tests over `uses_control_bit_dependency` / `uses_trace_mode_scoreboard`
  (new header `dependency_path.h`). Key assertions: True-Path selection
  `uses_control_bit_dependency(trace=true, captured=true, scoreboarding=false)` and a
  `MutualExclusivity` test asserting control-bit vs scoreboard are mutually exclusive when
  `trace_mode=true`.
- `test_pipeline_routing.cc` — 7 tests over `resolve_int_predicate_target` / `resolve_sp_op_target`
  (new header `pipeline_routing.h`): unified-INT predicate routing, `fp32-in-INT` steering, IMAD
  always → SP.

**Absorption plan & required adaptation.**

- Fold items (1) and (2) into the stage-3 ordered steps (§7 steps 1 and 5) — they are the retirement,
  re-applied against current `dev_dzw`, not a separate merge.
- Item (3) `-is_loog_enabled` registration: absorb as a small standalone fix (or drop entirely if
  §1c deletes `result_bus`/RRS and the LOOG surface — decide with Risk R6).
- Tests: adopt `dependency_path.h` + its True-Path tests as a **regression guard during** retirement,
  but note the erosion — after the legacy path is fully retired, `uses_trace_mode_scoreboard` and the
  `is_remodeling_scoreboarding_enabled` parameter describe a branch that no longer exists; the 5
  `UsesTraceModeScoreboard.*` tests and the two-branch `MutualExclusivity` premise become vestigial
  and should be trimmed to the surviving control-bit assertion. `pipeline_routing.h` and its 7 tests
  are **stale as-is**: `resolve_int_predicate_target`/`resolve_sp_op_target` are not called by any
  simulator source even on the tag (a parallel reimplementation of routing that still lives inline in
  `REM/subcore.cc`/`REM/functional_unit.cc`) — to make them meaningful, first refactor the inline
  routing to call these functions, then they become the primitive unit tests roadmap §3.1 wants; until
  then they test a mirror that can drift.
- Wire the GTest target into the build (or at least a documented `ctest` invocation) before treating
  it as a gate; the current Python gate stays authoritative.

---

## 7. 有序步骤计划（gate-green commits + 依赖方向账本）

Constraints (roadmap 阶段三): dedicated branch; every small step an independent commit with a green
gate; any red → stop and report, no scope creep. Reverse-include ledger must shrink **monotonically**
from the stage-2 baseline. Baseline (from `2026-07-17-dead-weight.md`, run in `SRC`):
`abstract_hardware_model.cc`, `gpu-sim.h`, `scoreboard.cc`, `scoreboard_reads.cc`, `shader.cc`,
`shader.h`, `shader_core_wrapper.h` reverse-include `remodeling/`; plus `abstract_hardware_model.h`
references `functional_unit` at 4 sites.

| Step | What moves / dies | Gate | Reverse-include delta | Golden re-approval? | Revert point |
| --- | --- | --- | --- | --- | --- |
| **1. Remove exec + make SM unconditional** | Delete `exec_gpgpu_sim`,`exec_simt_core_cluster`,`exec_shader_core_ctx` (§4.1) + `exec_shader_core_ctx::sim_init_thread` in `gpu-sim.cc`; delete `-is_SM_remodeling_enabled` registration/member/branches (§5); `create_shader_core_ctx` → unconditional `new SM`; delete `trace_shader_core_ctx` + its `create_shd_warp`/`get_next_inst`/… ; delete validation early-return | build + unittest + `check` 4/4 | no change yet (`shader.cc/h` still host KEEP facilities) | **YES** — 39 config files lose the option line → config-hash re-approval (§5) | commit revert |
| **2. Extract KEEP facilities to L1 headers** | Move `shader_core_config`, `shader_core_stats(_pod)`, `shd_warp_t` (+`function_call_entry_info`,`thread_ctx_t`), `barrier_set_t`, `shader_core_mem_fetch_allocator`, `ifetch_buffer_t`, `shader_memory_interface`/`perfect_memory_interface` out of `shader.{h,cc}` into dedicated L1 headers/TUs; **de-inline `shd_warp_t` ctor** into a `.cc`; delete the stray `remodeling/l0_icnt.h` include | build + unittest + `check` 4/4 | `shader.h` drops `remodeling/ibuffer_remodeled.h`/`warp_dependency_state.h` (moved with `shd_warp_t`, ctor de-inlined to use fwd-decls) + the stray `l0_icnt.h` → **`shader.h` reverse-include removed** | pure code | commit revert |
| **3. Delete legacy pipeline machinery** | Delete scheduler family, `opndcoll_rfu_t`, `simd_function_unit`/`pipelined_simd_unit`+subclasses, legacy `ldst_unit`, `pipeline_stage_name_t`, `insn_latency_info`, `register_bank`/`coalesced_segment`/`check_kernel_launch_limitation`, `result_bus`/RRS (Risk R6), and the now-empty `shader_core_ctx` base | build + unittest + `check` 4/4 | `shader.cc` reverse-include removed (its `remodeling/sm.h`,`new_stats.h` includes go when `shader_core_ctx` body is gone) → **`shader.cc` reverse-include removed** | pure code | commit revert |
| **4. Shrink `shader_core_ctx_wrapper` (Option B)** | Remove the ~55 unused virtuals (per-op stat incrementers + legacy accessors); keep the ~30-method surface (§2); re-type or document the `Element_stats` methods' `new_stats.h` include | build + unittest + `check` 4/4 | `shader_core_wrapper.h` becomes the formal L3↔L2 contract; its `new_stats.h` include either removed (if `Element_stats` re-typed) or **retained as the one sanctioned seam** | pure code | commit revert |
| **5. Absorb scoreboard True-Path cleanup** | Re-apply `465af43` items (2): remove `use_traditional_scoreboarding` guarded blocks in `REM/sm.cc`/`subcore.cc`/`functional_unit.cc`; drop `SM::m_scoreboard`/`m_scoreboard_WAR` members; delete the now-dead `Scoreboard::*_remodeling` / `Scoreboard_reads::*_remodeling` methods; relocate `check_is_reserved_regs_remodeling`'s `register_file.h` dependency (inline the reserved-reg encoding or move the helper to L1) | build + unittest + `check` 4/4 | `scoreboard.cc`/`scoreboard_reads.cc` drop `#include remodeling/sm.h`(+`register_file.h`) → **both scoreboard reverse-includes reach ZERO** | pure code | commit revert |
| **6. Add C++ primitive tests (adapted)** | Absorb `dependency_path.h` + trimmed True-Path tests; wire a build/ctest hook; defer `pipeline_routing.h` until routing is refactored to call it | build + unittest + `check` 4/4 + `run_tests` green | none | pure code | commit revert |

**End-of-stage ledger target.** After step 5, `remodeling/`-inbound reverse-includes are:
`shader.cc` (gone, step 3), `shader.h` (gone, step 2), `scoreboard.cc`/`scoreboard_reads.cc` (gone,
step 5), `shader_core_wrapper.h` (now the sanctioned contract — either zero or one documented
`new_stats.h` seam, step 4), `gpu-sim.h`/`gpu-sim.cc` (allowed L3→L2, unchanged). The ledger shrinks
monotonically every step and never grows.

**Explicitly deferred to stage 4 (with reason):** `abstract_hardware_model.cc` →
`remodeling/register_file.h` and `abstract_hardware_model.h` → `functional_unit` (4 sites). Both are
bound to `warp_inst_t`'s remodeling members (`m_fu_assigned`; `warp_inst_t::get_number_of_uses_per_operand`
uses a `register_file.h` helper at `abstract_hardware_model.cc`). The roadmap assigns
"`warp_inst_t`/`shd_warp_t` remodeling 成员的归属重整" to **stage 4**; attempting the L0→L2 severance
here would drag warp_inst_t restructuring into the high-risk retirement stage. Ledger records these
two as **deferred, not regressed** (they do not grow the count). Likewise the residual L2→L4 edge
`REM/sm.cc` → `trace_driven.h` (trace-stream ownership, §3 Option 1) is deferred to stage 4.

---

## 8. 风险登记（Risk register）

| ID | Risk | Detection method |
| --- | --- | --- |
| **R1** | Hidden legacy-method calls from cuda-sim into `shader_core_ctx` (e.g. `functionalCoreSim`/`checkExecutionStatusAndUpdate` reaching a deleted method) | Before step 3, `grep -rn 'shader_core_ctx\|->func_exec_inst\|->checkExecutionStatusAndUpdate' cuda-sim libcuda`; and after each step, a clean-tree rebuild — a missed caller fails the **link** (no `--gc-sections`, so undefined refs surface immediately). Any new undefined symbol = stop. |
| **R2** | Barrier / atomic paths shared by both models (`barrier_set_t`, `decrement_atomic_count`, membar/gridbar) subtly change when the class moves or when the wrapper shrinks | `barrier_set_t` semantic stats are golden-locked (`gpu_tot_sim_cycle`, throttle counters in `safety-net.md`). Run `check` 4/4 after step 2 and step 4; the FP64/HALF fixtures exercise DP throttle + barrier retire. Any stat drift = revert. |
| **R3** | Kernel-scheduler / icnt-handler **friend** relationships or direct member access into `shader_core_ctx` internals break when the class is deleted | `grep -rn 'friend .*shader_core_ctx\|friend class simt_core_cluster' SRC`; inspect `gpu-sim.cc` (kernel scheduler `issue_block2core` path) and `L0_icnt`/`icnt_wrapper` for direct field access. Compile-time failure catches broken friendship; enumerate before step 3. |
| **R4** | Stats printing reaching legacy members — `gpu-sim.cc` prints per-SM stats via the wrapper; shrinking the interface (step 4) may drop a virtual a print path calls | The stat print set is golden-locked (`safety-net.md` "Remodeled semantic statistics", byte-identical requirement). `grep -rn 'get_stats\|gather_gpu_per_sm\|m_stats_map\|create_gpu_per_sm_stats' gpu-sim.cc` to enumerate print callers before pruning virtuals; keep every virtual in that set. `check` 4/4 is the oracle. |
| **R5** | `SM::func_exec_inst`/`SM::sim_init_thread` exec `else`-branches touch KEEP cuda-sim functional types; over-eager simplification changes functional-sim behavior on non-captured traces | Keep those branches unless a fixture proves the `else` unreachable under the supported contract (captured-binary traces). Gate any change behind byte-identical goldens **and** an added SFU/non-captured fixture if one is introduced (safety-net notes flag SFU coverage as a gap). Default: do not touch in stage 3. |
| **R6** | `result_bus`/`RRS` wrongly deleted — a live `REM` path uses it despite `SM::get_loog_rrs` throwing | Before step 3: `grep -rn 'RRS\|result_bus\|m_res_bus\|get_loog_rrs\|is_loog_enabled' REM`; confirmed today only `SM::get_loog_rrs` (throws) + `get_is_loog_enabled` reference it. Re-run at step 3; if any live call appears, keep `result_bus` and only sever the `opndcoll_rfu_t*` param. |
| **R7** | `check_kernel_launch_limitation` guard silently lost — `SM::issue_block2core` never called it, so retirement drops a launch sanity check | `grep -rn 'check_kernel_launch_limitation\|can_issue_1block\|issue_block2core' SRC TRACE`; decide explicitly to re-host in `SM::issue_block2core` or delete. Behavior is golden-neutral (tested configs launch within limits), so absence won't fail the gate — must be a **conscious** call, logged in the validation record. |
| **R8** | Config-hash golden lock (step 1) mishandled → false "no drift" or spurious mismatch | Follow the documented re-approval procedure exactly (`2026-07-17-dead-weight.md` end): `observe` first, script-assert stats byte-identical and comparison contract differs ONLY in config `sha256`, then `check`. A stats diff of any non-hash field = real behavior change = stop. |
| **R9** | The new L2→L4 edge (`REM/sm.cc`→`trace_driven.h`) is forgotten and silently persists past stage 4 | Add an explicit ledger line in the stage-3 validation record for `remodeling/`-**outbound** high-layer includes (`grep -n 'trace-driven\|#include.*trace' REM/*.cc`), separate from the inbound-reverse-include ledger, so it is tracked to zero in stage 4. |
| **R10** | Independent-rerun / OMP pinning drift — a "hang" that is really libgomp oversubscription masks a real regression | Per `safety-net.md`, pin `OMP_NUM_THREADS=1` on every manual run; require an independent-agent rerun of build+unittest+`check` for the stage sign-off, as the roadmap execution model mandates. |

---

## Appendix — key grep commands (reproducible, run in `SRC` unless noted)

- Wrapper size: `grep -cE '= 0;' gpgpu-sim/shader_core_wrapper.h` → 90.
- Cluster call set: `grep -oE 'm_core\[[^]]*\]->[A-Za-z_]+' gpgpu-sim/shader.cc | sed 's/.*->//' | sort -u` → 28 (+commented `m_stats_map`).
- Reverse-include ledger: `grep -rln '#include.*remodeling/' --include=*.cc --include=*.h . | grep -v '/remodeling/' | sort`.
- SM/wrapper inheritance: `grep -n 'class SM' gpgpu-sim/remodeling/sm.h`; `grep -n 'class shader_core_ctx ' gpgpu-sim/shader.h`.
- Factory branch: `grep -n 'is_SM_remodeling_enabled' ../trace-driven/trace_driven.cc gpgpu-sim/shader.cc gpgpu-sim/gpu-sim-config.cc`.
- Remodeling holds `SM*`: `grep -rn 'SM *\*m_' gpgpu-sim/remodeling/{subcore.h,functional_unit.h,ldst_unit_sm.h}`.
- Config file count: `grep -rl is_SM_remodeling_enabled --include=*.config <gpu-simulator>` → 39.
- Archive tag: `git merge-base dev_dzw archive/true-path-scoreboard-cleanup`; `git diff --stat dev_dzw...archive/true-path-scoreboard-cleanup`.

---

## 9. 统筹者复核裁定（2026-07-17）

本设计经统筹者复核批准，以下三个设计中留白的决定点按删除原则裁定：

1. **`check_kernel_launch_limitation`（R7）：删除，不再宿主。** 支持路径（`SM::issue_block2core`）
   从未调用它；tested configs 的启动均在限制内（golden 中性）。删除记入验证记录作为有意识决定。
2. **LOOG/`result_bus`/RRS（§1c、R6、§6 item 3）：整面删除。** `SM::get_loog_rrs` 抛异常即宣告
   不兼容；`-is_loog_enabled` 从未注册（归档分支补注册的做法不吸收）。删除 `result_bus.{h,cc}`、
   wrapper 中 `get_loog_rrs`/`get_is_loog_enabled` 虚函数、`is_loog_enabled` 成员及其全部读点。
   步骤三执行前按 R6 复验 grep。
3. **`Element_stats`/`new_stats.h` 接口缝（§2 Option B 尾注）：阶段三保留为唯一被记账的 L3→L2
   契约内缝。** 重新定型 `Element_stats` 属阶段四配置/统计收敛工作，不并入高风险退役阶段。

步骤计划（§7）按 6 步执行不变；执行分支命名 `refactor/legacy-shader-retirement`。
