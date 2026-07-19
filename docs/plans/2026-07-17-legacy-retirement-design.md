# 阶段三动刀前置：Legacy Shader 时序路径退役 — 目标架构细化设计

Status: REVISED after adversarial review (no-ship, 6 findings) and the user's control-bit-only
scope decision (§0.2). Design only; no simulator source modified — this document is the single
edited file. The 6 findings were verified against live code by the coordinator; the anchors below
were re-grepped for this revision.
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

## 0.2 控制位唯一（Control-Bit-Only）范围决定 — 贯穿全文

用户已就退役范围拍板：**scoreboard 依赖模式随 legacy shader 路径一并退役，只保留 control-bit
（`Dependency_State`：stall counter、wait barriers、yield）路径。** 本决定改写 §6/§9 与步骤计划，
下述所有清单据此裁定。这一并解决了 adversarial finding 2（"scoreboard 不是 legacy 遗留物"）——
scoreboard 不再被当作"遗留残渣"，而是**被有意退役的特性**。

**决定的具体含义（code-anchored）：**

1. **删除 scoreboard 分支，只留 control-bit 分支。** `use_traditional_scoreboarding` 分支存在于
   `REM/subcore.cc`（`Subcore::issue`：`use_traditional_scoreboarding` 计算 +
   `shared_sm->get_scoreboard()->checkCollision_remodeling` 的 ready 检查，经 `issue_warp` 参数透传）、
   `REM/sm.cc`（`SM::issue_warp` 的 `reserveRegisters[_remodeling]`、`SM::instruction_retirement`
   的 `releaseRegisters[_remodeling]`）、`REM/functional_unit.cc`
   （`m_sm->get_scoreboard_WAR()->releaseRegisters_remodeling`）。三处的 scoreboard 分支删除，只留
   `else`（control-bit：yield/stall_counter/wait-barrier/ldgsts）分支。另有两处**无条件**引用
   scoreboard 的点需同步归一（见 §6，golden-neutral 论证）：`SM::check_if_warp_has_finished_executing_and_can_be_reclaim`
   把 `!m_scoreboard->pendingWrites && !m_scoreboard_WAR->pendingReads` 两个合取项去掉，只留
   `!are_pending_dependencies() && !is_atomic_pending()`；`SM::warp_waiting_at_mem_barrier` 去掉
   `use_traditional_scoreboarding` 分支，无条件走 `are_all_wait_barrier_ready(warp_id)`。
   证据：`grep -rn 'use_traditional_scoreboarding' REM/` → `sm.cc`/`subcore.cc`/`sm.h`/`subcore.h`；
   调用点 `subcore.cc`（`checkCollision_remodeling`）、`functional_unit.cc`
   （`releaseRegisters_remodeling`）、`sm.cc`（`reserveRegisters_remodeling`/`releaseRegisters_remodeling`/
   `pendingWrites`/`pendingReads`）。

2. **删除 `Scoreboard` 类族与其成员。** `Scoreboard`（`scoreboard.{cc,h}`）、`Scoreboard_reads`
   （`scoreboard_reads.{cc,h}`），以及 `SM::m_scoreboard`/`m_scoreboard_WAR`（`REM/sm.h`）、
   `SM::get_scoreboard`/`get_scoreboard_WAR`（`REM/sm.cc`）。**注意** `ldst_unit_sm` 也持有
   `m_scoreboard`/`m_scoreboard_reads`（`REM/ldst_unit_sm.h`），但只在 ctor 里赋值、从不调用其方法
   （`grep -n 'm_scoreboard' REM/ldst_unit_sm.cc` → 仅 line 162–163 的 dead store），故连同 ctor
   的 scoreboard 形参一并删除。删除后 `scoreboard.cc` 对 `remodeling/sm.h`+`remodeling/register_file.h`、
   `scoreboard_reads.cc` 对 `remodeling/sm.h` 的反向 include **归零** —— 计入 §7 step 5 账本。
   证据：`grep -n '#include' scoreboard.cc scoreboard_reads.cc | grep -i 'remodeling'`。

3. **删除 scoreboard 选项族与 8 个 SC 配置。** 选项 `-is_remodeling_scoreboarding_enabled`
   （`gpu-sim-config.cc`，member `shader_core_config::is_remodeling_scoreboarding_enabled`，`shader.h`）、
   `-scoreboard_war_max_uses_per_reg`、`-scoreboard_war_mode`（后两者的 member 亦在 `shader.h`）——
   三者只被 `Scoreboard`/`Scoreboard_reads` 消费（`scoreboard_reads.cc` ctor + `SM::init` 构造
   `m_scoreboard_WAR` 时读 `scoreboard_war_max_uses_per_reg`）。8 个 `SM86_RTXA6000_SC_*` 配置目录
   **整目录删除**（它们的存在理由就是把 `-is_remodeling_scoreboarding_enabled` 置 1 跑 scoreboard
   依赖矩阵；随 scoreboard 退役即失去意义）。这是 config-file 改动 → 与选项 sweep 相同的 golden
   重批流程（`2026-07-17-dead-weight.md` 末尾）。
   **实测校正（诚实记账）**：SC 配置与基线 `SM86_RTXA6000` 的差异不止两个旋钮 —— 除
   `-is_remodeling_scoreboarding_enabled 1` 外还翻转了 `-is_instruction_prefetching_enabled 1`、
   `-prefetch_per_stream_buffer_size 8`、`-scoreboard_war_mode`。但这些都是同一 scoreboard 测试矩阵
   的附带设置，不构成保留 SC 目录的理由，故整目录删除而非逐行改；而 scoreboard 选项族要从**剩余**
   config 里删除（39 个带 `-is_remodeling_scoreboarding_enabled` 的配置里，两个 gate 配置也含之），
   这才是触发 golden 重批的部分。
   证据：`grep -rl 'is_remodeling_scoreboarding_enabled 1' --include=gpgpusim.config` → 恰好 8 个 SC
   目录；`grep -rl 'is_remodeling_scoreboarding_enabled' --include=gpgpusim.config` → 39（与
   `is_SM_remodeling_enabled` 同集）。

4. **新语义契约：拒绝 non-captured kernel。** 今天 `!is_captured_from_binary` 的 kernel 会*回退到
   scoreboard*（`SM::instruction_retirement`/`SM::issue_warp` 里
   `use_traditional_scoreboarding = !m_physical_warp[warp_id]->get_kernel_info()->is_captured_from_binary`）。
   该回退删除后，non-captured kernel 必须在启动/launch 处以明确 fatal error 拒绝 —— 支持契约收紧为
   **"captured-from-binary traces only"**。落点：配置层
   `gpgpu_sim_config::validate_supported_trace_contract`（`gpu-sim-config.cc`，删除
   `-is_SM_remodeling_enabled` early-return 后无条件运行；今天它已 fatal-reject PTX/子核关闭/IBuffer
   关闭等）承担配置级断言；**per-kernel 守卫**落在 `kernel_scheduler::add_kernel`
   （`kernel-scheduler.cc`，此处已读 `!kinfo->is_captured_from_binary` 累加
   `num_kernel_not_in_binary`）—— 在同一读点把"计数"升级为"trace 模式下遇非 captured kernel 即
   fatal error"。需配一条 negative 契约测试（§6/§7 step 5）。
   证据：`grep -rn 'is_captured_from_binary' --include=*.cc --include=*.h` → 读点 `main.cc`、
   `kernel-scheduler.cc`、`REM/sm.cc`、`REM/subcore.cc`、`REM/functional_unit.cc`；契约函数
   `grep -n 'validate_supported_trace_contract' gpu-sim-config.cc`。

5. **VERIFIED 安全事实（退役 gate-可验证的依据）。** 每个 golden fixture 归档的 kernel 在其
   checked-in `enhanced_execution_info.json` 中 `is_captured_from_binary` 均为 `true`；两个 gate 配置
   （`SM89_RTX4090`、`SM86_RTX3080`）均 `-is_remodeling_scoreboarding_enabled 0`。因此 4/4 gate case
   早已跑**纯 control-bit** 模式，删除 scoreboard 路径不改变任何 fixture 的行为，golden gate 仍是有效
   oracle —— 这正是"删除可被 gate 验证"的理由。**其反面必须点破**：gate **从不执行** scoreboard 分支
   （`-...enabled 1` 只在不入 gate 的 8 个 SC 配置里），所以 scoreboard 删除的正确性**不是**靠 gate 触达
   该路径来保证，而是靠"所有 fixture 都 captured、gate 配置都置 0"这一事实 —— 见 §8 新增风险 R11。
   证据：`for t in tests/remodeled_trace/fixtures/*.tar.gz; do tar xzOf "$t" --wildcards '*/enhanced_execution_info.json' | grep -o '"is_captured_from_binary":[a-z]*'; done` → 全 `true`；
   `grep -n is_remodeling_scoreboarding_enabled .../SM89_RTX4090/gpgpusim.config .../SM86_RTX3080/gpgpusim.config` → 均 `0`。

6. **README 契约同步（记为义务，非本次编辑）。** 退役使 README features #4（"Configurable
   dependence handling: scoreboards or control bits"）、#5（enhanced scoreboard register coverage）、
   #6（additional WAR scoreboard）成为**假命题**。退役步骤（§7 step 5）必须在同一提交内更新
   `README.md` 这三条 feature；本文件不编辑 README，仅登记此义务。
   证据：`grep -n 'scoreboard\|control bits' README.md` → 第 14–16 行三条 feature。

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
| `opndcoll_rfu_t` (operand collector, incl. nested `op_t`,`allocation_t`,`arbiter_t`,`input_port_t`,`collector_unit_t`,`dispatch_unit_t`) | register-file/operand collector | 4 refs outside `shader.{cc,h}`, all in `CORE/result_bus.{h,cc}` (`RRS::init(unsigned,unsigned,opndcoll_rfu_t*)` + `m_rf` member). That coupling is itself legacy (see TRANSFORM `result_bus`). Remodeling uses its own `Register_file`/`Register_file_cache` (`REM/register_file.h`), never `opndcoll_rfu_t`. |
| `simd_function_unit`, `pipelined_simd_unit`, `sfu`, `dp_unit`, `tensor_core`, `int_unit`, `sp_unit`, `specialized_unit` | legacy EX pipeline units | `simd_function_unit`,`pipelined_simd_unit`,`class sfu`,`class dp_unit` = 0 refs outside `shader.{cc,h}`. `sp_unit`/`int_unit`/`tensor_core`/`specialized_unit` outside-hits are all substrings of config fields (`m_config->...`, `gpgpu_num_sp_units`, `specialized_unit_params`, `tensor_core_avail`, `OP_*`) or option strings in `cuda-sim.cc`/`trace_driven.cc` — no use of the **classes**. Remodeling EX pipeline is `REM/functional_unit.{h,cc}`. |
| `ldst_unit` (legacy, `class ldst_unit : public pipelined_simd_unit`) | legacy LD/ST | 0 refs outside `shader.{cc,h}` after excluding `ldst_unit_sm` and the forward decl `ldst_unit_remake`. Remodeling LD/ST is `REM/ldst_unit_sm.{h,cc}`. |
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

**Reclassified out of DELETE (adversarial finding 3).** The scheduler *classes*
(`scheduler_unit`+subclasses) stay in the DELETE row above, but the two enum families
`scheduler_prioritization_type`/`concrete_scheduler` and `pipeline_stage_name_t` are **not**
legacy-only and moved to §1c TRANSFORM: `concrete_scheduler` is read by
`gpgpu_sim_config::init()` (`gpu-sim.h`, scheduler-string parse that stores
`m_shader_config.warp_scheduling_mode` and `assert`s the result — runs in trace mode), and
`N_PIPELINE_STAGES` sizes `pipe_widths[N_PIPELINE_STAGES]` inside the KEEP class
`shader_core_config` (`shader.h`). The earlier "0 refs in `REM/`" proof was
necessary-not-sufficient: it missed the L3-config-init and the KEEP-class array-sizing consumers.

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
| `shader_core_ctx_wrapper` (`shader_core_wrapper.h`) | 90 pure-virtual methods (`grep -cE '= 0;'` = 90; `grep -cE '^\s*virtual '` = 91 incl. dtor). Includes `remodeling/new_stats.h` for `Element_stats`. | Shrink to the surviving call-set (**≈44-method floor**, derived empirically in §2 — not a fixed target); `SM` becomes sole implementor. Keep as the formal L3↔L2 contract (roadmap §3.1). |
| `simt_core_cluster` (25 methods, `create_shader_core_ctx()=0`) | Holds `std::vector<shader_core_ctx_wrapper *> m_core`; abstract via one pure virtual factory. Only concrete subclasses = exec/trace cluster. | KEEP as L3 orchestration. After exec removal, only `trace_simt_core_cluster` remains; its factory `else` branch dies (unconditional `new SM`). Decide `m_core` element type per §2. |
| `shader_core_mem_fetch_allocator` | Defined in `shader.h`, subclass of L0 `mem_fetch_allocator`. `create_inst_memory_access` etc. | KEEP the class; only re-home out of the legacy TU. Verify no method touches a deleted legacy type. |
| `enum scheduler_prioritization_type`, `enum concrete_scheduler` (`shader.h`) + option `-gpgpu_scheduler`/member `gpgpu_scheduler_string` | Reclassified from DELETE (finding 3). `gpgpu_sim_config::init()` (`gpu-sim.h`) parses `gpgpu_scheduler_string` into a `concrete_scheduler`, stores `m_shader_config.warp_scheduling_mode = scheduler`, and `assert`s it is not `NUM_CONCRETE_SCHEDULERS` — this runs on **every trace launch**. The parse result only selects a legacy scheduler the SM path never instantiates, so it is inert-but-live. | **Recommended (minimal-risk): RETAIN** the enum constants + option + parse (inert on the SM path, but cheap and build-load-bearing); delete only the scheduler *classes* (§1a). Removing `-gpgpu_scheduler`/`gpgpu_scheduler_string`/`warp_scheduling_mode` is a **config-coupled** migration — it must delete the `init()` parse+`assert` in the same change or every run aborts — and therefore rides a golden re-approval, not a pure-code step-3 delete. |
| `enum pipeline_stage_name_t` / `N_PIPELINE_STAGES` (`shader.h`) + option `-gpgpu_pipeline_widths`/member `pipe_widths[N_PIPELINE_STAGES]` | Reclassified from DELETE (finding 3). `N_PIPELINE_STAGES` is the array size of `shader_core_config::pipe_widths[N_PIPELINE_STAGES]` and the loop bound in its parse of `pipeline_widths_string` (`shader.h`). The KEEP class `shader_core_config` cannot compile without the enum. `REM/` never reads `pipe_widths`/`N_PIPELINE_STAGES`/`ID_OC_SP` (grep empty), so the array is set-but-unused on the SM path. | **Recommended (minimal-risk): RETAIN** `pipeline_stage_name_t`/`N_PIPELINE_STAGES` as the sizing constant + keep `pipe_widths[]` + `-gpgpu_pipeline_widths` (inert). Removing them requires deleting the parse + the member together and replacing `N_PIPELINE_STAGES` with a literal — a config-coupled migration behind golden re-approval, not a step-3 delete. |
| `result_bus.h` `RRS` / `m_res_bus_improved` | `RRS::init(unsigned,unsigned,opndcoll_rfu_t*)` (`result_bus.h`), called at `shader_core_ctx` ctor `m_res_bus_improved.init(..., &m_operand_collector)` (`shader.cc`). `SM::get_loog_rrs()` **throws** `std::logic_error("LOOG is not compatible with this new accurate remodeling")` (`REM/sm.cc`). | LOOG/RRS is legacy-only on the remodeled path (the getter throws; only `shader_core_ctx` calls `RRS::init` with the operand collector). DELETE `result_bus.{h,cc}` with the operand collector, drop the `get_loog_rrs`/`get_is_loog_enabled` wrapper virtuals, **and delete the uncalled remodeled `ldst_unit_sm::get_first_key_pending_writes`** — see §9 ruling 2 and Risk R6 for the semantic-normalization framing (pending-write key ≡ `warp_id`). |

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

Union of the above ≈ **44 methods** (adversarial-review recount: 28 cluster calls + 3 inline
gather methods in `shader.h` + `create_gpu_per_sm_stats` + 2 `L0_icnt` methods + 9 surviving
trace/warp/barrier methods + `inc_simt_to_mem`) — the step-4 executor must derive the exact set
empirically from all surviving wrapper-typed callers after steps 1–3, not from this estimate.
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
  surviving call-set (≈44-method floor, empirically derived above); `SM` remains sole implementor;
  cluster/L0_icnt keep `shader_core_ctx_wrapper *`.
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
Under control-bit-only (§0.2), `shd_warp_t` **keeps** its two remodeling members
`m_dependency_state` and `m_IBuffer_remodeled` (both still `new`-ed/`delete`-d by the ctor/dtor) and
**loses** only `m_scheduler`.

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
  Move the class + de-inline the constructor, but let `SM::create_shd_warp` keep `new trace_shd_warp_t`
  and the casts. **Honest include accounting (adversarial finding 5):** de-inlining removes all three
  `remodeling/` includes *from `shader.h`*, but because `shd_warp_t`'s ctor/dtor `new`/`delete`
  `IBuffer_Remodeled` and `Dependency_State` (`shader.h` ctor body, members
  `m_IBuffer_remodeled`/`m_dependency_state`), **two of them (`remodeling/ibuffer_remodeled.h`,
  `remodeling/warp_dependency_state.h`) relocate to the new `shd_warp.cc` translation unit** — the
  reverse-include *moves*, it does not reach zero. Only the stray `remodeling/l0_icnt.h` vanishes
  outright (unused in the header). Consequence: `shader.h`'s reverse-include is genuinely removed
  (step 2), but a **new `shd_warp.cc`→`remodeling/` edge appears** and must be recorded as
  **deferred to stage 4**, alongside the surviving L2→L4 `SM::create_shd_warp` downcast. Medium
  payoff, small surface, lowest risk. The `scoreboard.cc`/`scoreboard_reads.cc` reverse-includes are
  handled independently by the scoreboard deletion (§6 / step 5, → zero); the
  `abstract_hardware_model.*` reverse-includes are independent and deferred to stage 4.
- Option 3 — **`shd_warp_t` → L2 (into `remodeling/`).** Since its only live owner is `SM` and it
  carries three remodeling members, co-locating with the SM model removes all `shader.h`→`REM`
  edges by absorption. Consequence: L1 no longer owns the warp abstraction; `barrier_set_t` and the
  memory interfaces (which reference `shd_warp_t` indirectly) would then sit above or beside it. But
  `shd_warp_t` also holds generic warp state used by L1 `barrier_set_t`/scoreboard-style facilities,
  and the roadmap explicitly lists `shd_warp_t`归属 as an L1-candidate decision — pushing it into L2
  contradicts the "L1 共享部件层" intent and would make any future non-SM consumer depend on L2.

**Recommendation: Option 2 for stage 3, converging to Option 1 in stage 4.** Rationale: the
mandatory, low-risk win this stage is *de-inlining the constructor and moving `shd_warp_t` to a
dedicated L1 header* — that removes `shader.h`'s three `remodeling/` includes and lets `shd_warp_t`
survive the deletion of `shader.h`'s legacy body. **But this move is a relocation, not a severance
(finding 5):** two of those includes reappear in the new `shd_warp.cc`, so the reverse-include
ledger gets a **new inbound entry `shd_warp.cc`→`remodeling/`** that replaces the `shader.h` entry.
The stage-3 ledger must record this honestly as **deferred to stage 4** — true severance needs an
L2-owned factory (Option 1) that hands `SM` ready-made `shd_warp_t*` so L1 never `new`s an L2 type.
Re-homing the trace-stream ownership (Option 1) is a genuine ownership redesign that touches
`SM::create_shd_warp`, `SM::func_exec_inst`, and the trace driver together; per the roadmap it
belongs with the stage-4 "warp_inst_t/shd_warp_t remodeling 成员归属重整" work. Note the distinction
between the two deferred edges: the L2→L4 `SM::create_shd_warp`→`trace_driven.h` downcast is
remodeling-**outbound** (does not grow the inbound reverse-include ledger), whereas
`shd_warp.cc`→`remodeling/` is remodeling-**inbound** and does occupy a ledger slot until stage 4.
Keep `trace_shd_warp_t` in L4.

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
  `gpgpu-sim/libcuda/cuda_runtime_api.cc`. Unreachable from `MAIN` **at runtime, but not at
  link/compile time** (adversarial finding 1): `src/Makefile` compiles and links every top-level
  `.cc`, so `gpgpusim_entrypoint.cc`'s `new exec_gpgpu_sim` is a hard link reference. Deleting
  `exec_gpgpu_sim` therefore breaks the build unless `gpgpu_ptx_sim_init_perf` is first rewritten as
  a **fatal-error stub** ("PTX execution mode removed; use the trace frontend") that retains the
  libcuda API symbol while the exec classes die. This is step 1's obligation (§7); the row below
  lists the referrer that the stub neutralizes.

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

**Same procedure, step 5, for the scoreboard option family.** The control-bit-only retirement removes
`-is_remodeling_scoreboarding_enabled`/`-scoreboard_war_max_uses_per_reg`/`-scoreboard_war_mode` and
deletes the 8 `SM86_RTXA6000_SC_*` config dirs (§0.2 item 3, §7 step 5). That is a second, independent
config-hash change on the two gate configs and rides the **identical** re-approval procedure; stats
stay byte-identical because the gate already runs `-is_remodeling_scoreboarding_enabled 0` with all
kernels captured (§0.2 item 5).

---

## 6. 停滞分支吸收（`archive/true-path-scoreboard-cleanup`）

**范围对齐（§0.2）。** 用户选择 control-bit-only，因此重新落地 `465af43` 的 scoreboard 清理不再是
"顺带吸收一个停滞分支"，而是**与本阶段退役目标一致的正式工作**：`465af43` 移除了 scoreboard 的
热路径 *调用者*，本阶段在其之上**走到终点** —— 删除 `Scoreboard`/`Scoreboard_reads` 类、
`m_scoreboard*` 成员、8 个 SC 配置、scoreboard 选项族，并新增 non-captured kernel 的启动拒绝与一条
negative 契约测试。下面先对账 `465af43` 的内容，再给出"走到终点"所需的**增量**。

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

**Critical caveat — the tag is a partial precursor; control-bit-only takes it to the finish line.**
`465af43` never touches `scoreboard.cc`/`scoreboard_reads.cc`; on the tag those files still
`#include "remodeling/sm.h"` (and `register_file.h`), and `SM` still holds
`std::shared_ptr<Scoreboard> m_scoreboard` / `m_scoreboard_WAR`. The tag removes the hot-path
**callers** of the scoreboard — a **prerequisite** for deleting the `_remodeling` scoreboard methods
and the `m_scoreboard*` members, which is the step that actually breaks the L1→L2 cycle and drives
the two scoreboard reverse-includes to zero (§7 step 5). Under control-bit-only that finish-line work
is now **in scope**, so step 5 does everything the tag left undone:

- delete the `Scoreboard`/`Scoreboard_reads` classes (`scoreboard.{cc,h}`,
  `scoreboard_reads.{cc,h}`) and the `m_scoreboard`/`m_scoreboard_WAR` members + `get_scoreboard*`
  getters on `SM`, and the dead-store `m_scoreboard`/`m_scoreboard_reads` on `ldst_unit_sm`
  (assigned in its ctor, never called);
- **semantic normalization at the two unconditional scoreboard reads** (golden-neutral because the
  scoreboard is never populated on the control-bit path — `reserveRegisters*` only runs inside the
  now-deleted branch): `SM::check_if_warp_has_finished_executing_and_can_be_reclaim` drops the
  `!m_scoreboard->pendingWrites(warp_id) && !m_scoreboard_WAR->pendingReads(warp_id)` conjuncts and
  keeps `!warp->get_dependency_state()->are_pending_dependencies() && !warp->is_atomic_pending()`;
  `SM::warp_waiting_at_mem_barrier` drops the `if (use_traditional_scoreboarding)` arm and
  unconditionally sets `clear_membar = are_all_wait_barrier_ready(warp_id)`;
- delete the scoreboard option family (`-is_remodeling_scoreboarding_enabled`,
  `-scoreboard_war_max_uses_per_reg`, `-scoreboard_war_mode`) with their `shader_core_config`
  members, remove those lines from every config that carries them, and delete the 8
  `SM86_RTXA6000_SC_*` config directories wholesale (§0.2 item 3);
- add the non-captured-kernel startup/launch rejection (§0.2 item 4) and a negative contract test;
- update `README.md` features #4/#5/#6 (§0.2 item 6).

**Stat-field caveat (do not over-delete).** `shader_core_stats` carries scoreboard-named counters
(`num_scheduler_stall_cycle_due_to_war_scoreboard`,
`num_scheduler_stall_cycle_dependencies_other_reasons_not_war_scoreboard`, `shader.h`). These stay
`0` on the control-bit path but are printed into the golden-locked stdout; removing a printed field
changes the byte-identical output. Retain any scoreboard-named stat field that appears in the golden
stat dump (or re-approve) — verify against the gate output before pruning (Risk R4/R11).

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
  re-applied against current `dev_dzw`, not a separate merge. Item (2) is the **starting point** of
  step 5's finish-line work above, not the whole of it.
- Item (3) `-is_loog_enabled` registration: **do not absorb.** Per §9 ruling 2 (finding 6) the LOOG
  surface is deleted, not registered — deleting `result_bus`/RRS, the `get_loog_rrs`/
  `get_is_loog_enabled` wrapper virtuals, the `is_loog_enabled` member and its reads, and the
  uncalled remodeled `ldst_unit_sm::get_first_key_pending_writes`. Registering the option would
  contradict the deletion.
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
gate; any red → stop and report, no scope creep. Reverse-include ledger must be **non-increasing**
at every step (the roadmap's "monotonic shrink" intent, relaxed honestly for the step-2 `shd_warp.cc`
relocation — see the end-of-stage ledger). Baseline (from `2026-07-17-dead-weight.md`, run in `SRC`):
`abstract_hardware_model.cc`, `gpu-sim.h`, `scoreboard.cc`, `scoreboard_reads.cc`, `shader.cc`,
`shader.h`, `shader_core_wrapper.h` reverse-include `remodeling/`; plus `abstract_hardware_model.h`
references `functional_unit` at 4 sites.

| Step | What moves / dies | Gate | Reverse-include delta | Golden re-approval? | Revert point |
| --- | --- | --- | --- | --- | --- |
| **1. Remove exec + make SM unconditional** (config) | Delete `exec_gpgpu_sim`,`exec_simt_core_cluster`,`exec_shader_core_ctx` (§4.1) + `exec_shader_core_ctx::sim_init_thread` in `gpu-sim.cc`; **`gpgpu_context::gpgpu_ptx_sim_init_perf` (gpgpusim_entrypoint.cc) becomes a fatal-error stub** ("PTX execution mode removed; use the trace frontend") because it is compile-linked into the .so regardless of runtime reachability (adversarial finding 1) — the libcuda API symbol survives, the exec classes die; delete `-is_SM_remodeling_enabled` registration/member/branches (§5); **also remove in the same sweep** the vestigial power options deferred from the dead-weight stage (its review response widened the sweep to `util/tuner/**` + job-launching yml `extra_params`) — a combined config-line sweep across the 39 shipped configs + `util/tuner/**` + emitters; `create_shader_core_ctx` → unconditional `new SM`; delete `trace_shader_core_ctx` + its methods; delete validation early-return. **NOT here:** `-gpgpu_scheduler`/`-gpgpu_pipeline_widths` are **retained** (§1c) — their enums are live in `gpgpu_sim_config::init()` and size `shader_core_config::pipe_widths[]`, so removing them is a config-coupled migration for a later stage, not this sweep | build + unittest + `check` 4/4 | no change yet (`shader.cc/h` still host KEEP facilities) | **YES** — one config-hash re-approval covers the `-is_SM_remodeling_enabled` + power option-line removals (§5 procedure) | commit revert |
| **2. Extract KEEP facilities to L1 headers** (pure code) | Move `shader_core_config`, `shader_core_stats(_pod)`, `shd_warp_t` (+`function_call_entry_info`,`thread_ctx_t`), `barrier_set_t`, `shader_core_mem_fetch_allocator`, `ifetch_buffer_t`, `shader_memory_interface`/`perfect_memory_interface` out of `shader.{h,cc}` into dedicated L1 headers/TUs; **de-inline `shd_warp_t` ctor** into a `.cc`; delete the stray `remodeling/l0_icnt.h` include | build + unittest + `check` 4/4 | `shader.h` drops all three `remodeling/` includes → **`shader.h` reverse-include removed**; but the ctor `new`s `IBuffer_Remodeled`/`Dependency_State`, so `remodeling/ibuffer_remodeled.h`+`warp_dependency_state.h` **relocate to the new `shd_warp.cc`** (finding 5) → **new inbound edge `shd_warp.cc`→`remodeling/`, deferred to stage 4** (not a net shrink for this pair — a relocation). Only the stray `l0_icnt.h` vanishes outright | pure code | commit revert |
| **3. Delete legacy pipeline machinery** (pure code) | Delete scheduler **classes** (`scheduler_unit`+subclasses — the `concrete_scheduler`/`scheduler_prioritization_type` **enums are retained**, §1c), `opndcoll_rfu_t`, `simd_function_unit`/`pipelined_simd_unit`+subclasses, legacy `ldst_unit`, `insn_latency_info`, `register_bank`/`coalesced_segment`/`check_kernel_launch_limitation` (§9 ruling 1: delete, no re-host), `result_bus`/RRS + the LOOG surface (§9 ruling 2, Risk R6), and the now-empty `shader_core_ctx` base. **Keep** `pipeline_stage_name_t`/`N_PIPELINE_STAGES` (sizes KEEP `shader_core_config::pipe_widths[]`, §1c) | build + unittest + `check` 4/4 | `shader.cc` reverse-include removed (its `remodeling/sm.h`,`new_stats.h` includes go when `shader_core_ctx` body is gone) → **`shader.cc` reverse-include removed** | pure code | commit revert |
| **4. Shrink `shader_core_ctx_wrapper` (Option B)** (pure code) | Remove the ~55 unused virtuals (per-op stat incrementers + legacy accessors); keep the ~44-method surface derived empirically (§2); re-type or document the `Element_stats` methods' `new_stats.h` include | build + unittest + `check` 4/4 | `shader_core_wrapper.h` becomes the formal L3↔L2 contract; its `new_stats.h` include either removed (if `Element_stats` re-typed) or **retained as the one sanctioned seam** | pure code | commit revert |
| **5. Retire the scoreboard dependency mode** (config) | Control-bit-only finish line (§0.2, §6): remove the `use_traditional_scoreboarding` blocks in `REM/sm.cc`/`subcore.cc`/`functional_unit.cc` (keep only the `dependency_state` arm); normalize the two unconditional scoreboard reads in `SM::check_if_warp_has_finished_executing_and_can_be_reclaim` + `SM::warp_waiting_at_mem_barrier` (golden-neutral, §6); delete `Scoreboard`/`Scoreboard_reads` classes + `SM::m_scoreboard`/`m_scoreboard_WAR` + getters + `ldst_unit_sm` dead-store members + its ctor scoreboard params; delete the scoreboard option family (`-is_remodeling_scoreboarding_enabled`, `-scoreboard_war_max_uses_per_reg`, `-scoreboard_war_mode`) + members, strip those lines from every config that carries them, **delete the 8 `SM86_RTXA6000_SC_*` config dirs wholesale**; add the non-captured-kernel startup/launch rejection (`validate_supported_trace_contract` + `kernel_scheduler::add_kernel`, §0.2 item 4); update `README.md` #4/#5/#6. **Retain** scoreboard-named `shader_core_stats` fields that print into the golden stdout (§6 stat-field caveat) | build + unittest + `check` 4/4 (re-approved) | `scoreboard.cc`/`scoreboard_reads.cc` drop `#include remodeling/sm.h`(+`register_file.h`) → **both scoreboard reverse-includes reach ZERO** | **YES** — the option-line strip + 8-dir deletion changes the two gate configs' `sha256`; re-approve per §5 procedure (stats byte-identical because the gate already runs pure control-bit, §0.2 item 5) | commit revert |
| **6. Add C++ primitive tests (adapted) + negative contract test** (pure code) | Absorb `dependency_path.h` + **trimmed** True-Path tests (drop the 5 `UsesTraceModeScoreboard.*` and the two-branch `MutualExclusivity` premise — that branch no longer exists); add a **negative contract test** asserting a non-captured kernel is fatally rejected (§0.2 item 4); wire a build/ctest hook; defer `pipeline_routing.h` until routing is refactored to call it | build + unittest + `check` 4/4 + `run_tests` green | none | pure code | commit revert |

**End-of-stage ledger target.** Baseline inbound set (7 files): `{abstract_hardware_model.cc`,
`gpu-sim.h`, `scoreboard.cc`, `scoreboard_reads.cc`, `shader.cc`, `shader.h`,
`shader_core_wrapper.h}`. After step 5 the inbound set is: `shader.cc` (gone, step 3),
`shader.h` (gone, step 2 — but **relocated** to a new `shd_warp.cc` entry, finding 5),
`scoreboard.cc`/`scoreboard_reads.cc` (**gone, reach ZERO**, step 5), `shader_core_wrapper.h` (now
the sanctioned contract — either zero or one documented `new_stats.h` seam, step 4),
`gpu-sim.h`/`gpu-sim.cc` (allowed L3→L2, unchanged), plus the new `shd_warp.cc` (deferred, stage 4).
Net trajectory of the inbound file count: **7 → 7 (step 2, flat: `shader.h` out, `shd_warp.cc` in) →
6 (step 3) → 6 (step 4) → 4 (step 5)**. The set is **non-increasing at every step** (step 2 is a
relocation, not a shrink — stated honestly rather than claimed as monotonic) and ends at 4:
`abstract_hardware_model.cc`, `gpu-sim.h`, `shd_warp.cc`, and `shader_core_wrapper.h` (the sanctioned
seam). `scoreboard.cc`, `scoreboard_reads.cc`, `shader.cc`, and `shader.h` legitimately reach zero.

**Explicitly deferred to stage 4 (with reason):**

- `abstract_hardware_model.cc` → `remodeling/register_file.h` and `abstract_hardware_model.h` →
  `functional_unit` (4 sites), both bound to `warp_inst_t`'s remodeling members (`m_fu_assigned`;
  `warp_inst_t::get_number_of_uses_per_operand` uses a `register_file.h` helper at
  `abstract_hardware_model.cc`). The roadmap assigns "`warp_inst_t`/`shd_warp_t` remodeling
  成员的归属重整" to stage 4; attempting the L0→L2 severance here would drag warp_inst_t
  restructuring into the high-risk retirement stage.
- **New (finding 5): `shd_warp.cc` → `remodeling/ibuffer_remodeled.h`+`warp_dependency_state.h`**,
  created by de-inlining the `shd_warp_t` ctor (step 2). This is remodeling-**inbound** and occupies
  a ledger slot until an L2-owned factory (§3 Option 1) severs it in stage 4.
- The residual L2→L4 edge `REM/sm.cc` → `trace_driven.h` (trace-stream ownership, §3 Option 1) —
  remodeling-**outbound**, does not occupy the inbound ledger, tracked separately (Risk R9).

All three are recorded as **deferred, not regressed**.

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
| **R11** | **Scoreboard-removal gate-blindness (step 5).** The gate **never executes** the scoreboard branch — `-is_remodeling_scoreboarding_enabled 1` lives only in the 8 `SM86_RTXA6000_SC_*` configs, none of which is a gate case; the 4 gate cases all set it `0` and all their kernels are `is_captured_from_binary=true` (§0.2 item 5). So a byte-identical `check` does **not** prove the deleted branch was dead — it proves the *surviving* control-bit path is unchanged. | The correctness argument is **not** "the gate exercises the removed path" but "the removed path was never taken by any gate input": verify (a) every fixture archive's `enhanced_execution_info.json` has `is_captured_from_binary=true`, (b) both gate configs set `-is_remodeling_scoreboarding_enabled 0`, and (c) on the control-bit path `reserveRegisters*` is never called, so the two normalized reads (`pendingWrites`/`pendingReads`) were already returning empty (§6). Record all three as the explicit ship gate for step 5. Optionally add an SC-derived captured-only fixture to positively cover control-bit dependency resolution. |
| **R12** | **Scoreboard-named stat fields over-deleted (step 5).** `shader_core_stats` counters like `num_scheduler_stall_cycle_due_to_war_scoreboard` are `0` on the control-bit path but printed into the golden-locked stdout; deleting a printed field breaks byte-identity. | Before pruning any scoreboard-named stat field, `grep` it against the gate stdout / golden stat dump; retain (leave the field, wired to `0`) any that print, or fold its removal into the step-5 golden re-approval. `check` 4/4 is the oracle. |

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
- Scoreboard branch (§0.2): `grep -rn 'use_traditional_scoreboarding' gpgpu-sim/remodeling/` → `sm.cc`/`subcore.cc`/`sm.h`/`subcore.h`; scoreboard reverse-includes: `grep -n '#include' gpgpu-sim/scoreboard.cc gpgpu-sim/scoreboard_reads.cc | grep remodeling`.
- Scoreboard option/config集 (§0.2 item 3): `grep -rl 'is_remodeling_scoreboarding_enabled 1' --include=gpgpusim.config <gpu-simulator>` → 8 (SC dirs); `grep -rl 'is_remodeling_scoreboarding_enabled' --include=gpgpusim.config <gpu-simulator>` → 39.
- Captured-binary safety (§0.2 item 5): `for t in tests/remodeled_trace/fixtures/*.tar.gz; do tar xzOf "$t" --wildcards '*/enhanced_execution_info.json' | grep -o '"is_captured_from_binary":[a-z]*'; done` → all `true` (run in repo root).
- Non-captured guard site: `grep -n 'is_captured_from_binary' gpgpu-sim/kernel-scheduler.cc` (kernel registration); contract fn `grep -n 'validate_supported_trace_contract' gpgpu-sim/gpu-sim-config.cc`.
- Scheduler/pipeline enum consumers (finding 3): `grep -n 'gpgpu_scheduler_string\|warp_scheduling_mode' gpgpu-sim/gpu-sim.h`; `grep -n 'pipe_widths\|N_PIPELINE_STAGES' gpgpu-sim/shader.h`.
- LOOG dead interface (finding 6): `grep -rn 'get_first_key_pending_writes\|get_is_loog_enabled\|is_loog_enabled' gpgpu-sim/remodeling/`; `grep -n 'is_loog_enabled' gpgpu-sim/gpu-sim-config.cc` (empty = unregistered).

---

## 9. 统筹者复核裁定（2026-07-17，adversarial 复核后修订）

adversarial 复核返回 no-ship（6 findings）。用户就范围拍板后，裁定如下：

0. **范围：control-bit-only（§0.2）。** scoreboard 依赖模式随 legacy shader 路径一并退役，只保留
   control-bit（`Dependency_State`）路径。这直接消解 finding 2（"scoreboard 不是 legacy 遗留物"）——
   它现在是**被有意退役的特性**，不再以"遗留残渣"措辞对待。落地含：删 scoreboard 分支与
   `Scoreboard`/`Scoreboard_reads` 类族、`m_scoreboard*` 成员；删 scoreboard 选项族与 8 个 SC 配置
   （golden 重批）；`!is_captured_from_binary` 的 kernel 改为启动/launch fatal 拒绝（契约收紧为
   captured-from-binary only）+ negative 契约测试；同步 `README.md` #4/#5/#6。集中在 §7 step 5。

1. **`check_kernel_launch_limitation`（R7、finding 略）：删除，不再宿主。** 支持路径
   （`SM::issue_block2core`）从未调用它；tested configs 的启动均在限制内（golden 中性）。删除记入
   验证记录作为有意识决定（step 3）。

2. **LOOG/`result_bus`/RRS（§1c、R6、finding 6）：整面删除，作为语义归一。** 措辞按 finding 6 更正：
   不再用"`SM::get_loog_rrs` 抛异常故接口即死"来搪塞，而是正面归一——**pending-write 的 first key
   恒等于 `warp_id`**。理由：`ldst_unit_sm::get_first_key_pending_writes`（`REM/ldst_unit_sm.cc`）
   仅在 `m_core->get_is_loog_enabled()` 为真时返回 `inst->m_cu_rrs_id`，否则返回 `inst->warp_id()`；
   而 `is_loog_enabled` **从未注册**（`grep` `gpu-sim-config.cc` 为空，indeterminate read），且这个
   remodeled 版本**无任何调用者**（tree-wide 只有 legacy `ldst_unit::get_first_key_pending_writes`
   在 `shader.cc` 被调，随退役而死）。故删除：`result_bus.{h,cc}`、wrapper 的
   `get_loog_rrs`/`get_is_loog_enabled` 虚函数、`is_loog_enabled` 成员及其全部读点、以及**未被调用的
   remodeled `ldst_unit_sm::get_first_key_pending_writes`**——归一为 key≡`warp_id`，golden-neutral。
   step 3 执行前按 R6 复验 grep。归档分支补注册 `-is_loog_enabled` 的做法**不吸收**（§6）。

3. **`Element_stats`/`new_stats.h` 接口缝（§2 Option B 尾注）：阶段三保留为唯一被记账的 L3→L2
   契约内缝。** 重新定型 `Element_stats` 属阶段四配置/统计收敛工作，不并入高风险退役阶段。

4. **finding 3（枚举分类纠正）：`concrete_scheduler`/`scheduler_prioritization_type` 与
   `pipeline_stage_name_t` 从 §1a DELETE 改判 §1c TRANSFORM。** 它们被 KEEP 路径消费
   （`gpgpu_sim_config::init()` 的 scheduler 串解析、`shader_core_config::pipe_widths[]` 的
   `N_PIPELINE_STAGES` 定长），非纯代码删除。阶段三**保留**这两族枚举/选项（inert 但 build-load-bearing），
   只删 scheduler *类*；`-gpgpu_scheduler`/`-gpgpu_pipeline_widths` 的删除属 config-coupled 迁移，
   若做须与 `init()` 解析、`pipe_widths` 定长一并处理并走 golden 重批，不在 step 1/step 3 的最小范围内。

步骤计划（§7）仍为 6 步，但 step 5 已按 control-bit-only 重写为"退役 scoreboard 依赖模式"（含配置/
选项删除、non-captured 拒绝、README 同步），step 1/2/3 的清单按 finding 1/3/5 校正；执行分支命名
`refactor/legacy-shader-retirement`。
