# Remodeling 模块架构文档（True-Path / Control-Bit）

## 0. Scope & Assumptions

本文只描述一个单一路径：`remodeling/` 在以下前提下的有效执行路径。

- `is_trace_mode=true`
- `is_captured_from_binary=true`
- `is_SM_remodeling_enabled=true`
- `is_remodeling_scoreboarding_enabled=false`（唯一例外）
- `is_ibuffer_remodeled_enabled=true`
- `is_interwarp_coalescing_enabled=true`
- `is_instruction_prefetching_enabled=true`
- `is_fp32_and_int_unified_pipeline=true`
- `is_fp32ops_allowed_in_int_pipeline=true`
- `is_dp_pipeline_shared_for_subcores=true`
- `is_rf_cache_enabled=true`
- `is_loog_enabled=true`
- `is_vpreg_enabled=true`

在该前提下，依赖控制主路径是 control-bit（stall/yield/wait barrier），不是传统 scoreboard 冲突判定路径。

实现注意：`is_fp32_and_int_unified_pipeline=true` 时不会创建独立 `INT` pipeline，本文按统一 SP 发射路径建模。

## 1. Runtime Topology

`SM` 是顶层调度体，内部由 `Subcore` 阵列和 SM 级共享结构组成。

- `Subcore`（每个）
- 8 级子流水线（在 `Subcore::cycle()` 内按逆序驱动）
- 私有 `L0I`、`L0C`
- 私有 `IBuffer_Remodeled`（每 warp）
- 私有 RF 对象：`regular RF` + `uniform RF`

- SM 级共享
- `ldst_unit_sm`（L1D/L1T/L1C/SMEM + PRT）
- `L0_icnt`（L0 与 L1I 之间互连）
- `L1I_L1_half_C_cache`
- `barrier_set_t`
- pending wait-barrier action stacks

warp 到 subcore 的映射：
- 归属 subcore：`warp_id % num_subcores`
- subcore 内本地序号：`warp_id / num_subcores`

## 2. SM Top-Level Execution Order

`SM::cycle()` 的顺序是：

1. `L0_icnt::cycle()`
2. `m_L1I_L1_half_C_cache->cycle()`
3. `m_shared_dp_unit->cycle()`（本前提下启用）
4. `m_ldst_unit_shared_of_sm->cycle()`
5. 对每个 subcore 调用 `Subcore::cycle()`
6. 消费 pending wait-barrier increments
7. 消费 pending wait-barrier decrements
8. 递减 `m_num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline`

该顺序决定了：
- 指令互连与 L1I 响应先推进，再执行共享执行资源。
- wait-barrier 的增减动作在 SM 周期尾部统一落地。

## 3. Subcore Pipeline

`Subcore::cycle()` 在活跃 warp 数大于 0 时按以下顺序执行：

1. `writeback`
2. `execute`
3. `read_rf`
4. `allocate`
5. `control_stage`
6. `issue`
7. `decode`
8. `fetch`

末尾固定推进：`m_L0C_cache->cycle()` 与 `m_L0I->cycle()`。

### 3.1 Issue

调度顺序采用 `order_greedy_then_highest_id`：先 greedy warp，再按 dynamic warp id 降序。

在本文前提下，发射就绪条件核心是：
- IBuffer 头指令有效
- stall counter 为 0
- yield ready
- wait barriers ready
- 非 `LDGDEPBAR` 挂起
- 非 programmer barrier 等待
- FU 可发射
- 结果队列可用（固定延迟指令）
- 常量操作数命中路径可继续推进

### 3.2 Control + Allocate + Read RF

- `control_stage` 在 control-bit 路径下处理新 read/write barrier 的 pending increment。
- `allocate` 负责 RF 读端口与 FU latency 槽位预留。
- `read_rf` 执行读阶段推进，并在合适时机 release read barrier。

### 3.3 Execute + Writeback

- `execute` 驱动所有 subcore FU。
- `writeback` 从固定延迟写回队列、可变延迟写回 latch、SM 共享单元返回 latch 完成退休。

## 4. Dependency Model（Control-Bit Only）

在本前提下，依赖状态由 `Dependency_State` 主导：

- `stall counter`
- `yield`
- `wait barriers`
- `pending LDGSTS counter`

每个 issue 周期开始时 `modify_warp_state()` 会调用 `Dependency_State::cycle()`，更新 stall/yield 时间行为。

### 4.1 Wait-Barrier 生命周期

1. 指令在 `control_stage` 识别到新 read/write barrier 时，SM 记录 pending increment。
2. `SM::cycle()` 尾部消费 increment，更新目标 warp 的 barrier 计数器。
3. 指令执行期间：
- read barrier 在 `functional_unit::release_read_barrier` 触发 pending decrement。
- write barrier 在 `SM::instruction_retirement` 触发 pending decrement。
4. `SM::cycle()` 尾部消费 decrement。
5. 后续 issue 通过 `is_wait_barriers_ready_entry_point` + `wait_barriers_to_check_*` 判定可发射。

### 4.2 DEPBAR / LDGDEPBAR

- `DEPBAR` 使用 control bits + 操作数（`SB`/立即数）构造 barrier 检查集合。
- `LDGDEPBAR` 受 `m_num_pending_ldgsts` 约束，直到 pending 数归零才放行。

## 5. Instruction Supply Path

### 5.1 IBuffer_Remodeled

每 warp 使用 `IBuffer_Remodeled`：
- `get_next_pc_to_fetch_request()` 预分配 fetch/decode 宽度数量的槽位。
- `decode` 填充 `IBuffer_Entry`（`valid/pc/inst`）。
- `issue` 消费队首；trace 分支跳转时触发 flush 并更新 next PC。

实现事实：当前主路径始终通过 `IBuffer_Remodeled`；`is_ibuffer_remodeled_enabled` 在该路径上未形成旧 IBuffer 回退分流。

### 5.2 L0I + L0_icnt + L1I

- `fetch` 向 `L0I` 发请求。
- `L0I` miss 经 `L0_icnt` 转发到 `L1I_L1_half_C_cache`。
- `L0_icnt` 负责 L0<->L1 的双向端口和延迟队列推进。

### 5.3 Prefetch Path

在本前提下：
- 指令预取开启（stream buffers 工作）。

说明：`ibuffer_coalescing` 不是 `is_*` 开关，本文不固定其取值。

## 6. Execution Resources

### 6.1 Functional Units（本前提下的创建结果）

`Subcore::create_pipeline()` 中实际使用的执行资源：
- `SP`
- `UNIFORM`
- `TENSOR`
- `BRANCH`
- `SFU`
- `MISC_QUEUE`
- `MISC_NO_QUEUE`
- `MEM_SUBCORE_UNIT`
- `DP_SUBCORE_UNIT`（将结果送入 SM 共享 DP 单元路径）

实现映射：
- `HALF_OP` 进入 `SP` pipeline。
- unified 开启时，`INTP_OP/PREDICATE_OP` 走 `SP` pipeline。

### 6.2 Register File

Subcore 级 RF 对象是两套：
- `regular RF`
- `uniform RF`

谓词与统一谓词通过操作数类型和统计逻辑建模，不对应独立 `Register_file` 对象实例。

RF cache 在本前提下开启，作用于 regular RF 的 cacheable 读路径。

### 6.3 Result Queues

固定延迟指令结果进入：
- regular fixed-latency RF write queue
- uniform fixed-latency RF write queue

共享 SM 结构返回使用：
- `m_EX_WB_sm_shared_units_latch`
- `m_EX_WB_sm_variable_latency_latch`

## 7. Shared Memory Pipeline（ldst_unit_sm + PRT）

### 7.1 生命周期

1. `ldst_unit_sm::issue()` 接收来自 subcore/icnt 的访存指令。
2. `PendingRequestTable::assign_entry` 绑定 `warp_inst_t`。
3. 生成并推进 `mem_access_t`，进入各子队列：L1D/L1T/L1C/SMEM/bypass/misc。
4. cache/pipeline 响应后调用 `pending_access_logic`，递减对应 PRT pending 计数。
5. `pop_entry/pop_entries` 退休完成项并释放 entry。

### 7.2 仲裁与吞吐

- `m_dispatch_subpipeline_arb_between_icnt_and_subcores`：子流水线分发仲裁。
- `m_writeback_arb_icnt_and_subcores`：写回仲裁。
- shared/texture/constant 的每周期分发上限由内部状态位限制。

### 7.3 LOOG + VPREG 对 key 的影响

- pending writes first key：`is_loog_enabled=true` 时使用 `m_cu_rrs_id`。
- pending writes second key：`is_vpreg_enabled=true` 时使用 `vpreg_virtual_out[idx]`。

### 7.4 Interwarp Coalescing

本前提下 interwarp coalescing 启用，具体弹出优先由 `interwarp_coalescing_selection_policy` 决定。

## 8. Effective Configuration Matrix（本文固定语义）

| 参数 | 固定值 | 文中语义 |
|---|---:|---|
| `is_trace_mode` | `true` | 指令来源于 trace；dependency 走 control-bit 逻辑 |
| `is_captured_from_binary` | `true` | 配合 trace，使 control-bit 路径成立 |
| `is_SM_remodeling_enabled` | `true` | 运行入口选择 `SM`（而非 legacy shader core） |
| `is_remodeling_scoreboarding_enabled` | `false` | 禁用传统 remodeling scoreboard 路径 |
| `is_ibuffer_remodeled_enabled` | `true` | 采用 `IBuffer_Remodeled` 主路径 |
| `is_interwarp_coalescing_enabled` | `true` | 启用跨 warp 合并逻辑 |
| `is_instruction_prefetching_enabled` | `true` | 启用 L0I stream buffer 预取 |
| `is_fp32_and_int_unified_pipeline` | `true` | INT/PREDICATE 与 SP 共享发射资源 |
| `is_fp32ops_allowed_in_int_pipeline` | `true` | 在 unified 模式下不引入独立 INT 资源；该项主要影响非 unified 配置 |
| `is_dp_pipeline_shared_for_subcores` | `true` | DP 走 subcore->SM shared DP 路径 |
| `is_rf_cache_enabled` | `true` | regular RF 使用 cacheable 读路径 |
| `is_loog_enabled` | `true` | pending write 首键使用 `m_cu_rrs_id` |
| `is_vpreg_enabled` | `true` | pending write 次键使用 `vpreg_virtual_out` |

## 9. Source Anchors

### SM 主循环与退休
- `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc`
- `SM::cycle`
- `SM::instruction_retirement`
- `SM::consume_pending_wait_barrier_actions`

### Subcore 流水线与调度
- `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc`
- `Subcore::cycle`
- `Subcore::issue`
- `Subcore::control_stage`
- `Subcore::is_wait_barriers_ready_entry_point`
- `Subcore::wait_barriers_to_check_generic`
- `Subcore::wait_barriers_to_check_depbar`

### 功能单元与 read barrier 释放
- `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc`
- `functional_unit::release_read_barrier`
- `functional_unit::instruction_finishing_execution`

### 指令供应与 IBuffer
- `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/ibuffer_remodeled.cc`
- `IBuffer_Remodeled::get_next_pc_to_fetch_request`
- `IBuffer_Remodeled::issued`
- `IBuffer_Remodeled::flush`
- `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/l0_icnt.cc`
- `L0_icnt::cycle`

### 共享访存单元与 PRT
- `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/ldst_unit_sm.cc`
- `ldst_unit_sm::issue`
- `ldst_unit_sm::cycle`
- `PendingRequestTable::assign_entry`
- `PendingRequestTable::pop_entries`

### 依赖状态对象
- `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/warp_dependency_state.cc`
- `Dependency_State::cycle`
- `Dependency_State::action_over_wait_barrier`
- `Dependency_State::are_wait_barriers_ready`
