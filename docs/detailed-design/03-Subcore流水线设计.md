# 第 3 章：Subcore 流水线设计

---

## 3.1 模块接口概览

### Input Ports

| 端口名 | 类型 | 宽度 | 来源 | 说明 |
|---|---|---|---|---|
| `m_EX_WB_sm_shared_units_latch` | `register_set_uniptr` | 1 inst | SM 共享单元（DP/MEM）的 result_ports | 接收 SM 级共享单元完成的指令 |
| `m_EX_WB_sm_variable_latency_latch` | `register_set_uniptr` | 1 inst | SFU/MISC_QUEUE FU 的 result_port | 接收 variable latency FU 完成的指令 |
| L0I cache 响应 | 通过 `m_L0I` | - | L0_icnt | 指令 cache 命中/填充响应 |
| L0C cache 响应 | 通过 `m_L0C_cache` | - | L0_icnt | 常量 cache 响应 |

### Output Ports

| 端口名 | 类型 | 宽度 | 去向 | 说明 |
|---|---|---|---|---|
| `m_EX_MEM_shared_sm_reception_latch` | `register_set_uniptr*` | 1 inst | SM 的 `m_EX_MEM_reception_latches_per_subcore[subcore_id]` | 访存指令发往 SM 共享访存单元 |
| `m_EX_DP_shared_sm_reception_latch` | `register_set_uniptr*` | 1 inst | SM 的 `m_EX_DP_shared_sm_reception_latch` | DP 指令发往 SM 共享 DP 单元 |
| L0I miss 请求 | 通过 `m_L0I` | - | L0_icnt → L1I | 指令 cache miss 请求 |

### 内部 Pipeline Latch

| Latch 名 | 类型 | 宽度 | 连接 |
|---|---|---|---|
| `m_inst_fetch_decode_latch` | `ifetch_buffer_t` | 1 | fetch → decode |
| IBuffer entries（per warp） | `deque<IBuffer_Entry>` | `ibuffer_remodeled_size` | decode → issue |
| `m_ISSUE_CONTROL_latch` | `register_set_uniptr` | 1 | issue → control |
| `m_CONTROL_ALLOCATE_latch` | `register_set_uniptr` | 1 | control → allocate（仅固定延迟指令） |
| `m_pipeline_read_stage_latency_reg[0..N-1]` | `unique_ptr<warp_inst_t>` | N 级（普通 3 级，tensor 4-reg 6 级） | allocate → read_rf |
| `m_read_stage_aux_latch` | `register_set_uniptr` | 1 | read_rf → FU dispatch |
| `m_regular_fixed_latency_rf_write_queue` | `register_set_uniptr` | `max_size_rf_write_queue` | FU → writeback（regular RF 目标） |
| `m_uniform_fixed_latency_rf_write_queue` | `register_set_uniptr` | `max_size_rf_write_queue` | FU → writeback（uniform RF 目标） |

### 关键配置参数

| 参数 | 说明 |
|---|---|
| `ibuffer_remodeled_size` | IBuffer 深度（per warp） |
| `fetch_decode_width` | 每次 fetch/decode 的指令数 |
| `max_size_register_file_write_queue_for_fixed_latency_instructions` | 固定延迟结果队列深度 |
| `max_pops_per_cycle_register_file_write_queue_for_fixed_latency_instructions` | 每周期从结果队列弹出的最大数量 |
| `gpgpu_num_reg_banks / num_subcores` | 每 subcore 的 RF bank 数 |
| `num_regular_register_file_read_ports_per_bank` | regular RF 每 bank 读端口数 |
| `num_regular_register_file_write_ports_per_bank` | regular RF 每 bank 写端口数 |

---

## 3.2 Subcore 类结构

```
Subcore
├── Warp 管理
│   └── m_warps_of_subcore: vector<shd_warp_t*>   // 归属本 subcore 的 warp
│       映射: sm_warp_id = subcore_warp_id * num_subcores + subcore_id
│
├── 寄存器文件
│   ├── m_regular_rf: Register_file*
│   │   ├── banks: gpgpu_num_reg_banks / num_subcores
│   │   ├── read_ports/bank: num_regular_register_file_read_ports_per_bank
│   │   ├── write_ports/bank: num_regular_register_file_write_ports_per_bank
│   │   └── RF cache: 启用（is_rf_cache_enabled=true）
│   └── m_uniform_rf: Register_file*
│       ├── banks: gpgpu_num_reg_banks / num_subcores
│       ├── read_ports/bank: MAX_SRC（无限制）
│       └── write_ports/bank: MAX_DST（无限制）
│
├── 私有 Cache
│   ├── m_L0I: first_level_instruction_cache*
│   └── m_L0C_cache: read_only_cache*
│
├── Functional Units
│   ├── m_sp_pipeline: functional_unit*              // FP32 + INT/PRED（unified）
│   ├── m_uniform_pipeline: functional_unit*         // Uniform 指令
│   ├── m_tensor_pipeline: functional_unit*          // Tensor Core
│   ├── m_branch_pipeline: functional_unit*          // 分支
│   ├── m_sfu_pipeline: functional_unit_sfu*         // SFU（variable latency）
│   ├── m_miscellaneous_with_queue_pipeline: functional_unit_with_queue*  // MISC 带队列
│   ├── m_miscellaneous_no_queue_pipeline: functional_unit*              // MISC 无队列
│   ├── m_memory_unit_subcore: functional_unit_with_queue*  // 访存（→SM 共享）
│   └── m_dp_pipeline: functional_unit_with_queue*          // DP（→SM 共享）
│
└── Pipeline Latches（见 3.1 表格）
```

---

## 3.3 8 级逆序流水线详解

`Subcore::cycle()` 在活跃 warp 数 > 0 时按逆序驱动流水线：

```cpp
Subcore::cycle() {
  writeback(m_sm);      // ⑧ 写回 + 退休
  execute();            // ⑦ FU 推进
  read_rf(m_sm);        // ⑥ 寄存器读取
  allocate(m_sm);       // ⑤ RF 端口 + FU latency 预留
  control_stage(m_sm);  // ④ Barrier 设置 + latch 转移
  issue(m_sm);          // ③ Warp 调度 + 发射
  decode(m_sm);         // ② Trace 解码 + IBuffer 填充
  fetch(m_sm);          // ① L0I 访问 + IBuffer 槽位预分配

  m_L0C_cache->cycle(); // L0C cache 推进
  m_L0I->cycle();       // L0I cache 推进
}
```

逆序驱动的设计意图：后级先执行，腾出 latch 空间供前级写入，避免同一 cycle 内数据覆盖。

---

## 3.4 Fetch 阶段

该流水级负责从 L0I 指令缓存获取指令，并在 IBuffer 中预分配槽位。

在正常状态下（`m_inst_fetch_decode_latch.m_valid == false`，即 fetch-decode latch 空闲）：

* 首先检查 L0I 是否有 pending 的 cache 响应（之前 miss 的请求已被 L1I 填充），若有则处理响应并填充 latch
* 若无 pending 响应，则按 greedy 调度顺序遍历本 subcore 的所有 warp：
  * 检查该 warp 的 IBuffer 是否有空间：`can_fetch() = (m_num_entries + m_fetch_decode_width <= m_num_max_entries)`
  * 若有空间，调用 `get_next_pc_to_fetch_request()` 获取下一个 fetch PC，该调用同时在 IBuffer 尾部预分配 `fetch_decode_width` 个槽位（`m_valid=false, m_pc=base_pc+16*i, m_inst=NULL`）
  * 向 L0I 发起访问：`m_L0I->access(pc, ...)`
    * **HIT**：填充 `m_inst_fetch_decode_latch`（PC、warp_id、size），设置 `m_valid = true`，本周期 fetch 完成
    * **MISS**：请求进入 L0_icnt 队列等待 L1I 响应，本周期 fetch 未完成，latch 保持无效
    * **RESERVATION_FAIL**：MSHR 满，本周期不 fetch，latch 保持无效
  * 一旦有一个 warp 成功 fetch（HIT），即停止遍历

在 latch 被占用状态下（`m_inst_fetch_decode_latch.m_valid == true`）：

该流水级什么都不做，等待 decode 阶段消费 latch 后释放。

Stall 条件：
* `m_inst_fetch_decode_latch.m_valid == true`（decode 未消费上一次 fetch 结果）
* 所有 warp 的 IBuffer 均满（无 warp 可 fetch）
* L0I 访问返回 MISS 或 RESERVATION_FAIL（所有候选 warp 均 miss）

```
fetch(SM *shared_sm)
├── 前置条件: m_inst_fetch_decode_latch.m_valid == false
├── 检查 L0I 是否有 pending 响应并处理
├── 遍历 warp（greedy 调度顺序）:
│   ├── 检查 IBuffer 有空间: warp->get_IBuffer_remodeled()->can_fetch()
│   ├── 获取 fetch PC: warp->get_IBuffer_remodeled()->get_next_pc_to_fetch_request()
│   │   └── 预分配 fetch_decode_width 个 IBuffer 槽位（m_valid=false, m_pc=pc+16*i）
│   ├── 访问 L0I: m_L0I->access(pc, ...)
│   │   ├── HIT: 填充 m_inst_fetch_decode_latch, m_valid = true
│   │   ├── MISS: 请求发往 L0_icnt → L1I
│   │   └── RESERVATION_FAIL: 本周期不 fetch
│   └── HIT 时停止遍历
└── 输出: m_inst_fetch_decode_latch（PC, warp_id, size）
```

---

## 3.5 Decode 阶段

该流水级负责从 trace 获取指令并解码，将结果填入 IBuffer 中 fetch 阶段预分配的槽位。

在正常状态下（`m_inst_fetch_decode_latch.m_valid == true`）：

* 从 latch 获取 PC 和 warp_id
* 在目标 warp 的 IBuffer deque 中查找匹配的 entry（PC 匹配且 `m_valid == false`）
* 对每个匹配 entry 执行 `single_decode()`：
  * 从 trace 获取指令：`m_trace_warp->get_next_trace_inst(pc)`
  * 设置 warp ID 到指令中
  * 生成常量 cache 访问请求（如果指令有常量操作数）
  * 递增 warp 的 in-pipeline 指令计数：`warp->inc_inst_in_pipeline()`
  * 根据指令类型和 trace 信息生成执行 latency
  * 设置 `ibuffer_entry.m_valid = true`，`ibuffer_entry.m_inst = pI`
  * 若 interwarp coalescing 启用，记录该指令的依赖信息供后续合并决策使用
* 处理完所有匹配 entry 后，清除 `m_inst_fetch_decode_latch.m_valid = false`，释放 latch 供 fetch 阶段使用

在 latch 无效状态下（`m_inst_fetch_decode_latch.m_valid == false`）：

该流水级什么都不做，等待 fetch 阶段填充 latch。

Stall 条件：
* `m_inst_fetch_decode_latch.m_valid == false`（fetch 未产出新指令）

```
decode(SM *shared_sm)
├── 前置条件: m_inst_fetch_decode_latch.m_valid == true
├── 从 latch 获取 PC 和 warp_id
├── 在目标 warp 的 IBuffer 中找到匹配的 entry（PC 匹配且 m_valid==false）
├── 对每个匹配 entry:
│   ├── 从 trace 获取指令: m_trace_warp->get_next_trace_inst(pc)
│   ├── single_decode():
│   │   ├── 设置 warp ID
│   │   ├── 生成常量 cache 访问
│   │   ├── warp->inc_inst_in_pipeline()
│   │   ├── 根据指令类型生成 latency
│   │   ├── ibuffer_entry.m_valid = true
│   │   └── ibuffer_entry.m_inst = pI
│   └── 如果 interwarp coalescing 启用，记录依赖信息
├── 清除 m_inst_fetch_decode_latch.m_valid
└── 输出: IBuffer entry（m_valid=true, m_inst 指向解码后的指令）
```

---

## 3.6 Issue 阶段

该流水级负责从各 warp 的 IBuffer 中选择一条就绪指令发射到下级流水线。这是 Subcore 流水线中逻辑最复杂的阶段，涉及 warp 调度、依赖检查、资源可用性检查。

在正常状态下（`m_ISSUE_CONTROL_latch.has_free()` 且 `m_num_pending_cycles_with_issue_port_busy == 0`）：

* 首先调用 `modify_warp_state()`，对本 subcore 的每个 warp 调用 `Dependency_State::cycle()`，递减 stall counter 和 yield
* 按 `order_greedy_then_highest_id` 顺序遍历 warp：先尝试 greedy warp（上次成功发射的 warp），再按 dynamic warp id 降序遍历其余 warp
* 对每个候选 warp，执行以下检查：
  * **IBuffer 就绪**：`warp->get_IBuffer_remodeled()->is_next_valid()` — 队首 entry 必须已解码
  * **依赖就绪**（True-Path，`use_traditional_scoreboarding = false`）：
    * `dependency_state->is_stall_counter_0()` — stall counter 必须为 0
    * `dependency_state->is_yield_ready()` — yield 必须就绪
    * `is_wait_barriers_ready_entry_point(pI, subcore_warp_id)` — 指令 control bits 中指定的所有 wait barrier 必须满足
    * `!is_waiting_ldgdepbar(pI, subcore_warp_id)` — 若为 LDGDEPBAR 指令，pending LDGSTS 计数必须为 0
    * `!warp->waiting()` — 不在 programmer barrier 等待中
  * **资源就绪**：
    * `fu->can_issue(pI)` — 目标 FU 可接受新指令（initiation interval 满足）
    * `are_l1c_operands_ready(shared_sm, pI)` — 常量操作数已就绪
    * 结果队列有空间（仅固定延迟指令）：对应的 `rf_write_queue.has_free()`
* 若所有条件满足，执行 `issue_warp()`：
  * `pI->set_fu_assigned(fu)` — 记录分配的 FU
  * `SM::issue_warp()` — 将指令移入 `m_ISSUE_CONTROL_latch`，调用 `IBuffer::issued()` 弹出队首，执行功能模拟 `func_exec_inst()`，设置 stall counter 和 yield
  * 预留结果队列槽位（固定延迟指令）
  * `fu->reserve_unit()` — 预留 FU
* 发射成功后更新 greedy pointer 指向当前 warp
* 每周期最多发射 1 条指令

在 issue port 繁忙状态下（`m_num_pending_cycles_with_issue_port_busy > 0`）：

该流水级仅执行 `modify_warp_state()` 更新依赖状态，不尝试发射。issue port 繁忙由 SM 共享单元的调度间隔控制。

在下级 latch 被占用状态下（`!m_ISSUE_CONTROL_latch.has_free()`）：

该流水级仅执行 `modify_warp_state()`，不尝试发射。等待 control 阶段消费 latch。

Stall 条件：
* `m_ISSUE_CONTROL_latch` 被占用（control 阶段未消费）
* `m_num_pending_cycles_with_issue_port_busy > 0`（SM 共享单元调度间隔）
* 所有 warp 均不满足就绪条件（依赖未解除、FU 不可用、结果队列满等）

### 调度顺序

采用 `order_greedy_then_highest_id`：
1. 先调度 greedy warp（上次成功发射的 warp）
2. 再按 dynamic warp id 降序遍历

### 就绪条件检查（True-Path）

```
issue(SM *shared_sm)
├── modify_warp_state(): 对每个 warp 调用 Dependency_State::cycle()
├── 检查 issue port: m_num_pending_cycles_with_issue_port_busy == 0
├── 检查下级 latch: m_ISSUE_CONTROL_latch.has_free()
├── 遍历 warp（greedy 调度顺序）:
│   ├── IBuffer 头指令有效: warp->get_IBuffer_remodeled()->is_next_valid()
│   ├── 获取指令: pI = warp->get_IBuffer_remodeled()->next_inst()
│   │
│   ├── [True-Path 就绪条件]
│   │   ├── use_traditional_scoreboarding = false
│   │   ├── stall counter == 0: dependency_state->is_stall_counter_0()
│   │   ├── yield ready: dependency_state->is_yield_ready()
│   │   ├── wait barriers ready: is_wait_barriers_ready_entry_point(pI, subcore_warp_id)
│   │   ├── 非 LDGDEPBAR 等待: !is_waiting_ldgdepbar(pI, subcore_warp_id)
│   │   └── 非 programmer barrier 等待: !warp->waiting()
│   │
│   ├── [资源就绪条件]
│   │   ├── FU 可发射: fu->can_issue(pI)
│   │   ├── L1C 操作数就绪: are_l1c_operands_ready(shared_sm, pI)
│   │   └── 结果队列有空间（固定延迟指令）:
│   │       ├── regular: m_regular_fixed_latency_rf_write_queue.has_free()
│   │       └── uniform: m_uniform_fixed_latency_rf_write_queue.has_free()
│   │
│   └── 全部满足 → issue_warp():
│       ├── pI->set_fu_assigned(fu)
│       ├── SM::issue_warp(): 移入 latch + IBuffer::issued() + func_exec_inst()
│       ├── 预留结果队列槽位（固定延迟指令）
│       └── fu->reserve_unit(dispatch_latch)
└── 更新 greedy pointer
```

---

## 3.7 Control 阶段

该流水级负责处理 wait barrier 的设置，并根据指令类型将指令分流到不同的下级路径。这是固定延迟指令和可变延迟指令的分流点。

在正常状态下（`m_ISSUE_CONTROL_latch.has_ready()`）：

* 获取指令和已分配的 FU
* 判断指令类型：`fu->is_fixed_latency_unit()`
* **Barrier 设置**（True-Path）：
  * 若指令 control bits 中标记了 `new_read_barrier`，调用 `SM::add_pending_wait_barrier_increment(inst, READ_WAIT_BARRIER, barrier_id)`，将 read barrier increment 推入 SM 的 pending stack
  * 若指令 control bits 中标记了 `new_write_barrier`，调用 `SM::add_pending_wait_barrier_increment(inst, WRITE_WAIT_BARRIER, barrier_id)`，将 write barrier increment 推入 SM 的 pending stack
  * 设置 `inst->m_has_perform_control_stage = true`
* **固定延迟指令**（SP/BRANCH/TENSOR/UNIFORM/MISC_NO_QUEUE）：
  * 检查 `m_CONTROL_ALLOCATE_latch.has_free()`
  * 若空闲，将指令从 `m_ISSUE_CONTROL_latch` 移动到 `m_CONTROL_ALLOCATE_latch`
  * 若被占用，指令停留在 `m_ISSUE_CONTROL_latch`，下周期重试
* **可变延迟指令**（SFU/MISC_QUEUE/MEM/DP）：
  * 检查 `fu->can_issue(inst)`（FU 内部队列未满）
  * 若可接受，直接调用 `fu->issue(m_ISSUE_CONTROL_latch)`，指令进入 FU 内部队列，跳过 allocate 和 read_rf 阶段
  * 若 FU 队列满，指令停留在 `m_ISSUE_CONTROL_latch`，下周期重试

在 latch 无指令状态下（`!m_ISSUE_CONTROL_latch.has_ready()`）：

该流水级什么都不做。

Stall 条件：
* 固定延迟指令：`m_CONTROL_ALLOCATE_latch` 被占用
* 可变延迟指令：FU 内部队列满（`!fu->can_issue(inst)`）

```
control_stage(SM *shared_sm)
├── 前置条件: m_ISSUE_CONTROL_latch.has_ready()
├── 获取指令和 FU
├── 判断固定/可变延迟: fu->is_fixed_latency_unit()
│
├── [True-Path] Barrier 设置:
│   ├── if new_read_barrier:
│   │   └── SM::add_pending_wait_barrier_increment(inst, READ_WAIT_BARRIER, barrier_id)
│   └── if new_write_barrier:
│       └── SM::add_pending_wait_barrier_increment(inst, WRITE_WAIT_BARRIER, barrier_id)
│   └── inst->m_has_perform_control_stage = true
│
├── 固定延迟指令:
│   ├── 检查: m_CONTROL_ALLOCATE_latch.has_free()
│   └── 移动: ISSUE_CONTROL_latch → CONTROL_ALLOCATE_latch
│
└── 可变延迟指令:
    ├── 检查: fu->can_issue(inst)
    └── 直接发射: fu->issue(m_ISSUE_CONTROL_latch)
        （跳过 allocate/read_rf，直接进入 FU）
```

关键分流点：
- **固定延迟指令**（SP/INT/BRANCH/TENSOR/UNIFORM/MISC_NO_QUEUE）→ `m_CONTROL_ALLOCATE_latch` → allocate → read_rf → FU
- **可变延迟指令**（SFU/MISC_QUEUE/MEM/DP）→ 直接 `fu->issue()` 进入 FU 内部队列

---

## 3.8 Allocate 阶段（仅固定延迟指令）

该流水级负责为固定延迟指令预留寄存器文件读端口和 FU 执行槽位。可变延迟指令不经过此阶段。

在正常状态下（`m_CONTROL_ALLOCATE_latch.has_ready()`）：

* 获取指令和 FU
* 确定读流水线延迟：
  * Tensor Core 4-reg-per-operand 指令：`MAXIMUM_LATENCY_READ_FIXED_LATENCY_INST = 6` 周期
  * 其他固定延迟指令：`NO_TENSOR_OP_4REG_PER_OP_LATENCY_READ_FIXED_LATENCY_INST = 3` 周期
* 检查读流水线入口是否空闲：`m_pipeline_read_stage_latency_reg[read_latency - 1]->empty()`
* 检查 RF 读端口可用性：
  * `m_regular_rf->is_possible_to_read_cacheable(inst, warp_id, read_cycles)` — 检查 regular RF 各 bank 的读端口，RF cache 命中的操作数不消耗读端口
  * `m_uniform_rf->is_possible_to_read_cacheable(inst, warp_id, read_cycles)` — uniform RF 读端口无限制，始终返回 true
  * `rf_requests.is_possible_to_read()` — 所有 RF 类型的读端口均可用
* 计算目标 FU latency：`target_latency = read_latency + inst->latency + inst->initiation_interval`
* 检查 FU latency 槽位：`fu->is_latency_available(target_latency)` — 使用 `occupied` bitset 检查
* 若所有条件满足：
  * `allocate_reads()` — 预留 RF 读端口，更新 RF cache（miss 时分配新 entry）
  * `fu->reserve_latency(target_latency)` — 在 `occupied` bitset 中标记目标槽位
  * 将指令从 `m_CONTROL_ALLOCATE_latch` 移入 `m_pipeline_read_stage_latency_reg[read_latency - 1]`
* 若任一条件不满足，指令停留在 `m_CONTROL_ALLOCATE_latch`，下周期重试

在 latch 无指令状态下：

该流水级什么都不做。

Stall 条件：
* 读流水线入口被占用
* RF 读端口不足（regular RF bank 冲突）
* FU 目标 latency 槽位已被占用

```
allocate(SM *shared_sm)
├── 前置条件: m_CONTROL_ALLOCATE_latch.has_ready()
├── 获取指令和 FU
├── 确定读延迟:
│   ├── Tensor 4-reg: MAXIMUM_LATENCY_READ_FIXED_LATENCY_INST = 6
│   └── 其他: NO_TENSOR_OP_4REG_PER_OP_LATENCY_READ_FIXED_LATENCY_INST = 3
│
├── 检查读流水线入口: m_pipeline_read_stage_latency_reg[read_latency - 1]->empty()
│
├── 检查 RF 读端口:
│   ├── m_regular_rf->is_possible_to_read_cacheable(inst, warp_id, read_cycles)
│   ├── m_uniform_rf->is_possible_to_read_cacheable(inst, warp_id, read_cycles)
│   └── rf_requests.is_possible_to_read()
│
├── 计算目标 FU latency: read_latency + inst->latency + inst->initiation_interval
├── 检查 FU latency 槽位: fu->is_latency_available(target_latency)
│
└── 全部满足:
    ├── allocate_reads(): 预留 RF 读端口 + RF cache 分配
    ├── fu->reserve_latency(target_latency): 预留 FU 执行槽位
    └── 移动: CONTROL_ALLOCATE_latch → pipeline_read_stage_latency_reg[read_latency - 1]
```

---

## 3.9 Read_RF 阶段

该流水级负责完成寄存器文件读取，将指令从读流水线头部送入 FU 的 dispatch register。同时负责推进多级读流水线。

在正常状态下（`!m_pipeline_read_stage_latency_reg[0]->empty()`，即读流水线头部有指令）：

* 获取指令已分配的 FU
* 调用 `fu->release_read_barrier(pipe_reg)`：
  * 对于固定延迟 FU：该函数内部检查 `!is_fixed_latency_unit()`，固定延迟 FU 不产生 read barrier decrement
  * 对于可变延迟 FU（理论上不会走到这里，因为可变延迟指令在 control 阶段直接进入 FU）
* 将指令移入辅助 latch：`m_read_stage_aux_latch.move_in(pipe_reg)`
* 发射到 FU：`fu->issue(m_read_stage_aux_latch)` — 指令进入 FU 的 `m_dispatch_reg`
* 推进读流水线：`pipeline_read_stage_latency_reg[i]` 的内容移动到 `pipeline_read_stage_latency_reg[i-1]`，逐级前移
* 推进 RF 状态：`m_regular_rf->cycle()` 和 `m_uniform_rf->cycle()` — 推进 bank 端口预留状态的时间窗口

在读流水线头部为空时：

仅执行读流水线推进和 RF cycle，不发射指令。

Stall 条件：
* 该阶段本身不产生 stall（读流水线头部有指令时必定能发射到 FU，因为 FU latency 已在 allocate 阶段预留）

```
read_rf(SM *shared_sm)
├── 检查读流水线头部: !m_pipeline_read_stage_latency_reg[0]->empty()
├── 获取 FU
├── 释放 read barrier: fu->release_read_barrier(pipe_reg)
│   └── 注意: 固定延迟 FU 的 release_read_barrier 不产生 barrier decrement
│             仅 variable latency FU（m_can_set_wait_barriers=true 且非固定延迟）才释放
├── 移动到辅助 latch: m_read_stage_aux_latch.move_in(pipe_reg)
├── 发射到 FU: fu->issue(m_read_stage_aux_latch)
│
├── 推进读流水线:
│   └── pipeline_read_stage_latency_reg[i] → pipeline_read_stage_latency_reg[i-1]
│
└── 推进 RF: m_regular_rf->cycle(), m_uniform_rf->cycle()
```

---

## 3.10 Execute 阶段

该流水级负责驱动所有 FU 的内部流水线推进。每个 FU 独立执行 `cycle()`，互不干扰。

在每个周期：

* 遍历 `m_all_subcore_ex_pipelines` 中的所有 FU，对每个 FU 调用 `fu->cycle()`
* 每个 FU 的 `cycle()` 行为（基类 `functional_unit::cycle()`）：
  1. 递减 `m_dispatch_pending_reserved_cycles`（dispatch 延迟计数器）
  2. 检查 predicate 流水线头部（`m_pipeline_extra_predicate_stages_reg[0]`）：若非空，调用 `instruction_finishing_execution()` 完成指令
  3. 推进 predicate 流水线：各级前移
  4. 检查主执行流水线头部（`m_pipeline_reg[0]`）：
     * 若指令有额外 predicate latency：移入 predicate 流水线尾部
     * 否则：调用 `instruction_finishing_execution()` 完成指令
  5. 推进主执行流水线：各级前移
  6. Dispatch 新指令（从 `m_dispatch_reg`）：
     * 检查 `m_dispatch_reg` 非空且 `!dispatch_delay()`
     * 计算起始 stage：`start_stage = latency - 1`
     * 若 `m_pipeline_reg[start_stage]` 为空：移入该 stage
     * 对可变延迟 FU：此时调用 `release_read_barrier()` 产生 read barrier decrement
     * `m_active_insts_in_pipeline++`

* 对于带队列的 FU（`functional_unit_with_queue::cycle()`），额外执行：
  1. 推进中间级（intermediate stages）：递减 `remaining_cycles`，完成时移入 result port
  2. 从队列取指令到中间级尾部
  3. 从 `m_dispatch_reg` 入队（若队列未满）
  4. 对于 SM 共享 FU（MEM/DP）：中间级完成时检查 SM 调度间隔，满足后移入 SM reception latch

`instruction_finishing_execution()` 行为：
* **固定延迟指令**（有目标寄存器）：将结果移入 `m_rf_write_queue`（regular 或 uniform），标记 `retired = true`
* **可变延迟指令**（有目标寄存器）：将结果移入 `m_result_port`（variable latency latch）
* **无目标寄存器的指令**：直接标记完成
* 递减 `m_active_insts_in_pipeline`

Stall 条件：
* Dispatch 阶段：`m_pipeline_reg[start_stage]` 被占用（指令堆积在 dispatch_reg）
* 带队列 FU：队列满时新指令无法入队
* SM 共享 FU：SM 调度间隔未满足时中间级完成的指令无法移入 reception latch

```
execute()
└── for each FU in m_all_subcore_ex_pipelines:
    └── fu->cycle()
        ├── 递减 dispatch pending cycles
        ├── 检查 predicate 流水线头部 → instruction_finishing_execution()
        ├── 推进 predicate 流水线
        ├── 检查主流水线头部:
        │   ├── 有 predicate latency → 移入 predicate 流水线
        │   └── 无 → instruction_finishing_execution()
        ├── 推进主执行流水线
        └── Dispatch 新指令:
            ├── 检查 dispatch_reg 非空且无 dispatch delay
            ├── 计算起始 stage: latency - 1
            ├── 目标 stage 为空 → 移入 pipeline_reg[start_stage]
            └── m_active_insts_in_pipeline++
```

`instruction_finishing_execution()` 对固定延迟指令：
- 将结果移入 `m_rf_write_queue`（regular 或 uniform）
- 标记 `retired = true`

---

## 3.11 Writeback 阶段

该流水级负责将执行完成的指令写回寄存器文件并触发退休。需要检查 RF 写端口可用性，不可用时指令停留在 latch 等待。

在每个周期，按以下顺序处理三个来源：

**1. 固定延迟写回队列**（优先级最高）：
* 处理 `m_regular_fixed_latency_rf_write_queue`：每周期最多弹出 `max_pops_per_cycle` 条指令
* 处理 `m_uniform_fixed_latency_rf_write_queue`：同上
* 对每条弹出的指令调用 `writeback_latch_proccess()`
* 若成功退休，释放结果队列槽位

**2. Variable latency latch**：
* 处理 `m_EX_WB_sm_variable_latency_latch`（SFU/MISC_QUEUE 完成的指令）
* 调用 `writeback_latch_proccess(latch, is_from_shared=false)`

**3. SM 共享单元返回 latch**：
* 处理 `m_EX_WB_sm_shared_units_latch`（DP/MEM 从 SM 共享单元返回的指令）
* 调用 `writeback_latch_proccess(latch, is_from_shared=true)`

`writeback_latch_proccess()` 的行为：
* 获取 latch 中的就绪指令
* 若指令有目标寄存器，检查每个目标寄存器对应 RF bank 的写端口：
  * `rf->is_rf_bank_write_port_available_this_cycle(bank_id)`
  * bank 由 `reg_id % num_banks` 计算
  * 需要区分目标寄存器类型（REG → regular RF，UREG → uniform RF）
* 若所有写端口可用：
  * 分配写端口：`rf->allocate_rf_bank_write_port_this_cycle(bank_id)`
  * 调用 `SM::instruction_retirement(inst)` 触发退休（write barrier decrement、LDGSTS 计数递减、warp 计数更新等）
  * 清除指令
* 若任一写端口不可用：
  * 指令停留在 latch，下周期重试
  * 这会反压上游（FU 无法将新结果写入 latch）

Stall 条件：
* RF 写端口冲突（多条指令同时写同一 bank）
* 固定延迟写回队列满（反压 FU 的 `instruction_finishing_execution()`）

```
writeback(SM *shared_sm)
├── 1. 处理固定延迟写回队列:
│   ├── writeback_process_fixed_latency_write_queue(m_regular_fixed_latency_rf_write_queue)
│   │   └── 每周期最多弹出 max_pops_per_cycle 条
│   └── writeback_process_fixed_latency_write_queue(m_uniform_fixed_latency_rf_write_queue)
│
├── 2. 处理 variable latency latch:
│   └── writeback_latch_proccess(m_EX_WB_sm_variable_latency_latch, is_from_shared=false)
│
└── 3. 处理 SM 共享单元返回 latch:
    └── writeback_latch_proccess(m_EX_WB_sm_shared_units_latch, is_from_shared=true)

writeback_latch_proccess():
├── 获取就绪指令
├── 检查目标 RF 写端口可用:
│   └── 对每个目标寄存器: rf->is_rf_bank_write_port_available_this_cycle(bank_id)
├── 全部可用:
│   ├── 分配写端口: rf->allocate_rf_bank_write_port_this_cycle(bank_id)
│   └── SM::instruction_retirement(inst)
└── 不可用: 指令停留在 latch，下周期重试
```

---

## 源码锚点

| 文件 | 函数 | 说明 |
|---|---|---|
| `subcore.cc` | `Subcore::cycle()` | 8 级逆序流水线入口 |
| `subcore.cc` | `Subcore::fetch()` | L0I 访问 + IBuffer 预分配 |
| `subcore.cc` | `Subcore::decode()` | Trace 解码 + IBuffer 填充 |
| `subcore.cc` | `Subcore::issue()` | Warp 调度 + 就绪判定 |
| `subcore.cc` | `Subcore::control_stage()` | Barrier 设置 + latch 分流 |
| `subcore.cc` | `Subcore::allocate()` | RF 端口 + FU latency 预留 |
| `subcore.cc` | `Subcore::read_rf()` | RF 读取 + read barrier 释放 |
| `subcore.cc` | `Subcore::execute()` | FU cycle 驱动 |
| `subcore.cc` | `Subcore::writeback()` | RF 写回 + 退休 |
| `subcore.cc` | `Subcore::issue_warp()` | 发射执行 |
| `subcore.cc` | `Subcore::is_wait_barriers_ready_entry_point()` | Barrier 就绪检查入口 |
| `subcore.cc` | `Subcore::create_pipeline()` | FU 创建与配置 |
