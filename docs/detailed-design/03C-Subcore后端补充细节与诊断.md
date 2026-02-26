# 第 3 章（续C）：Subcore 后端补充细节与诊断

---

> **本文件是第 3 章后端补充续篇**，包含 Allocate/RF 端口窗口、FU 内部执行语义、共享管线节流与回压诊断。
> - 前文见 [03B-Subcore后端执行与写回设计.md](./03B-Subcore后端执行与写回设计.md)
> - 章节总览见 [03-Subcore流水线设计.md](./03-Subcore流水线设计.md)

---

## 3.B 后端补充细节

### 3.B.1 Allocate 的端口窗口与 slack 计算

`allocate()` 的“可分配”判断并不只是当前周期空闲，而是检查一个**时间窗口**是否容纳所有读请求：

- cacheable 路径（固定延迟）调用：
  - `Register_file::is_possible_to_read_cacheable(inst, warp_id, read_cycles)`
  - 每个 bank 检查 `is_read_available(1, 1 + latency_read_fixed_latency_inst, requested_reads * read_cycles)`
- `max_slack_due_to_double_use_of_banks` 来自 `compute_read_slack(max_num_uses)`，用于补偿多次使用同一 bank 的拥塞。

落地到端口预留时，`allocate_reads_cacheable()` 会把窗口扩展为：

```text
[1, 1 + latency_read_fixed_latency_inst + slack)
```

这意味着“同一条指令”的读端口影响会跨多个未来周期，而不仅仅是当前周期。

### 3.B.2 固定延迟 FU 的 token 语义

固定延迟 FU 同时使用两类资源 token：

1. `occupied` bitset（`reserve_latency()`）  
2. `m_dispatch_pending_reserved_cycles`（`reserve_unit()`）

两者分别约束：
- **执行完成时点是否冲突**（latency slot）
- **发射间隔是否满足 initiation interval**（dispatch token）

`allocate()` 失败时会调用 `fu->add_extra_cycle_initiation_interval()`，把冲突反馈成后续更长的发射间隔，从而在时间上“推开”后继指令。

### 3.B.3 队列型 FU 的三段生命周期

`functional_unit_with_queue::cycle()` 内部可拆成三段：

1. `dispatch_reg -> m_queue`：只要 queue 未满即可入队。  
2. `m_queue -> m_intermediate_stages[k]`：需满足 RF 非 cacheable 读端口可用。  
3. intermediate stage 推进并在末级完成：  
  - 若到达末级则 `instruction_finishing_execution()`  
  - 否则按 `m_num_cycles_per_intermediate_stage` 跳转到下一个有效 stage

对 SFU/MEM/DP 这类可变延迟路径，WAR 释放与执行完成可能分离：  
`m_num_cycles_to_wait_to_free_WAR` 倒计时归零时才执行 `release_read_barrier()`。

### 3.B.4 Subcore -> SM 共享管线节流协议

对于需要经过 SM 共享单元的指令（MEM/共享 DP）：

- Subcore 侧完成后尝试写入 `m_EX_MEM_shared_sm_reception_latch` 或 `m_EX_DP_shared_sm_reception_latch`
- 必须同时满足：
  - 目标 latch `has_free()`
  - `SM::can_send_inst_from_subcore_to_sm_shared_pipeline() == true`

一旦成功发送，SM 会设置：

```cpp
set_num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline(...)
```

后续若干周期内禁止再次发送，实现共享管线节流，防止单 subcore 独占共享后端。

### 3.B.5 Writeback 冲突与链式回压

`writeback_latch_proccess()` 冲突后果不仅是“本条退休失败”，还会向上游逐层反压：

1. 写端口不可用 -> 指令停留在 WB latch/queue  
2. 对应 FU 的 result port 无法腾空  
3. `instruction_finishing_execution()` 无法 move_out（尤其 variable-latency 路径）  
4. FU pipeline 头部变慢，最终影响 `dispatch_reg` 清空速度  
5. 反向传导至 allocate/control/issue

可观测统计：
- `total_num_times_wb_port_conflict`
- `total_num_times_wb_evaluated`
- `total_num_evals_rf_with_conflict`

### 3.B.6 周期级示例：固定延迟指令（无冲突）

以普通 fixed-latency 指令为例（`read_latency=3`）：

1. Cycle N: issue 成功，进入 `ISSUE_CONTROL_latch`  
2. Cycle N+1: control -> `CONTROL_ALLOCATE_latch`  
3. Cycle N+2: allocate 预留 RF + FU latency，写入 `pipeline_read_stage_latency_reg[2]`  
4. Cycle N+5: read_rf 头部出队，`fu->issue()` 进入 `dispatch_reg`  
5. Cycle N+6..: FU 主流水推进  
6. 完成周期: `instruction_finishing_execution()` 写入 fixed-latency result queue  
7. 下一周期 writeback: 检查 bank 写端口并退休

如果第 7 步写端口冲突，则第 6 步开始的完成路径会被持续反压，直到某周期写端口可用。

### 3.B.7 源码锚点（后端补充）

| 文件 | 函数 | 关键行为 |
|---|---|---|
| `subcore.cc` | `Subcore::allocate()` | 固定延迟读端口 + latency slot 双重预留 |
| `subcore.cc` | `Subcore::read_rf()` | 读流水推进 + `release_read_barrier()` + `fu->issue()` |
| `subcore.cc` | `Subcore::writeback_latch_proccess()` | RF bank 写端口检查与共享写回处理 |
| `functional_unit.cc` | `functional_unit::cycle()` | dispatch/pipeline/predicate 三段推进 |
| `functional_unit.cc` | `functional_unit_with_queue::cycle()` | queue + intermediate stages + WAR 延迟释放 |
| `register_file.cc` | `is_possible_to_read_cacheable()` | bank 粒度读端口窗口校验与 rf-cache 命中 |
| `register_file.cc` | `allocate_reads_cacheable()` | 读端口窗口实际分配 |
| `sm.cc` | `SM::cycle()` | pending barrier action 提交与共享发送节流计数衰减 |
