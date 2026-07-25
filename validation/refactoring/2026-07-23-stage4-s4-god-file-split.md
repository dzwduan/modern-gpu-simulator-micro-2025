# 阶段四 S4：god file 拆分与超长方法分解

对应设计：`docs/plans/2026-07-22-stage4-reorganization-design.md` §4 S4。
性质：纯结构等价变换，零行为变化，golden 必须字节一致。

基线 commit：`784a639`（`refactor: let the sm own the per-warp remodeling state`）。

产出两个 commit：

| commit | 标题 | 范围 |
| --- | --- | --- |
| `eac6adc` | `refactor: split the ldst unit god file by responsibility` | 纯文件拆分，零逻辑编辑 |
| 本记录所在 commit | `refactor: decompose the ldst cycle and subcore issue stages` | 超长方法抽取为具名阶段方法 |

## 1. 类 → 文件映射（`eac6adc`）

拆分前 `remodeling/ldst_unit_sm.{h,cc}` 一个文件对承载 6 个类型。拆分后：

| 类型 | 目标文件 |
| --- | --- |
| `l1d_queue_element` | `remodeling/access_queue.h` |
| `AccessQueue` | `remodeling/access_queue.{h,cc}` |
| `PendingRequestTableEntry` | `remodeling/pending_request_table.{h,cc}` |
| `cluster_prt_candidate` | `remodeling/pending_request_table.h` |
| `PendingRequestTable` | `remodeling/pending_request_table.{h,cc}` |
| `pop_interwarp_result` | `remodeling/interwarp_coalescing_unit.h` |
| `InterWarpCoalescingUnit` | `remodeling/interwarp_coalescing_unit.{h,cc}` |
| `ldst_unit_sm` | `remodeling/ldst_unit_sm.{h,cc}`（保留） |
| `calculate_constant_address`（自由函数） | `remodeling/ldst_unit_sm.{h,cc}`（保留） |

所有声明与方法体逐字搬移：无重命名、无签名变化、无语句重排、无顺带清理。搬移后逐段
`diff` 校验与原文件对应行区间字节一致（`access_queue.cc` 对应原 1800-1835、
`pending_request_table.cc` 对应原 1271-1360 与 1362-1798、
`interwarp_coalescing_unit.cc` 对应原 1837-2070，头文件同理），全部 OK。

`namespace remodel { ... } // namespace remodel` 在每个新文件中保留。

## 2. include 拓扑（每个新头文件自足）

按 include-what-you-use 拆分，而不是把 `ldst_unit_sm.h` 的整块 include 复制到各文件：

| 文件 | include | 前向声明 |
| --- | --- | --- |
| `access_queue.h` | `<deque>`, `<queue>` | `class mem_access_t;`、`class mem_fetch;`（头内仅出现指针） |
| `access_queue.cc` | 自身头、`<cassert>`、`../../abstract_hardware_model.h`（`delete mem_access_t` 需完整类型） | — |
| `pending_request_table.h` | `<cstdio>`, `<limits>`, `<memory>`, `<queue>`, `<vector>`, `../../abstract_hardware_model.h`（`warp_inst_t` 作 `shared_ptr` 成员需完整类型；另需 `mem_access_t`/`address_type`/`PRTSelectionPolicies`） | `class ldst_unit_sm;`（仅回指指针） |
| `pending_request_table.cc` | 自身头、`<iostream>`, `<limits>`、`../gpu-sim.h`、`ldst_unit_sm.h`、`sm.h` | — |
| `interwarp_coalescing_unit.h` | `<map>`, `<vector>`, `../../abstract_hardware_model.h`（`new_addr_type`/`memory_space_t`/`mem_access_t`/`InterWarpCoalescingSelectionPolicies`） | `class ldst_unit_sm;`（仅回指指针） |
| `interwarp_coalescing_unit.cc` | 自身头、`<algorithm>`, `<iostream>`, `<limits>`、`ldst_unit_sm.h`、`sm.h` | — |
| `ldst_unit_sm.h` | 原有 include + `access_queue.h`、`interwarp_coalescing_unit.h`、`pending_request_table.h` | 原有前向声明保留 |

**自足性验证命令**（对每个头单独编译一个只 `#include` 该头的 TU）：

```
cd simulator-remodeled && source ./gpu-simulator/setup_environment_no_git.sh
D=<abs path>/gpgpu-sim/src/gpgpu-sim/remodeling
for h in access_queue.h pending_request_table.h interwarp_coalescing_unit.h ldst_unit_sm.h; do
  printf '#include "%s/%s"\n' "$D" "$h" > /tmp/hdr_test.cc
  g++ -std=c++17 -fsyntax-only -fopenmp -I"$CUDA_INSTALL_PATH/include" /tmp/hdr_test.cc
done
```

结果：4 个头全部退出码 0（standalone OK）。

外部消费者（`sm.h`、`subcore.cc`）无需修改：`ldst_unit_sm.h` 转包三个新头，接口面不变。
构建无需改 Makefile：`remodeling/Makefile` 用 `SRCS = $(shell ls *.cc)`，父级链接 glob
`remodeling/*.o`；构建后确认 `access_queue.o`、`pending_request_table.o`、
`interwarp_coalescing_unit.o` 三个新目标文件生成，且未删除任何既有源文件（无孤儿 .o）。

## 3. 行数（前 / S4a 后 / S4b 后）

| 文件 | `784a639` | `eac6adc` | S4b 后 |
| --- | --- | --- | --- |
| `ldst_unit_sm.h` | 390 | 254 | 265 |
| `ldst_unit_sm.cc` | 2072 | 1271 | 1297 |
| `access_queue.h` | — | 61 | 61 |
| `access_queue.cc` | — | 74 | 74 |
| `pending_request_table.h` | — | 127 | 127 |
| `pending_request_table.cc` | — | 569 | 569 |
| `interwarp_coalescing_unit.h` | — | 73 | 73 |
| `interwarp_coalescing_unit.cc` | — | 275 | 275 |
| `subcore.h` | 217 | 217 | 227 |
| `subcore.cc` | 1278 | 1278 | 1304 |

S4b 的行数净增来自新增的方法签名、返回语句与声明行；被抽取的语句本身逐字保留。

## 4. S4b 抽取的阶段方法

### 4.1 `ldst_unit_sm::cycle()`

原 271 行（原 `.cc` 754-1014）降为 18 行的有序调用序列。抽取的私有阶段方法（行号为
S4b 后 `remodeling/ldst_unit_sm.cc` 中的跨度）：

| 方法 | 行跨度 | 原区间 |
| --- | --- | --- |
| `service_writeback_clients()` | 754-762 | 756-762 |
| `solve_missed_accesses_of_caches()` | 764-768 | 764-766 |
| `process_response_fifo()` | 770-827 | 768-823 |
| `dispatch_accesses_to_caches()` | 829-843 | 828-840 |
| `stage_l1d_accesses_through_tlb()` | 845-860 | 846-859 |
| `route_next_accesses_to_subpipelines()` | 862-913 | 861-910 |
| `refill_next_accesses_from_prt()` | 915-952 | 912-947 |
| `issue_incoming_memory_instructions()` | 954-1006 | 949-999 |
| `update_interwarp_coalescing_warppool_policy()` | 1008-1021 | 1001-1012 |
| `cycle()`（重写为调用序列） | 1023-1040 | — |

阶段体逐字保留（缩进层级未变，无需重排版）；已逐个 `diff` 与原区间比对，全部字节一致。

`cycle()` 中直接调用的既有方法保持原位不变：`global_shared_latency_queue_cycle()`、
`cache_cycles()`、`reset_is_this_l1d_bank_allocated_this_cycle()`、
`dispatch_access_directly_to_l2()`、`execute_miscellaneous_dispatch()`、
`shared_dispatch()`、`m_prt->management_entries_to_process()`。

控制流说明：`route_next_accesses_to_subpipelines()` 内的 `break`（原 908 行）位于其自身
`while` 内、`issue_incoming_memory_instructions()` 内的 `break`（原 990 行）位于其自身
`for` 内，均不跨越方法边界；跨阶段的局部量（`can_continue_this_bank`、`num_trials`、
`has_been_issued`、`icnt_id`）全部只在单一阶段内活跃，随阶段整体下沉，未提升为成员。

### 4.2 `Subcore::issue()`

原 129 行（原 `subcore.cc` 353-481）降为 23 行。抽取的私有方法（行号为 S4b 后
`remodeling/subcore.cc` 中的跨度）：

| 方法 | 行跨度 | 原区间 |
| --- | --- | --- |
| `has_fixed_latency_result_queue_space(fu, pI, has_dst_regs, dst_type)` | 353-369 | 405-415 |
| `update_l1c_greedy_window(subcore_warp_id, is_l1c_ready)` | 371-385 | 424-433 |
| `select_and_issue_ready_warp(shared_sm, is_valid_inst)` | 387-465 | 366-401 / 416-421 / 434-461 |
| `account_issue_stage_stats(shared_sm, 4 个标志)` | 467-483 | 466-478 |
| `issue()`（重写为阶段序列） | 485-507 | 354-365 / 462-465 / 479-480 |

跨阶段局部量以参数传递，未提升为成员：`is_valid_inst` 以引用传入
`select_and_issue_ready_warp`（唯一写点是置 `true`），`is_issued_inst` 由其返回值承接，
`is_issue_port_busy` / `is_next_stage_availabe` 留在 `issue()` 内并按值传给统计方法。
`has_dst_regs` / `dst_type` 以引用作为出参，声明顺序与原文一致。

被抽取的语句块逐字保留（仅按新嵌套深度整体减少缩进：`405-415`、`424-433` 减 6 空格，
`366-401`、`416-421`、`434-461` 减 2 空格），已逐段 `diff` 校验去缩进后完全一致。

**刻意保留在原地的部分及控制流原因**：

1. `issue()` 中 `m_num_pending_cycles_with_issue_port_busy` / `m_ISSUE_CONTROL_latch.has_free()`
   的 `if / else if / else` 链保留在 `issue()` 内：三个分支分别写入两个调用方局部量
   （`is_issue_port_busy`、`is_next_stage_availabe`），而这两个量之后被统计阶段读取；
   抽取需要两个出参换 6 行代码，得不偿失，且会把分支条件与其后果拆到两处。
2. `select_and_issue_ready_warp()` 内 79 行的 warp 选择 `for` 循环不再进一步拆分：循环体
   内含 `continue`（原 371）与两处 `break`（原 449、455），这些语句控制的是循环本身；
   把循环体拆到另一个方法会改变控制流语义，只能靠额外返回码重建，风险高于收益。
   循环体内可无风险摘出的两段（结果队列可用性、L1C 贪心窗口）已按上表抽出。

## 5. 门禁结果

构建命令：

```
cd simulator-remodeled && source ./gpu-simulator/setup_environment_no_git.sh \
  && make -j$(nproc) -C ./gpu-simulator/
```

门禁命令（**从仓库根目录执行**）：

```
python3 -m unittest discover -s tests
OMP_NUM_THREADS=1 python3 tests/remodeled_trace/run_regression.py check
```

| commit | build 退出码 | unittest 退出码 | 结果 | regression 退出码 | 结果 |
| --- | --- | --- | --- | --- | --- |
| `eac6adc`（S4a） | 0 | 0 | Ran 21 tests, OK | 0 | passed 4 / total 4，`failed` 0，`observed_not_golden` 0 |
| S4b（本记录所在 commit，门禁在提交前的工作树上执行） | 0 | 0 | Ran 21 tests, OK | 0 | passed 4 / total 4，`failed` 0，`observed_not_golden` 0 |

Golden：两次门禁 `observed_not_golden` 均为 0，即 4 条回归的观测统计与 golden 字节一致。
本步未改任何配置项，无 golden 重批准。

补充的整清重建校验（排除增量构建残留目标文件掩盖缺失 include 的可能）：

```
cd simulator-remodeled && source ./gpu-simulator/setup_environment_no_git.sh
make clean -C ./gpu-simulator/          # 退出码 0
make -j$(nproc) -C ./gpu-simulator/     # 退出码 0，构建日志中 error 计数 0
```

在整清重建产物上复跑门禁：`unittest` 退出码 0（Ran 21 tests, OK）、
`run_regression.py check` 退出码 0（passed 4 / total 4，`observed_not_golden` 0），
回归报告中 `source.commit` 为本记录所在的 S4b commit。报告的
`tracked_worktree_dirty` 为 `true`，来源是仓库中一处与本步无关的 `.gitignore` 改动
（未纳入本步任一 commit），源码树本身与该 commit 一致。

## 6. 反向依赖账本

检查命令：

```
cd simulator-remodeled
grep -rn '#include *"[^"]*remodeling/' --include=*.h --include=*.cc . \
  | grep -v "gpgpu-sim/src/gpgpu-sim/remodeling/"
```

结果（与阶段四设计 §2 的目标终态一致，保持 2 条合规下向边）：

| 文件 | include | 定性 |
| --- | --- | --- |
| `gpgpu-sim/gpu-sim.h` | `remodeling/new_stats.h`、`remodeling/fusedMemory/coalescingStats.h` | L3→L2，合规，保留 |
| `gpgpu-sim/shader_core_wrapper.h` | `remodeling/new_stats.h` | L3↔L2，认可的契约缝，保留 |

S4 在 `remodeling/` 内新增 3 个文件对，未引入任何新的入边：新头只被
`ldst_unit_sm.h` 与 `remodeling/` 内部的 `.cc` include。

## 7. commit 校验

```
git log --oneline -3
```

预期输出：本记录所在的 S4b commit、`eac6adc`（S4a）、基线 `784a639`。两个 S4 commit
均未 push，由 coordinator 评审后推送。

## 8. 未覆盖范围

- 回归覆盖 `tests/remodeled_trace` 的 4 条 workload；未覆盖仓库中其它未纳入 golden 的
  trace 与配置组合。
- 本步只做结构等价变换，未新增针对拆分后 primitive（`AccessQueue`、
  `PendingRequestTable`、`InterWarpCoalescingUnit`）的单元测试——该项属于 S6 范围。
