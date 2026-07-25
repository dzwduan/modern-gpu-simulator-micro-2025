# 阶段四设计：退役后重组（2026-07-22）

## 1. 背景与前置

阶段三已把 legacy shader 路径退役并合回 dev_dzw（合并提交 98c3cb7），反向依赖账本
7→4。阶段四是 roadmap 的最后阶段——**纯结构等价变换**，不改仿真行为、不删能力，只
做组织性重组。风险低于阶段三，但体量大，逐 commit 回归把关。

**全局红线**：每个 commit 全量门禁（build + `python3 -m unittest discover -s tests` +
`run_regression.py check` 4/4 byte-identical）。golden 任何漂移即停——阶段四不应有任何
配置或行为变化（若某项重组需要配置变化，单列并走受控 golden 重批准，但预期为零）。

## 2. 剩余 4 条反向依赖的精确定性（阶段四侦察结论）

细分后，4 条里只有 **2 条是真正的层级违规**（低层 include 高层），另 2 条合规：

| 文件 | include | 方向 | 定性 |
| --- | --- | --- | --- |
| `abstract_hardware_model.{cc,h}` | `register_file.h` + `functional_unit`（4 处，`warp_inst_t::m_fu_assigned`） | L0→L2 | **真违规**，清零目标 |
| `shader.cc` | `ibuffer_remodeled.h` + `warp_dependency_state.h`（`shd_warp_t` 构造/析构） | L1→L2 | **真违规**，清零目标 |
| `gpu-sim.h` | `new_stats.h` + `coalescingStats.h` | L3→L2 | 合规（高依低），保留 |
| `shader_core_wrapper.h` | `new_stats.h` | L3↔L2 | 认可的契约缝，保留或经 Element_stats 重定型消除 |

"账本清零"的准确目标 = 消除 `abstract_hardware_model` 与 `shader.cc` 两条 L0/L1→L2 违规。

## 3. 目标与非目标

**目标**：remodeling 收敛为独立命名空间；延迟配置单一权威源；god file 按类/职责拆分为
可读文件；`warp_inst_t`/`shd_warp_t` 的 remodeling 成员归属重整以消除两条真违规；每个
primitive 补最小单测；命名/拼写/魔数清理。

**非目标**：不改时序算法与统计语义；不动 golden；不改保留的上游代码格式；gpu-sim.h /
shader_core_wrapper.h 的合规下向 include 不强行消除。

## 4. 有序步骤（各步 golden byte-identical，独立 commit，gpt-5.6-sol 评审）

低风险先行、高耦合的成员归属重整压后：

**S1 — namespace 包裹 remodeling**（最低风险，纯符号域）
给 `remodeling/` 全部类与自由函数包 `namespace remodel { }`；全局自由函数
（`num_bytes_cache_req`、`get_pc_of_request`、`translate_reg_to_global_id`、
`check_is_reserved_regs_remodeling`、`calculate_constant_address` 等）收编；外部消费者
（shader.cc/abstract_hardware_model/gpu-sim/trace-driven）加 `remodel::` 限定或 using。
纯符号变换，build 驱动收敛。

**S2 — 头文件卫生**：头内全局数组（`subcore_dispatch_latch_decode[]`）与应在 `.cc` 的
方法体移出头；重复 `#define STRSIZE` 归一；`sm.h` 延迟常量 `#define`→`constexpr`；
`new_stats.h` 未用的 `<typeindex>/<variant>` 清理。

**S3 — 延迟配置收敛**：三套并行延迟命名空间（`-ptx_opcode_latency_*` /
`-trace_opcode_latency_initiation_*` / remodeling 独立选项）收敛为单一权威源；旧命名空间
选项删除，`configs/` 自带配置同步迁移（此步可能触发 golden 配置哈希重批准——若行为不变则
仅哈希变化，走受控流程）。

**S4 — god file 拆分**：
- `ldst_unit_sm.cc`（2068 行，6 类）按类拆为 `access_queue` / `pending_request_table` /
  `interwarp_coalescing_unit` / `ldst_unit_sm` 各自文件。
- `ldst_unit_sm::cycle`（约 271 行）与 `Subcore::issue` 分解为具名阶段方法。
- 拆分为纯移动 + 声明调整，build + golden 把关。

**S5 — warp_inst_t / shd_warp_t 成员归属重整**（本阶段最高耦合，消除 2 条真违规）：
- `warp_inst_t::m_fu_assigned`（`functional_unit*`）：抽为按指令 id 的旁表或
  `RemodelInstExt` 指针，使 `abstract_hardware_model.h` 不再 name `functional_unit`；
  `get_number_of_uses_per_operand` 的 `register_file.h` 依赖内联或移至 L1，消除
  `abstract_hardware_model.cc` 的 include。→ L0→L2 违规清零。
- `shd_warp_t` 的 `IBuffer_Remodeled`/`Dependency_State` 所有权：设计 §3 Option 1——由
  L2 持有的工厂/接口创建，或把 trace-stream 所有权下沉，使 `shader.cc` 不再 include
  remodeling 头。→ L1→L2 违规清零。
- 同时处理 §3 记账的 L2→L4 出边（`remodeling/sm.cc` include `trace_driven.h` +
  `new trace_shd_warp_t`）。

**S6 — primitive 单测**：为拆分出的独立部件（functional_unit、register_file、
IBuffer_Remodeled、PendingRequestTable、InterWarpCoalescingUnit、stream_buffer）补最小
单元测试；吸收停滞分支 `pipeline_routing.h` 思路的正版——先把生产路由重构为调用具名函数
（golden 中性），再对该函数补测；wire GoogleTest/ctest 钩子。

**S7 — 命名/拼写/魔数**：公开 API 拼写（`fordward`/`proccess`/`intermidiate`/
`miscelanous` 等）、类命名风格统一、遗留西班牙语注释翻译或删除；魔数（SASS 指令长度 16、
保留寄存器 255/63/7、wait barrier 数 6、SPACE_BITS 等）入配置或具名常量。

## 5. 顺序与验收

S1→S7 顺序执行；S5 依赖 S1（namespace 稳定后再动成员归属）。每步验收：全量门禁绿、
golden byte-identical（S3 若配置变化则仅哈希、经断言）、独立 agent 复跑、gpt-5.6-sol 评审。
账本目标：S5 结束时 `abstract_hardware_model` 与 `shader.cc` 两条违规清零，最终账本
= gpu-sim.h + shader_core_wrapper.h 两条合规下向边（保留）。

## 6. 状态

| 步 | 状态 |
| --- | --- |
| S1 namespace | 完成（golden 字节一致；`validation/refactoring/2026-07-22-stage4-s1-namespace.md`） |
| S2 头卫生 | 完成（golden 字节一致；`2026-07-23-stage4-s2-header-hygiene.md`） |
| S3 配置收敛 | 完成（深度重推导在未改 golden 下单独验证，其后仅配置哈希重批准；`2026-07-23-stage4-s3-latency-convergence.md`） |
| S4 god file 拆分 | 完成（golden 字节一致；`2026-07-23-stage4-s4-god-file-split.md`） |
| S5 成员归属重整 | 完成（**账本 4→2，两条真违规清零**；golden 字节一致；`2026-07-23-stage4-s5-ownership.md`） |
| S6 primitive 单测 | 完成（寄存器编码 helper 抽为独立 primitive，30 个 GoogleTest 链接生产目标文件；`2026-07-23-stage4-s6-primitive-tests.md`） |
| S7 命名/魔数 | 完成（11 符号改名、4 选项改名+清扫、8 处注释翻译、12 个具名常量；`2026-07-23-stage4-s7-naming.md`） |

S5 完成后账本只剩 `gpu-sim.h` 与 `shader_core_wrapper.h` 两条 L3→L2 下向边——按 §2 的定性属允许方向，非违规。目标架构的依赖方向要求已满足。

计划外收尾：遗留 AccelWattch 功耗表面清理（阶段二、五各自记下的欠账），删除不可达死分支与 31 个无引用 XML，−19787 行；验证记录 `validation/refactoring/2026-07-23-vestigial-power-removal.md`。

阶段四全部步骤完成。
