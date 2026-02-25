# Plan: 将 detailed-design 重构为硬件手册主规范（支持 framework 映射与 RTL 实现）

## Context

当前 `docs/detailed-design/` 的章节主要以 cycle-driven C++ 实现视角描述（`register_set_uniptr`、`warp_inst_t*`、函数调用接口、"1 inst" 宽度）。

目标不是把文档写成某个特定框架说明书，而是产出可直接用于以下场景的微架构硬件手册：

1. 新人基于手册理解微架构设计原理与关键时序。
2. 将设计语义映射到 event-driven framework（例如 `/home/duanzhenwei/HopperArchSim/framework`）。
3. 作为 RTL 建模输入（信号、位宽、协议、状态机与约束完整）。

## 核心定位与边界

### A. 双层语义（必须同时存在）

1. 规范层（Normative Spec，主内容）
- 使用硬件术语定义模块、接口、位宽、协议、时序、状态机、约束。
- 不出现 C++ 指针/容器/函数调用接口作为主描述。
- 必须可被 RTL 工程师直接消费。

2. 实现映射层（Implementation Mapping，附录或子节）
- 说明规范层如何映射到具体实现（framework、软件模型等）。
- framework 只是一个映射目标，不是手册主语义来源。

### B. 协议语义要求

- 规范层统一采用硬件握手语义（`valid/ready` + payload）。
- 映射层必须明确：在 framework 中 `ready` 的等价物是 `try_send()` 返回状态。
  - `kOk` 等价 `ready=1` 的成功传输。
  - `kAgain` 等价 `ready=0`（背压），发送端必须等待 `kChannelAvailable` 事件后重试。
- 明确写出这是“语义等价映射”，不是“实现必须有 ready 线网”。

### C. 多粒度要求

所有模块都需提供 `CA/TL/FN` 三粒度行为说明：
- CA: 周期级阶段行为与逐周期握手/资源占用。
- TL: 请求/响应事务语义与可配置延迟。
- FN: 仅功能正确性语义，时序抽象最小化。

## 策略

先做模板再推广：
1. 先产出“总规范文档（接口/映射/模板）”。
2. 先重写 Ch2（SM 顶层）作为风格样板。
3. 通过 review 固化写作规则后再推广到其余章节。

---

## Task 1: 创建总规范文档（硬件手册写作基线）

### 文件

- 新建：`docs/detailed-design/00A-接口规范与映射约定.md`

说明：当前目录已有 `00-关键路径时序图.md`，使用 `00A` 避免编号冲突和已有链接失效。

### 内容要求

#### 1.1 文档约定（规范层 vs 映射层）

定义每个模块章节必须包含两个视图：
- `规范层`：硬件定义。
- `实现映射层`：framework/软件实现映射。

并规定视觉标记：
- 规范层小节标题统一前缀：`[Spec]`
- 映射层小节标题统一前缀：`[Map]`

#### 1.2 核心数据类型规范（规范层）

以增强 Markdown 表格定义统一 payload（非 SV 代码），至少包含：
- `warp_inst_core_t`（由 `warp_inst_t` 抽象出的硬件最小必要字段）
- `mem_req_t` / `mem_rsp_t`（由 `mem_fetch_t` 抽象）
- `wait_barrier_mod_t`

要求：
- 每个字段标注位宽、编码空间、语义。
- 对“来自软件模型但不应直接暴露到硬件接口”的字段单独标注（避免泄露实现细节）。

#### 1.3 接口协议约定

定义端口表统一列：
- 信号名 | 方向 | 位宽 | 协议 | 来源/去向 | 粒度差异（CA/TL/FN） | 约束说明

协议说明至少覆盖：
- 传输成立条件（`valid && ready`）
- 背压行为
- 顺序保证（同源/同通道）

#### 1.4 事件与握手的语义映射（映射层）

新增映射规则表：
- `valid/ready` 握手 <-> `try_send / kAgain / kChannelAvailable`
- 输入到达 <-> `kMsgArrive`
- 组件内部延迟/定时推进 <-> `kComponentWake`

要求写清：
- 哪些是规范层强约束。
- 哪些是 framework 的实现策略。

#### 1.5 多粒度标注模板

定义统一模板，明确：
- CA 需要的时序细节最小集合（阶段、资源、仲裁、阻塞点）。
- TL 需要的事务定义最小集合（请求、响应、延迟参数、并发限制）。
- FN 需要的功能语义最小集合（输入到输出、异常条件）。

#### 1.6 软件接口到硬件接口映射规则

保留原有“软件模式 -> 硬件模式”方向，但增强为三列：
- 软件结构
- 规范层硬件表示
- framework 映射策略

必须覆盖：
- `register_set_uniptr`
- `accept_xxx(mem_fetch*)`
- `push(mem_fetch*)`
- `stack/queue`
- C++ 枚举到编码信号

#### 1.7 模块章节标准模板

每章标准结构更新为：
1. `[Spec]` 模块概述
2. `[Spec]` 参数定义
3. `[Spec]` 数据类型
4. `[Spec]` 外部接口
5. `[Spec]` 内部结构与状态机
6. `[Spec]` 多粒度行为（CA/TL/FN）
7. `[Spec]` 设计原理（为什么）
8. `[Map]` framework 映射
9. `[Map]` 源码锚点

---

## Task 2: 重写 Ch2 `02-SM顶层设计.md`（模板样章）

### 文件

- 重写：`docs/detailed-design/02-SM顶层设计.md`

### 内容要求

#### 2.1 `[Spec]` 模块概述

- SM 在 GPU 微架构中的职责边界。
- 与 Subcore、共享执行资源、访存层次、互连的关系。

#### 2.2 `[Spec]` 参数定义

- `NUM_SUBCORES`、`MAX_WARPS_PER_SM`、`MAX_CTA_PER_SM` 等。
- 给出参数含义、取值范围、对位宽/资源规模的影响。

#### 2.3 `[Spec]` 外部接口表（硬件信号级）

将现有 5 输入 + 2 输出展开为接口级信号定义（`valid/ready/payload` 语义），包含：
- `subcore_mem_inst[N-1:0]`
- `ldgsts_reentry_inst`
- `subcore_dp_inst`
- `icache_resp`
- `mem_resp`
- `subcore_wb_inst[N-1:0]`
- `icnt_req`

每个接口必须给出：
- payload 类型引用
- 位宽与编码说明
- CA/TL/FN 下差异

#### 2.4 `[Spec]` 内部结构与关键数据流

- 保留原有 SM 结构主线（Subcore、共享 DP、共享 LDST、barrier 管理等）。
- 改写为硬件结构图/数据流描述，避免 C++ 对象关系作为主表述。

#### 2.5 `[Spec]` 时序行为

- 保留原 Phase 1-8 语义，重写为“规范时序序列”。
- 明确各 phase 的输入前提、状态更新、输出结果。
- 给出 CA/TL/FN 的行为裁剪规则。

#### 2.6 `[Spec]` 指令退休与 Wait-Barrier

- 以状态更新和接口动作描述退休流程。
- 解释 increment/decrement 两阶段消费的必要性与正确性约束。

#### 2.7 `[Spec]` 设计原理（Why）

至少解释：
- 共享访存单元相对每-subcore独立访存的取舍。
- barrier 两阶段消费对一致性/吞吐的影响。
- 与真实 GPU 设计的对应关系（只讲可验证共性，不写无法确认的实现细节）。

#### 2.8 `[Map]` framework 映射附录

为 Ch2 的每个外部接口给出映射条目：
- 规范层握手语义
- framework 事件/通道落地方式
- 背压与恢复路径

并补充映射锚点示例（来自 framework）：
- `GPUChannel::try_send / kAgain / notify_available`
- `GPUSimEngine::dispatch` 中的 `kMsgArrive / kChannelAvailable / kComponentWake`

#### 2.9 `[Map]` 源码锚点

- 保留现有微架构源码锚点（`simulator-remodeled/...`）作为主锚点。
- framework 锚点单列为“实现参考锚点”，避免混淆为规范定义来源。

---

## 涉及文件

| 文件 | 操作 |
|---|---|
| `docs/detailed-design/00A-接口规范与映射约定.md` | 新建 |
| `docs/detailed-design/02-SM顶层设计.md` | 重写 |
| `docs/plans/claude-plan.md` | 更新（本次） |

---

## 验证标准（完成判据）

1. 硬件手册完整性
- Ch2 能独立回答“模块做什么、接口是什么、何时更新状态、为何这样设计”。

2. 规范层纯度
- 规范层不出现 `warp_inst_t*`、`mem_fetch*`、`register_set_uniptr` 作为接口定义本体。

3. framework 可移植性
- 每个规范接口均有对应的 `[Map]` 映射条目。
- 映射条目覆盖背压与恢复机制，不遗漏 `kAgain -> kChannelAvailable`。

4. RTL 可实现性
- 所有接口均有位宽、方向、协议与编码说明。
- 状态更新条件与时序顺序可直接转成 RTL 状态机/流水线规则。

5. 一致性
- Ch2 的术语、表格列、粒度标注完全符合 `00A` 总规范。

6. 可读性
- 每个关键机制都包含“为什么”解释，而不仅是“做什么”。

---

## 推广策略

在 Ch2 通过 review 后，再按同模板推广到：
- `03-Subcore流水线设计.md`
- `04-依赖模型设计.md`
- `05-指令供给子系统设计.md`
- `06-执行资源设计.md`
- `07-共享访存单元设计.md`
