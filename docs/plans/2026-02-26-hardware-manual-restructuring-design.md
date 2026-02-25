# 硬件设计手册重构设计

## 1. 目标

将 `docs/detailed-design/` 下的 11 份文档从"模拟器实现指南"提升为"硬件设计手册"，兼顾 RTL/微架构工程师和模拟器开发者两类读者。通过 `[Spec]`（规范层）和 `[Map]`（映射层）的严格分离，让硬件工程师能忽略 C++ 实现细节直接阅读规范，让模拟器开发者能通过映射层快速定位源码。

## 2. 约束

- 语言：保持中文为主 + 英文术语的现有风格
- 范围：分两阶段，第一阶段原地修复，第二阶段逐章深化
- 不做：不补充新的硬件设计内容（时钟域/复位/异常/面积），不补全 TL/FN 多粒度行为差异节，不补全 Framework 映射节，不重新组织章节划分

## 3. 当前问题清单

### 3.1 编号冲突

| 文件 | 冲突 |
|---|---|
| Ch2（SM 顶层设计） | 两个 `2.3`（参数定义 vs 外部接口） |
| Ch3（Subcore 流水线） | 两个 `3.4`（类结构 vs 流水线详解） |
| Ch4（依赖模型） | 两个 `4.3`（接口概览 vs 状态机） |

### 3.2 交叉引用错误

Ch0 参考文档表的章节名称与实际不符：

| 当前错误 | 实际 |
|---|---|
| "第 2 章：Subcore 流水线与调度" | 第 3 章：Subcore 流水线设计 |
| "第 3 章：功能单元与 read barrier 释放" | 第 6 章：执行资源设计 |
| "第 4 章：指令供应与 IBuffer" | 第 5 章：指令供给子系统设计 |
| "第 5 章：寄存器文件" | 第 6 章：执行资源设计（§6.6） |
| "第 8 章：依赖状态对象" | 第 4 章：依赖模型设计 |

### 3.3 内容重复

Barrier 消费机制（increment 先于 decrement）在 Ch0、Ch1、Ch2、Ch4 中重复描述 4 次，内容几乎相同。

### 3.4 [Spec]/[Map] 视图缺失

Ch0A 定义了 `[Spec]`/`[Map]` 两个视图，但仅 Ch2 较好地遵循。Ch3~Ch7 无标注，C++ 实现细节（`std::stack`、`unique_ptr`、`vector`）作为主描述出现在规范层内容中。

## 4. 方案：两轮扫描

### 第一轮 R1：全局协调

解决"读都读不下去"级别的问题。

#### R1.1 编号修复

统一为 0A 章定义的 10 节标准结构：

```
X.1 [Spec] 模块概述
X.2 [Spec] 设计规格（含参数定义 + 存储结构位宽表）
X.3 [Spec] 数据类型
X.4 [Spec] 外部接口
X.5 [Spec] 内部结构与状态机
X.6 [Spec] 时序行为
X.7 [Spec] 多粒度行为差异
X.8 [Spec] 设计原理
X.9 [Map] Framework 映射
X.10 [Map] 源码锚点
```

有额外内容的章节在对应节下用子节展开（X.5.1, X.5.2...），不另起顶级编号。

具体修复：

| 文件 | 修复 |
|---|---|
| Ch2 | 参数定义保持 `2.3`，外部接口改为 `2.4`，后续顺延 |
| Ch3 | 类结构保持 `3.4`，流水线详解改为 `3.5`，后续顺延 |
| Ch4 | 接口概览保持 `4.3`，状态机改为 `4.4`，后续顺延 |

#### R1.2 去重策略

Barrier 消费机制的权威描述归 Ch4（依赖模型设计）。其他章节改为引用：

- Ch0：保留时序图和信号行为表，删除机制解释段落，加引用 "详见 §4.4"
- Ch1：保留 Dependency_State 状态机图（作为总览），删除消费顺序的详细描述，加引用
- Ch2：保留 Phase 6/7 的一句话描述，删除 §2.6.2 的完整流程图，改为引用 "完整机制见 §4.4"

#### R1.3 交叉引用修复

修正 Ch0 参考文档表中所有章节名称，对齐到实际标题。

### 第二轮 R2：逐章 [Spec]/[Map] 分离

#### R2.1 分离原则

**[Spec] 层**：
- 用硬件术语：端口、FIFO、寄存器、位宽、状态机、时序约束
- 不出现 C++ 类型名作为主描述
- 可在括号中注明对应 C++ 类型作为参考
- 接口表标注 CA/TL/FN 粒度差异列

**[Map] 层**：
- C++ 类名、函数签名、源码行号
- Framework 映射
- 背压/恢复路径的 framework 实现方式

#### R2.2 改造清单

| 章节 | 改造量 | 改造重点 |
|---|---|---|
| Ch0（时序图） | 轻 | 加标注 |
| Ch0A（接口规范） | 无 | 已有标注，作为参考标准 |
| Ch1（总览） | 轻 | 加标注，源码 guard condition 归 [Map] |
| Ch2（SM 顶层） | 无 | 已有较好分离，作为模板 |
| Ch3（Subcore） | 重 | 8 个流水线阶段描述改为硬件语言，C++ 伪代码移到 [Map] |
| Ch4（依赖模型） | 中 | 状态机改为硬件语言，代码片段移到 [Map] |
| Ch5（指令供给） | 中 | IBuffer 协议和 L0I 访问流程改为硬件描述 |
| Ch6（执行资源） | 中 | FU 类层次改为硬件模块层次，RF 描述改为硬件语言 |
| Ch7（共享访存） | 中 | PRT 生命周期和 InterWarp Coalescing 改为硬件描述 |
| Ch8（配置矩阵） | 轻 | 加 [Spec] 标注 |
| Ch9（源码索引） | 轻 | 加 [Map] 标注 |

#### R2.3 改造顺序

1. Ch0、Ch1、Ch8、Ch9 — 轻量改造
2. Ch4 — 去重后的权威章节
3. Ch5 — 中等改造
4. Ch6 — 中等改造
5. Ch7 — 中等改造
6. Ch3 — 重点改造（最后做，前面积累经验）

#### R2.4 [Spec] 层改写示例

当前写法（Ch3 Issue 阶段）：
```
在正常状态下（`m_ISSUE_CONTROL_latch.has_free()` 且
`m_num_pending_cycles_with_issue_port_busy == 0`）：
* 首先调用 `modify_warp_state()`，对本 subcore 的每个 warp
  调用 `Dependency_State::cycle()`
```

改为 [Spec] 层：
```
### X.5.3 [Spec] Issue 阶段

前置条件：issue→control latch 空闲 且 issue port busy 计数器 = 0

每周期操作：
1. 更新所有 warp 的依赖状态（stall/yield 衰减，详见 §4.5）
2. 按 Greedy-then-Highest-ID 顺序遍历 warp（§X.5.3.1）
3. 对每个候选 warp 检查就绪条件（§X.5.3.2）
4. 首个满足所有条件的 warp 发射，每周期至多 1 条
```

对应 [Map] 层：
```
### X.9.3 [Map] Issue 阶段映射

| 规范层概念 | 实现 |
|---|---|
| issue→control latch 空闲 | `m_ISSUE_CONTROL_latch.has_free()` |
| issue port busy 计数器 | `m_num_pending_cycles_with_issue_port_busy` |
| 更新依赖状态 | `modify_warp_state()` → `Dependency_State::cycle()` |
| Greedy-then-Highest-ID | `order_greedy_then_highest_id()` |
```

## 5. 交付物与验收标准

| 阶段 | 交付物 | 验收标准 |
|---|---|---|
| R1 | 11 个 .md 编号修复 + 去重 + 交叉引用修复 | 无编号冲突；barrier 机制仅在 Ch4 详述；Ch0 参考文档表与实际章节名一致 |
| R2 | 11 个 .md 的 [Spec]/[Map] 分离 | 每个 X.1~X.8 有 [Spec] 前缀；X.9~X.10 有 [Map] 前缀；[Spec] 层无 C++ 类型名作为主描述 |

## 6. Git 提交策略

- R1：一次提交 `docs(detailed-design): fix numbering conflicts, deduplicate barrier content, fix cross-references`
- R2：每章一次提交 `docs(detailed-design): add [Spec]/[Map] annotations to ChN`

## 7. 风险

- Ch3 的 [Spec]/[Map] 分离工作量最大（~1000 行），可能需要 2-3 轮迭代
- 去重时需确保引用路径正确，避免读者找不到被引用内容
