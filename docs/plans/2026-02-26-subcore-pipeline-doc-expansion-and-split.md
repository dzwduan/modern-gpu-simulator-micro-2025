# Subcore Pipeline Doc Expansion And Split Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 `03-Subcore流水线设计.md` 从“单文件长文档”重构为“总览 + 分卷详解”，并基于源码补充关键阶段细节，确保单文档超过 500 行时自动拆分。

**Architecture:** 保留 `03` 作为章节总览与公共规格；新增 `03A` 承载前端与调度路径（Fetch/Decode/Issue/Control）详解；新增 `03B` 承载后端执行路径（Allocate/Read_RF/Execute/Writeback）详解。通过源码锚点、状态机、停顿传播链路和周期级时序补足细节深度。

**Tech Stack:** Markdown 文档、bash（`sed`/`rg`/`wc`）

---

### Task 1: 建立拆分骨架与行数约束

**Files:**
- Modify: `docs/detailed-design/03-Subcore流水线设计.md`
- Create: `docs/detailed-design/03A-Subcore前端流水与调度设计.md`
- Create: `docs/detailed-design/03B-Subcore后端执行与写回设计.md`

**Step 1: 复制原文按章节切片**

Run:
```bash
sed -n '1,264p' docs/detailed-design/03-Subcore流水线设计.md > /tmp/03-main.md
sed -n '265,586p' docs/detailed-design/03-Subcore流水线设计.md > docs/detailed-design/03A-Subcore前端流水与调度设计.md
sed -n '587,1009p' docs/detailed-design/03-Subcore流水线设计.md > docs/detailed-design/03B-Subcore后端执行与写回设计.md
```
Expected: 生成 `03A/03B` 初始骨架，主文档内容暂存到 `/tmp/03-main.md`。

**Step 2: 主文档改为“总览入口 + 分卷导航”**

Run:
```bash
# 手工编辑 03 主文档，加入“文档拆分说明”和 03A/03B 链接
```
Expected: `03` 成为总览入口，明确拆分规则与阅读顺序。

**Step 3: 行数阈值检查**

Run:
```bash
wc -l docs/detailed-design/03-Subcore流水线设计.md \
      docs/detailed-design/03A-Subcore前端流水与调度设计.md \
      docs/detailed-design/03B-Subcore后端执行与写回设计.md
```
Expected: 任一文件若 `>500` 行，继续拆分出 `03C`（自动扩展）。

**Step 4: Commit**

```bash
git add docs/detailed-design/03-Subcore流水线设计.md \
        docs/detailed-design/03A-Subcore前端流水与调度设计.md \
        docs/detailed-design/03B-Subcore后端执行与写回设计.md
git commit -m "docs: split subcore pipeline chapter into overview and detailed volumes"
```

---

### Task 2: 补强前端路径细节（03A）

**Files:**
- Modify: `docs/detailed-design/03A-Subcore前端流水与调度设计.md`
- Reference: `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc`
- Reference: `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc`
- Reference: `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/warp_dependency_state.cc`

**Step 1: 增补阶段契约（entry/exit/invariant）**

- Fetch/Decode/Issue/Control 每级新增：
  - 入口条件
  - 出口保证
  - 不变量
  - 反压来源

**Step 2: 增补 Issue 判定链路细节**

- 细化 `are_switch_warp_conditions_ready` 各子条件与短路行为
- 增补 `m_num_pending_cycles_constant_cache_misses_before_switch_to_other_warp` 对 greedy warp 切换抑制逻辑
- 补充 stall 统计计数器映射

**Step 3: 增补 Barrier 与依赖模型时序**

- 描述 control 阶段设置 increment、SM 周期末消费 pending stack 的时序边界
- 说明 `yield/stall_counter` 的右移衰减机制与效果

**Step 4: 验证前端文档结构**

Run:
```bash
rg -n "^## |^### " docs/detailed-design/03A-Subcore前端流水与调度设计.md
```
Expected: 章节层次完整，新增细节段落可定位。

**Step 5: Commit**

```bash
git add docs/detailed-design/03A-Subcore前端流水与调度设计.md
git commit -m "docs: deepen fetch-decode-issue-control pipeline details"
```

---

### Task 3: 补强后端路径细节（03B）

**Files:**
- Modify: `docs/detailed-design/03B-Subcore后端执行与写回设计.md`
- Reference: `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.cc`
- Reference: `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc`
- Reference: `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/register_file.cc`
- Reference: `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.cc`

**Step 1: 增补 Allocate/RF 建模细节**

- 补充 `RF_instruction_read_request`、bank 端口窗口、`max_slack_due_to_double_use_of_banks` 机制
- 增补 regular/uniform RF 差异与 `rf_cache` 适用范围

**Step 2: 增补 Execute 内部流水细节**

- 固定延迟 FU：dispatch_reg → pipeline_reg → predicate extra stages 的推进规则
- 队列型 FU：`m_queue`、`m_intermediate_stages`、WAR 释放时点、向 SM 共享单元发送节流

**Step 3: 增补 Writeback 冲突与回压传播**

- 细化 `writeback_latch_proccess()` 在普通写回 vs SM 共享写回场景下的差异
- 明确“写端口冲突 -> latch 停留 -> 上游 result port 堵塞 -> execute 完成阻塞”的链式回压

**Step 4: 验证后端文档结构**

Run:
```bash
rg -n "^## |^### " docs/detailed-design/03B-Subcore后端执行与写回设计.md
```
Expected: 后端分支、共享路径、冲突处理均有独立小节。

**Step 5: Commit**

```bash
git add docs/detailed-design/03B-Subcore后端执行与写回设计.md
git commit -m "docs: deepen allocate-readrf-execute-writeback pipeline details"
```

---

### Task 4: 最终一致性与自动拆分检查

**Files:**
- Modify: `docs/detailed-design/03-Subcore流水线设计.md`
- Modify: `docs/detailed-design/03A-Subcore前端流水与调度设计.md`
- Modify: `docs/detailed-design/03B-Subcore后端执行与写回设计.md`
- Optional Create: `docs/detailed-design/03C-*.md`

**Step 1: 交叉链接检查**

Run:
```bash
rg -n "03A|03B|03C|续篇|前文见|后文见" docs/detailed-design/03*.md
```
Expected: 主文档与分卷互链完整。

**Step 2: 行数阈值自动检查**

Run:
```bash
wc -l docs/detailed-design/03*.md
```
Expected: 若任一文件 `>500`，继续拆分并更新导航。

**Step 3: 差异审查**

Run:
```bash
git diff -- docs/detailed-design/03-Subcore流水线设计.md \
             docs/detailed-design/03A-Subcore前端流水与调度设计.md \
             docs/detailed-design/03B-Subcore后端执行与写回设计.md
```
Expected: 变更集中在文档增强与拆分，不引入无关修改。

**Step 4: Commit**

```bash
git add docs/detailed-design/03*.md
git commit -m "docs: expand and split subcore pipeline detailed design"
```

---

**执行方式（已按用户要求预选）**

1. Subagent-Driven（本会话）(Recommended): 使用并行分工方式执行 Task 1-4，并在每个任务后做集成校验。
2. Parallel Session（独立会话）: 新开会话按 `executing-plans` 串行执行。
