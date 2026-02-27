# SM Architecture Diagram Enhancement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 `02-SM顶层设计.md` 增加与论文 `modern-gpu.pdf`（Figure 3 / Section 5）对齐的 SM 架构图，并补全详细说明文档，便于外部 reviewer（Claude）逐项核对。

**Architecture:** 主文档增加“论文对齐架构图”总览与术语映射；详细说明按两个独立主题拆分为前端/执行与访存/共享两份文档，形成“主图 + 分册详解”的可审阅结构。并行执行以减少串行等待。

**Tech Stack:** Markdown, Mermaid, local document cross-reference

---

### Task 1: 提取论文依据并形成映射清单

**Files:**
- Modify: `/tmp/modern-gpu.txt`（临时提取文本）
- Test: `n/a`

**Step 1: 抽取论文文本**

Run: `pdftotext '/home/duanzhenwei/modern-gpu-simulator-micro-2025/docs/modern-gpu.pdf' /tmp/modern-gpu.txt`
Expected: `/tmp/modern-gpu.txt` 生成成功。

**Step 2: 定位 Figure 3 与 Section 5 的关键术语**

Run: `rg -n "Figure 3|CGGTY|Front-end|Register File|Memory Pipeline|L0|L1|stream buffer|Dependence" /tmp/modern-gpu.txt`
Expected: 输出包含架构图组件与术语线索。

**Step 3: 建立主文档术语映射草表**

Run: 人工整理（写入后续 Task 2 的新章节）。
Expected: 形成“论文术语 -> 本章术语 -> 细化文档位置”清单。

### Task 2: 更新主文档架构图章节

**Files:**
- Modify: `/home/duanzhenwei/modern-gpu-simulator-micro-2025/docs/detailed-design/02-SM顶层设计.md`
- Test: `n/a`

**Step 1: 新增 2.1A 章节与论文对齐架构图**

Run: 编辑 Markdown，插入 Mermaid 架构图（Figure 3 对齐）。
Expected: 图中包含前端、Issue/Control/Allocate、RF/RFC、访存 local/shared 路径。

**Step 2: 新增术语映射表与阅读顺序**

Run: 编辑 Markdown，添加“论文术语映射表”和“主文档 -> 分册详解”的阅读路径。
Expected: reviewer 可从主文档跳转到细节文档。

**Step 3: 自检 Markdown 结构**

Run: `rg -n "2.1A|论文对齐架构图|02A-|02B-" '/home/duanzhenwei/modern-gpu-simulator-micro-2025/docs/detailed-design/02-SM顶层设计.md'`
Expected: 新章节与链接均可检索到。

### Task 3: 并行产出详解分册（独立任务）

**Files:**
- Create: `/home/duanzhenwei/modern-gpu-simulator-micro-2025/docs/detailed-design/02A-SM论文对齐架构图详解-前端与执行.md`
- Create: `/home/duanzhenwei/modern-gpu-simulator-micro-2025/docs/detailed-design/02B-SM论文对齐架构图详解-访存与共享路径.md`
- Test: `n/a`

**Step 1: 生成前端/执行分册（Agent-B）**

Run: 编辑 `02A-...md`
Expected: 覆盖 Fetch/Decode/Issue/CGGTY/Dependence/Control/Allocate/RF/RFC/写回仲裁。

**Step 2: 生成访存/共享分册（Agent-C）**

Run: 编辑 `02B-...md`
Expected: 覆盖 memory local/shared、L1D/SMEM/Texture、吞吐与延迟、LDGSTS、常量路径。

**Step 3: 校验分册拆分合理性（满足“内容大于 500 时拆分”要求）**

Run: `wc -l '/home/duanzhenwei/modern-gpu-simulator-micro-2025/docs/detailed-design/02A-SM论文对齐架构图详解-前端与执行.md' '/home/duanzhenwei/modern-gpu-simulator-micro-2025/docs/detailed-design/02B-SM论文对齐架构图详解-访存与共享路径.md'`
Expected: 已拆分为两份可独立审阅文档。

### Task 4: 集成与最终校验

**Files:**
- Modify: `/home/duanzhenwei/modern-gpu-simulator-micro-2025/docs/detailed-design/02-SM顶层设计.md`
- Test: `n/a`

**Step 1: 回填交叉引用**

Run: 在主文档中加入指向 `02A`、`02B` 的链接与用途说明。
Expected: reviewer 能按路径完成结构化审阅。

**Step 2: 检查标题和章节层级**

Run: `rg -n '^## |^### ' '/home/duanzhenwei/modern-gpu-simulator-micro-2025/docs/detailed-design/02-SM顶层设计.md'`
Expected: 新增章节位置合理且编号连续。

**Step 3: 差异核验**

Run: `git diff -- '/home/duanzhenwei/modern-gpu-simulator-micro-2025/docs/detailed-design/02-SM顶层设计.md' '/home/duanzhenwei/modern-gpu-simulator-micro-2025/docs/detailed-design/02A-SM论文对齐架构图详解-前端与执行.md' '/home/duanzhenwei/modern-gpu-simulator-micro-2025/docs/detailed-design/02B-SM论文对齐架构图详解-访存与共享路径.md'`
Expected: 仅包含本次文档增强改动。
