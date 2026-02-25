# arch.md 强力 Review 与 True-Path 重写设计

## 1. 目标
把 `arch.md` 重写为“实现即文档”的单一路径架构文档，聚焦 `remodeling/` 目录内 `is_*` 控制项为真时的有效运行路径；唯一例外为 `is_remodeling_scoreboarding_enabled=false`，以满足仅 control-bit 依赖路径建模。

## 2. 已确认范围与约束
- 交付物：仅修订后的 `arch.md` 正文，不保留审计清单章节。
- 文档形态：单一真值路径，不写“可选/否则/关闭时”。
- 锚点策略：每节给出简短源码锚点（文件+函数）。
- 特殊约束：
  - `is_remodeling_scoreboarding_enabled=false`（唯一例外）
  - `is_trace_mode=true`
  - kernel `is_captured_from_binary=true`

## 3. True-Path 假设集合（文档采用）
- `is_trace_mode=true`
- `is_captured_from_binary=true`
- `is_remodeling_scoreboarding_enabled=false`
- `is_ibuffer_remodeled_enabled=true`
- `is_interwarp_coalescing_enabled=true`
- `is_instruction_prefetching_enabled=true`
- `is_fp32_and_int_unified_pipeline=true`
- `is_fp32ops_allowed_in_int_pipeline=true`（文档会标注该项与 unified pipeline 的实现耦合风险）
- `is_dp_pipeline_shared_for_subcores=true`
- `is_rf_cache_enabled=true`
- `is_loog_enabled=true`
- `is_vpreg_enabled=true`

## 4. 主要修订方向
1. 修正执行顺序与资源映射错误
- `SM::cycle()` 顺序按源码重写。
- `Subcore::cycle()` 的逆序流水线顺序按源码重写。
- 删除不符合实现的结构描述（如 Subcore 私有 `dispatch_latches[]`、独立 HP 管线、4 套 RF 实例）。

2. 明确 control-bit 依赖路径
- 说明 issue/read-release/retire 三处对 stall/yield/wait barrier 的作用点。
- 明确该路径成立的前提：trace + captured_from_binary + remodeling scoreboard disabled。

3. 明确共享访存单元生命周期
- 以 PRT 为主线描述 `issue -> coalescing -> dispatch -> completion -> pop/release`。
- 纳入 `is_loog_enabled/is_vpreg_enabled=true` 对 pending writes key 的影响。

4. 清理文档污染与术语不一致
- 删除文末混入的用户提示文本。
- 统一术语与代码命名（函数名、结构名、路径名）。

## 5. 文档骨架（重写后）
1. Scope & Assumptions
2. SM Top-Level Execution Order
3. Subcore Pipeline (Issue/Control/Fetch-Decode)
4. Dependency Model (control-bit only)
5. Instruction Supply Path (IBuffer/L0I/L0_icnt/L1I)
6. Execution Resources (FU/RF/result queues)
7. Shared Memory Pipeline (`ldst_unit_sm` + PRT)
8. Effective Configuration Matrix
9. File Map & Source Anchors

## 6. 验收标准
- 文中不存在分支化叙述（可选、否则、关闭时）。
- 依赖模型主路径为 control-bit（非传统 scoreboard）。
- 执行顺序、资源映射与源码一致。
- 文末污染文本被移除。
- 每节具备可追溯源码锚点。
