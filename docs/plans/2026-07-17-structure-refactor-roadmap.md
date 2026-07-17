# 结构重构路线图（2026-07-17）

## 1. 背景与已确认决策

全仓结构评审（2026-07-17）确认的核心问题：

- `remodeling/` 与 legacy `shader.h` 双向循环依赖；`warp_inst_t` 内嵌约 41 个 remodeling 专用成员（含 `functional_unit*`）；`shader_core_ctx_wrapper` 为约 110 纯虚方法的过宽接口；`SM` 复制粘贴大量 legacy `shader_core_ctx` 方法。
- 已证实的死代码与死重：`remodeling/gmmu.*` 与 `page_table_walker.*` 从未实例化；intersim2 编译链接但 `INTERSIM` 分支启动即 `abort()`，而 `-network_mode` 默认值恰为该分支；两个从未参与最终链接的静态库。
- 三套并行的延迟配置命名空间（`-ptx_opcode_latency_*`、`-trace_opcode_latency_initiation_*`、remodeling 独立选项）互相重叠。
- 仓库卫生：根目录多份矛盾入口文档、空实验脚手架、脆弱的全局 `*.json` gitignore、上游文档无标注、`arch.md` 与 `docs/detailed-design/` 重复且存在 True-Path 假设与能力矩阵的实质矛盾。
- 已跟踪的 `tests/remodeled_trace/cases.json` 引用未跟踪的 fixture，新克隆会直接失败。

**已拍板决策**：

1. 退役 legacy shader 时序路径。`docs/plans/2026-07-15-remodeled-trace-p0-p1-semantic-repair.md` 已声明 remodeled-trace 为唯一支持模式，本决策与其一致。
2. 阶段顺序为：安全网 → 仓库卫生 → 死重清除 → 退役 legacy → 退役后重组。先退役再深度解耦，避免为兼容双模型做一次性接口拆分。

## 2. 目标与非目标

**目标**：可维护性——单向依赖、单一事实源（配置、文档、入口）、可复跑验证、克隆即可用。

**非目标**：

- 不改动 remodeling 时序模型的算法与统计语义（除非评审确认为 bug 且单独立项）。
- 不对上游 Accel-sim 遗留代码做整体格式化或改名（保持与上游 diff 可比性）。
- 不为未出现的需求预留抽象。

## 3. 执行模型

- 统筹者负责：阶段拆解、执行者简报、结果整合、独立复跑验证、阶段间排序。
- 执行者（子代理）负责：按简报实施、自测、产出验证记录。
- 每阶段完成后由独立评审者（Codex CLI）做只读评审；评审发现由统筹者确认适用性后决定是否修复。
- 每阶段的完成声明必须绑定 checked-in artifact：验证命令、退出码、结果摘要、独立复跑结果，落在 `validation/refactoring/` 下的带日期记录中。

**全局约束**（对所有执行者生效）：

- 代码与注释仅英文；commit message 用英文 conventional 风格，禁用进度类词汇与 AI 工具名称。
- 回归门禁：每阶段结束时 `python3 -m unittest discover -s tests` 与 `tests/remodeled_trace/run_regression.py` check 模式必须通过（安全网建立后）。
- 禁止在验证失败或环境阻塞时扩大改动范围；阻塞必须显式上报。
- 不推送远端；本地提交由用户决定何时推送。

## 4. 阶段划分

### 阶段〇：固化安全网

范围：

- 验证工作区在途的语义修复改动（构建 + 单测 + 回归 observe），按逻辑内聚原子提交：模拟器修复为一组；测试夹具、生成器、`cases.json`、`.gitignore` 白名单为一组；plan 文档（2026-07-15 与本文档）为一组。`cases.json` 必须与其引用的 fixture 同 commit。
- 按 `run_regression.py` 既定流程将 `goldens.json` 从 unapproved 转正，使 check 模式可用。
- 为初始导入 commit（`117f9dc`）打 `upstream-import` 标签，作为永久基线参照。
- 在根 `README.md` 声明测试入口命令。

验收标准：

- 新克隆视角下 `run_regression.py list` 可通过 manifest 校验；check 模式对全部用例通过并以退出码 0 结束。
- `git status` 清洁（不含刻意保留的本地未跟踪项）；`upstream-import` 标签存在。
- 验证记录落盘 `validation/refactoring/`。

风险与回滚：改动均为提交既有工作与配置转正，不新增逻辑；回滚即 revert 对应 commit。

### 阶段一：仓库卫生

范围：

- 删除：`comparison_results/` 空目录树、`tests/build/` 孤儿构建目录、`docs/refer.md` 空文件、`docs/plans/claude-plan.md` 草稿（其意图并入本路线图）。
- `refactoring-audit.html` 移入 `docs/` 并跟踪，或经用户确认后删除。
- `AGENTS.md` 改为指向 `CLAUDE.md` 的单行说明文件。
- `.gitignore`：以具体路径替换全局 `*.json` 与白名单；移除 `./` 前缀写法。
- 文档整合：上游文档（`simulator-remodeled/README.md`、`release.notes.md`、`gpgpu-sim4.md`、`AccelWattch.md`）加 UPSTREAM 标注；`arch.md` 降级为指向 `docs/detailed-design/` 的入口页；以 2026-07-15 plan 的能力矩阵为准消解 True-Path 布尔集矛盾；`simulator-remodeled/docs/plans/` 并入根 `docs/plans/`；修复 `09-源码锚点索引.md` 中两处过时调用链描述（不存在的 `Subcore::fetch_L0I_cycle`、`SM::cycle()` 阶段顺序）；`usage.md` 并入 README 或改为其链接；为 `docs/detailed-design/` 增加索引页。
- 处置停滞分支 `refactor/true-path-scoreboard-cleanup`：收编其 C++ 单测思路记录到本路线图阶段三待办，分支本身归档删除（保留说明）。

验收标准：文档链接可达、无断链；回归门禁通过（文档与忽略规则改动不影响行为，仍复跑以证明）；验证记录落盘。

风险与回滚：零行为风险；逐 commit 可 revert。

### 阶段二：死重清除与默认值修复

范围：

- 删除 `remodeling/gmmu.*`、`remodeling/page_table_walker.*` 及仅服务于它们的 TLB 占位管道。
- `-network_mode` 默认改为 LOCAL_XBAR；intersim2 移出编译与链接；`icnt_wrapper` 中 INTERSIM 分支改为明确的配置错误提示。
- `-gpgpu_sub_core_model` 等默认值对齐 2026-07-15 plan 的支持契约，或增加启动时非法组合校验。
- 构建清理：删除两个死静态库的归档规则；修复 `src/gpgpu-sim/Makefile` 中 `%.o` 对 remodeling 目录的递归依赖放大；清除 `remodeling/Makefile` 中无对应源文件的拷贝规则。
- 代码清理：`#if 0` 块、成片注释掉的埋点、热路径调试 `printf`、`Register_file_cache::print` 误调 `flush` 的修正。
- 从默认构建目标中剥离 `libopencl`、`cuobjdump_to_ptxplus`、`debug_tools`（保留为可选目标）。

验收标准：默认配置启动不再落入 abort 分支（新增用例或在验证记录中给出命令与退出码）；全量回归 check 与转正 golden 完全一致；构建产物与阶段〇基线二进制行为一致（以回归统计为准）；验证记录落盘。

风险与回滚：中低。每个删除项独立 commit；若回归波动，二分定位并 revert 单项。

### 阶段三：退役 legacy shader 时序路径

前置：阶段〇至二完成且回归门禁绿色。在专用分支实施，绿色并通过评审后并回 `dev_dzw`。

范围（按序小步）：

1. 收敛模型选择：删除 `exec_shader_core_ctx` / exec cluster 分支与重复的 `create_shader_core_ctx` 选择逻辑；`-is_SM_remodeling_enabled` 变为仅接受启用值（关闭即报错退出，含迁移提示）。
2. 拆解 `shader.cc`/`shader.h`：辨析并保留 SM 路径实际依赖的共享设施（`shd_warp_t`、Scoreboard 变体、cache、barrier 等），删除 legacy 流水线机器（调度器、操作数收集器、legacy ldst、simd 功能单元等）及其配置项。
3. 收缩 `shader_core_ctx_wrapper` 至 cluster 实际调用面；评估 cluster 直接持有 `SM` 的可行性。
4. 消除 `SM` 中与已删 legacy 方法同源的复制粘贴（保留唯一实现）。
5. 吸收停滞分支中 scoreboard legacy path 清理的有效部分；补充该分支曾尝试的 C++ 单元测试（依赖路径、流水线路由）作为回归的补强。

验收标准：全量回归 check 与阶段二基线完全一致；被删配置项在配置文件中出现时给出明确错误而非静默忽略；`remodeling/` 不再被 legacy 头反向 include（`grep` 证据入验证记录）；验证记录落盘。

风险与回滚：高。专用分支 + 每小步独立 commit + 每步回归；任何红色立即停止并上报，不得扩大范围。

### 阶段四：退役后重组

范围：

- `warp_inst_t`/`shd_warp_t` 中 remodeling 成员的归属重整（退役后按最终所有权就地收编或抽结构体）。
- `remodeling/` 引入 `namespace`；全局自由函数收编；头文件卫生（头内全局数组、应在 `.cc` 的方法体、重复 `#define`）。
- 延迟配置收敛为单一权威命名空间，旧选项保留为别名并告警。
- god file 拆分：`ldst_unit_sm.cc` 按类分文件、`ldst_unit_sm::cycle` 与 `Subcore::issue` 分解为具名阶段方法。
- 命名清理：公开 API 拼写错误（`fordward`、`proccess`、`intermidiate` 等）、类命名风格统一、遗留西班牙语注释翻译或删除。
- 魔数入配置或具名常量（SASS 指令长度、保留寄存器编码、wait barrier 数量等）。

验收标准：全量回归 check 与阶段三基线完全一致；每项重组独立 commit；验证记录落盘。

风险与回滚：中。纯结构等价变换，回归逐 commit 把关。

## 5. 待办与开放项

- 44 MB `util/accelwattch/accelwattch_benchmarks/validation.tgz` 与本地 `4.2/` SDK 的去留（影响克隆体积；等用户结论，不阻塞各阶段）。
- 阶段三第 2 步的共享设施清单需在实施前产出并评审（作为该阶段第一个交付物）。

## 6. 状态

| 阶段 | 状态 | 验证记录 |
| --- | --- | --- |
| 〇 安全网 | 进行中 | 待产出 |
| 一 仓库卫生 | 未开始 | — |
| 二 死重清除 | 未开始 | — |
| 三 退役 legacy | 未开始 | — |
| 四 退役后重组 | 未开始 | — |
