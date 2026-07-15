# Remodeled Trace Refactoring Scope and Baseline Spec

## 1. 目标

在修改 simulator 关键路径或结构之前，建立可复跑、可审计、不会把已知缺陷固化为正确行为的回归门禁。

本 spec 只冻结范围、验证契约、风险和验收标准。它不定义 P0 缺陷的实现修复方案，也不授权 P2 架构重构。

## 2. 已确认范围

### 2.1 正式执行模式

- 仅支持 remodeled trace mode。
- PTX execution path 不属于正式产品范围。
- legacy core 行为不属于正式产品范围，也不作为 remodeled core 的行为 oracle。

### 2.2 产品能力

以下能力属于产品范围，不能仅因当前 tested config 未启用而删除：

- PRT alternative selection policies；
- interwarp coalescing；
- instruction prefetching；
- IBuffer coalescing。

“属于产品范围”表示每项能力必须有明确配置契约和至少一条有效回归路径。在组合矩阵完成验证前，不因配置项可以同时设置，就默认承诺所有笛卡尔积组合均有效。

## 3. 基线分层

### 3.1 语义基线

语义基线用于证明关键路径没有路由错误、请求丢失、死锁、断言失败或非法生命周期。它优先于性能 golden。

| Case | 目的 | 必需 fixture | 当前状态 |
| --- | --- | --- | --- |
| `shared_lat` | remodeled trace 启动、shared-memory 路径、正常退出和析构 | checked-in SM89 trace | 已有 RTX 4090 候选 fixture，golden 未批准 |
| `half_pipeline` | HALF 必须进入预期执行管线 | checked-in HALF trace；可判定的管线 dispatch 统计 | 缺失 |
| `fp64_dispatch` | DP dispatch 间隔和共享 DP 路径 | checked-in FP64 trace；可判定的 DP dispatch 统计 | 缺失 |
| `memory_cache_e2e` | L1/L2/DRAM 端到端请求与完成路径 | checked-in memory-intensive trace | 已有 Ampere Rodinia `pathfinder` 候选 |

### 3.2 确定性性能基线

只有满足以下条件的结果才能进入 golden：

1. 对应 P0 语义问题已修复并通过专用语义断言；
2. trace、两个 config 和运行契约均已固定；
3. 使用 `OMP_NUM_THREADS=1` 建立首个确定性 oracle；
4. 同一输入至少连续复跑两次，所有 exact 字段一致；
5. 独立 agent 使用同一命令复跑并得到相同结果；
6. golden 记录 comparison contract、基线来源 commit、验证命令和批准状态。

Golden 的 comparison contract 包含 trace archive、两个 config、trace root/file、`OMP_NUM_THREADS`、`OMP_DYNAMIC`、`LC_ALL` 和 timeout。被测 simulator binary SHA 与当前 source commit 是 implementation provenance，只记录、不要求等于生成 golden 时的 binary；否则每次重编译都会在行为比较前必然失败。

当前代码的 observation 不能直接提升为 golden。

### 3.3 RTX 4090 硬件准确性基线

RTX 4090 硬件结果用于衡量模型准确性，不替代 simulator 自身的确定性 golden。硬件数据采用以下规则：

1. trace 与硬件测量必须来自同一个 binary 和同一输入；
2. 记录 GPU UUID、compute capability、driver、CUDA、binary hash 和 profiler 版本；
3. 未注入 tracer 的直接运行负责测量 microbenchmark 自身包围的 latency 区间；
4. Nsight Compute 负责测量整个 kernel，与 simulator 的 kernel 统计窗口对齐；
5. 每组先预热一次，再至少采样五次，使用中位数作为中心值，同时记录 min/max；
6. 硬件比较使用显式相对误差阈值，不能要求逐 cycle 精确相等。

`shared_lat` 的初始字段映射如下：

| Simulator | RTX 4090 | 比较含义 |
| --- | --- | --- |
| `gpu_tot_sim_cycle` | `sm__cycles_elapsed.avg` | 整个 kernel 的平均 SM elapsed cycles |
| `gpu_tot_sim_insn` | `sm__inst_executed.sum` | 整个 kernel 的 warp-level executed instructions |
| 无直接等价总量 | 程序输出 `Shared Memory Latency` | 2,048 次 pointer-chasing shared load 的平均 latency |

程序源码固定返回 `1`，因此硬件 runner 必须把 exit code `1` 作为该 microbenchmark 的预期退出状态；profiler 仍需成功产生完整指标。tracer 注入会显著扰动程序内的 clock 测量，注入运行只用于生成 trace，不进入硬件性能样本。

### 3.4 策略基线

P0 闭合后，为产品范围内的策略建立单变量矩阵：

- 默认 PRT policy，加每个 alternative policy；
- interwarp coalescing off/on；
- instruction prefetching off/on；
- IBuffer coalescing off/on。

先验证每项能力相对默认配置的单变量路径，再根据实际产品组合补交叉组合。每个合法组合必须完成、输出必需统计且无错误签名；每个非法组合必须在启动阶段给出明确错误。

## 4. 回归字段契约

### 4.1 当前可直接采集的 exact 字段

- process exit code；
- `GPGPU-Sim: *** exit detected ***` 完成标记；
- `gpu_tot_sim_cycle`；
- `gpu_tot_sim_insn`；
- `L2_total_cache_accesses`；
- `L2_total_cache_misses`；
- `total dram reads`；
- `total dram writes`；
- `gpgpu_n_shmem_bkconflict`。

`gpu_tot_ipc` 是派生并按四位小数打印的字段。它需要与 `gpu_tot_sim_insn / gpu_tot_sim_cycle` 一致，但 golden 比较必须使用显式绝对容差。

### 4.2 当前统计缺口

当前输出没有足以直接判定 HALF、SP、INT、DP 和 MEM 实际 dispatch 目的地的稳定聚合字段。补齐这些统计或等价测试观察点，是冻结 `half_pipeline` 和 `fp64_dispatch` golden 的前置条件。

## 5. Harness 契约

入口：`tests/remodeled_trace/run_regression.py`

- `list`：列出 manifest case，不运行 simulator；
- `observe`：运行 case 并输出带输入哈希的 observation；成功只表示“完成且字段可解析”；
- `check`：仅接受 schema 完整且已批准的 golden；golden 缺失、未批准、comparison contract 漂移或字段不匹配均返回非零退出码；
- trace 只解压到临时目录；不修改 checked-in archive；
- 默认记录 `OMP_NUM_THREADS=1`、timeout、source commit、binary/config/trace archive SHA-256；其中只有 config、trace 和运行契约进入 golden 相等比较；
- 原始日志默认不入库，checked-in validation artifact 只保存命令、退出码、输入身份和结果摘要。

## 6. 风险与边界

- Rodinia `pathfinder` 是端到端 memory/cache 候选，不能替代 HALF、FP64 或 `shared_lat` 专用 fixture。
- 当前 binary 可能不是由当前 source commit 新编译；因此 observation 同时记录 binary hash。
- `setup_environment_no_git.sh` 在 shell `nounset` 模式下会读取未定义变量；harness 必须按仓库现有用法在非 `nounset` shell 中加载它。
- 单线程 golden 不能证明 OpenMP 并行模式确定性或正确性；并行模式需作为后续独立基线层处理。
- 产品能力属于支持范围，不等于未经验证的全部策略组合已经受到支持。
- RTX 4090 的频率、温度、功耗状态和其他进程会影响硬件采样；每份 artifact 必须记录设备状态，并避免使用正在运行其他 workload 的 GPU。

## 7. 验收标准

- AC-01：本文件明确记录执行模式、产品能力、基线字段和 golden 晋升规则。
- AC-02：manifest 中至少有一个使用 checked-in trace archive 的可运行 remodeled trace case。
- AC-03：`observe` 能从干净临时目录运行该 case，并生成结构化统计摘要。
- AC-04：无 approved golden、approved golden schema 不完整或 comparison contract 漂移时，`check` 必须返回非零退出码，不能报告回归通过。
- AC-05：统计解析、容差比较、缺失字段和不安全 archive member 均有自动测试。
- AC-06：验证结果写入 checked-in artifact，包含 artifact 路径、验证命令、退出码和摘要。
- AC-07：独立 agent 只读复跑单元测试、observation 和缺失-golden guard。
- AC-08：RTX 4090 comparison artifact 记录直接 latency、Nsight whole-kernel 指标、simulator 对应字段和相对误差。

## 8. 后续退出条件

只有以下条件全部满足，才进入 simulator P0 修复：

- harness 的 AC-01 至 AC-07 闭合；
- HALF 和 FP64 两个缺失的专用 fixture 有明确生成或入库方案；
- HALF/DP 管线观察点的最小统计接口已确定。

只有 P0 修复和 approved golden 完成后，才进入 P1 支持矩阵；只有 P1 契约稳定后，才进入 P2 架构重构。
