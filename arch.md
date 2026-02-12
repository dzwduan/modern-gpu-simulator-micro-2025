# Remodeling 模块架构文档

## 1. 总体概述

### 1.1 项目定位

`remodeling` 模块是对 GPGPU-Sim 中 Streaming Multiprocessor (SM) 的深度重构，目标是在微架构层面精确建模现代 NVIDIA GPU（Volta/Turing/Ampere 架构）的 SM 内部结构。相比原始 GPGPU-Sim 的单层 shader core 模型，本模块引入了 **Subcore 分区架构**、**多级指令缓存层次**、**基于 SASS trace 的依赖控制位建模**、**分体式寄存器文件** 等关键特性。

### 1.2 设计理念

- **层次化分解**：SM → Subcore → 各流水线阶段，镜像真实硬件的两级分区结构
- **Trace 驱动**：基于 NVBit 提取的增强 SASS trace（`traced_instruction` / `traced_operand`），精确还原操作数类型、控制位（stall/yield/wait barrier）
- **可配置性**：所有微架构参数通过 `shader_core_config` 统一配置，支持灵活的架构探索

### 1.3 代码规模

| 文件 | 行数 | 职责 |
|------|------|------|
| `sm.h` / `sm.cc` | 405 / 1997 | SM 顶层控制器 |
| `subcore.h` / `subcore.cc` | 206 / 1260 | Subcore 完整流水线 |
| `ldst_unit_sm.h` / `ldst_unit_sm.cc` | 392 / 2090 | SM 级共享访存单元 |
| `functional_unit.h` / `functional_unit.cc` | 226 / 560 | 功能单元层次结构 |
| `register_file.h` / `register_file.cc` | 203 / 519 | 分体式寄存器文件 |
| `ibuffer_remodeled.h` / `ibuffer_remodeled.cc` | 281 / 188 | 重构指令缓冲区 |
| `first_level_instruction_cache.h` / `first_level_instruction_cache.cc` | 126 / 343 | L0 指令缓存 |
| `l0_icnt.h` / `l0_icnt.cc` | 180 / 280 | L0↔L1 指令缓存互连 |
| `stream_buffer.h` / `stream_buffer.cc` | 169 / 300 | 指令预取流缓冲 |
| `warp_dependency_state.h` / `warp_dependency_state.cc` | 126 / 157 | Warp 依赖状态追踪 |
| `gmmu.h` / `gmmu.cc` | 59 / 107 | GPU 内存管理单元 |
| `page_table_walker.h` / `page_table_walker.cc` | 51 / 84 | 页表遍历器 |
| `new_stats.h` | 197 | 可扩展统计框架 |
| `fusedMemory/coalescingStats.h` / `coalescingStats.cc` | ~200 / ~400 | 访存合并统计 |

总计约 **10,500 行** C++ 代码。

### 1.4 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                          SM (core_t)                                │
│                                                                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│  │ Subcore 0│ │ Subcore 1│ │ Subcore 2│ │ Subcore 3│              │
│  │          │ │          │ │          │ │          │              │
│  │ Fetch    │ │ Fetch    │ │ Fetch    │ │ Fetch    │              │
│  │ Decode   │ │ Decode   │ │ Decode   │ │ Decode   │              │
│  │ Issue    │ │ Issue    │ │ Issue    │ │ Issue    │              │
│  │ Control  │ │ Control  │ │ Control  │ │ Control  │              │
│  │ Allocate │ │ Allocate │ │ Allocate │ │ Allocate │              │
│  │ Read RF  │ │ Read RF  │ │ Read RF  │ │ Read RF  │              │
│  │ Execute  │ │ Execute  │ │ Execute  │ │ Execute  │              │
│  │ Writeback│ │ Writeback│ │ Writeback│ │ Writeback│              │
│  │          │ │          │ │          │ │          │              │
│  │ L0I Cache│ │ L0I Cache│ │ L0I Cache│ │ L0I Cache│              │
│  │ L0C Cache│ │ L0C Cache│ │ L0C Cache│ │ L0C Cache│              │
│  │ IBuffer  │ │ IBuffer  │ │ IBuffer  │ │ IBuffer  │              │
│  │ Reg File │ │ Reg File │ │ Reg File │ │ Reg File │              │
│  │ FU(SP,   │ │ FU(SP,   │ │ FU(SP,   │ │ FU(SP,   │              │
│  │  INT,SFU │ │  INT,SFU │ │  INT,SFU │ │  INT,SFU │              │
│  │  TC,BR..)│ │  TC,BR..)│ │  TC,BR..)│ │  TC,BR..)│              │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘              │
│       │             │             │             │                    │
│  ─────┴─────────────┴─────────────┴─────────────┴──────             │
│                    SM 级共享资源                                      │
│  ┌─────────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐             │
│  │ ldst_unit_sm│ │ L0_icnt  │ │Scoreboard│ │  GMMU  │             │
│  │ (L1D/L1T/   │ │(L0↔L1I) │ │(RAW/WAW/ │ │(PTW×N) │             │
│  │  L1C/SMEM)  │ │          │ │  WAR)    │ │        │             │
│  │ PRT         │ │ L1I/L1C  │ │          │ │        │             │
│  │ InterWarp   │ │ Stream   │ │          │ │        │             │
│  │ Coalescing  │ │ Buffers  │ │          │ │        │             │
│  └─────────────┘ └──────────┘ └──────────┘ └────────┘             │
│  ┌──────────────────┐ ┌──────────────────┐                         │
│  │ Shared DP Unit   │ │ Barrier Set      │                         │
│  │ (可选跨Subcore共享)│ │ (CTA 级屏障)      │                         │
│  └──────────────────┘ └──────────────────┘                         │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.5 类继承关系

```
core_t (GPGPU-Sim 基类)
  └── SM                                    SM 顶层，继承 core_t

read_only_cache (GPGPU-Sim 缓存基类)
  └── first_level_instruction_cache         L0 指令缓存

mem_fetch_interface (GPGPU-Sim 接口)
  └── L0_icnt                               L0↔L1 互连

functional_unit (执行单元基类)
  ├── functional_unit_sfu                   SFU 特化（can_issue 覆写）
  └── functional_unit_shared_sm_part        SM 共享单元（多端口接收）
       └── ldst_unit_sm                     Load/Store 单元

Single_stat_abstract (统计接口)
  └── Single_stat_base
       ├── Single_stat_unsigned_long_long
       └── Single_stat_double
```

---

## 2. 核心组件详解

### 2.1 SM — 流多处理器顶层控制器

**文件**: `sm.h` / `sm.cc`

SM 类继承自 `core_t`，是整个模块的入口点。它管理所有 Subcore 实例和 SM 级共享资源。

**关键职责**:
- **每周期调度** (`cycle()`): 主仿真循环入口，按逆流水线顺序驱动各阶段
- **CTA 分配** (`issue_block2core()`): 将新的 CTA 分配到 SM，初始化 warp 状态
- **资源管理**: 管理 Scoreboard、barrier、线程上下文等 SM 级共享状态
- **Wait Barrier 处理**: 维护待处理的 barrier 增减操作栈

**执行顺序** (`SM::cycle()`):
```
1. ldst_unit_sm->cycle()          // 访存单元（含 L1D/L1T/L1C/SMEM 子流水线）
2. L0_icnt->cycle()               // 指令缓存互连
3. 共享 DP 单元 cycle             // 可选的跨 Subcore 共享 FP64 单元
4. 对每个 Subcore:
   4.1 writeback()                // 写回
   4.2 execute()                  // 执行
   4.3 read_rf()                  // 寄存器文件读取
   4.4 allocate()                 // 资源分配
   4.5 control_stage()            // 控制阶段
   4.6 issue()                    // 发射
   4.7 decode()                   // 译码
   4.8 fetch()                    // 取指
5. 处理 wait barrier 增减
6. 更新统计信息
```

**关键数据成员**:
- `m_subcores[]`: Subcore 实例数组（通常 4 个）
- `m_ldst_unit_sm`: SM 级共享访存单元
- `m_L0_icnt`: L0↔L1 指令缓存互连
- `m_scoreboard` / `m_scoreboard_WAR`: RAW/WAW 和 WAR 冒险检测
- `m_barriers`: CTA 级屏障集合
- `m_pending_wait_barrier_decrements/increments`: 延迟处理的 barrier 操作

**Dispatch Latch 枚举**:
```cpp
enum subcore_dispatch_latch_name_t {
  DISPATCH_SP,            // FP32
  DISPATCH_DP,            // FP64
  DISPATCH_HP,            // FP16
  DISPATCH_INT,           // 整数
  DISPATCH_SFU,           // 特殊函数
  DISPATCH_TENSOR_CORE,   // 张量核心
  DISPATCH_UNIFORM,       // 统一操作
  DISPATCH_BRANCH,        // 分支
  DISPATCH_MISCELLANEOUS, // NOP 等
  N_DISPATCH_LATCHES
};
```

---

### 2.2 Subcore — 子核心流水线

**文件**: `subcore.h` / `subcore.cc`

每个 Subcore 拥有完整的 8 级流水线，独立处理分配给它的 warp。Warp 以轮询方式静态分配到各 Subcore。

**8 级流水线**:

```
Fetch → Decode → Issue → Control → Allocate → Read_RF → Execute → Writeback
```

| 阶段 | 方法 | 职责 |
|------|------|------|
| **Fetch** | `fetch()` | 从 L0I 缓存取指，填充 IBuffer |
| **Decode** | `decode()` | 译码指令，提取操作数信息 |
| **Issue** | `issue()` | Warp 调度 + 冒险检测，选择就绪 warp 发射 |
| **Control** | `control_stage()` | 处理 stall/yield 控制位 |
| **Allocate** | `allocate()` | 分配寄存器文件读写端口 |
| **Read RF** | `read_rf()` | 从分体式寄存器文件读取操作数 |
| **Execute** | `execute()` | 在功能单元中执行 |
| **Writeback** | `writeback()` | 将结果写回寄存器文件 |

**Warp 调度策略**:
- Greedy-then-Highest-ID (`order_greedy_then_highest_id`): 优先选择贪心指针指向的 warp，然后按 dynamic_warp_id 降序
- 发射前检查: Scoreboard (RAW/WAW/WAR)、wait barrier、yield、stall counter、IBuffer 非空

**每 Subcore 私有资源**:
- `m_ibuffers[]`: 每 warp 一个 `IBuffer_Remodeled` 实例
- `m_dependency_states[]`: 每 warp 一个 `Dependency_State` 实例
- `m_register_files`: 4 种寄存器文件（regular, uniform, predicate, uniform-predicate）
- `m_functional_units[]`: SP, INT, DP, HP, SFU, Tensor Core, Branch, Uniform 功能单元
- `m_L0I`: L0 指令缓存
- `m_L0C`: L0 常量缓存
- `m_dispatch_latches[]`: 9 个 dispatch latch（对应 `N_DISPATCH_LATCHES`）

**关键方法**:
- `issue_warp()`: 将指令从 IBuffer 发射到对应功能单元的 dispatch latch
- `assign_warp_to_subcore()`: 将 warp 绑定到此 Subcore
- `is_wait_barriers_ready()`: 检查 DEPBAR/LDGDEPBAR 等 wait barrier 条件
- `get_fu()`: 根据指令类型选择目标功能单元

---

### 2.3 ldst_unit_sm — SM 级共享访存单元

**文件**: `ldst_unit_sm.h` / `ldst_unit_sm.cc`（全模块最大文件，2090 行）

ldst_unit_sm 是所有 Subcore 共享的集中式访存单元，继承自 `functional_unit_shared_sm_part`。它管理所有数据访存请求的生命周期。

**内部子流水线**:

```
                    ┌──→ L1D (全局内存)
                    │     ├── preTLB 队列 (per bank)
                    │     ├── GMMU/TLB
                    │     └── postTLB 队列 (per bank)
Subcore 请求 ──→ PRT ──→ L1T (纹理内存)
                    ├──→ L1C (常量内存)
                    ├──→ Shared Memory
                    ├──→ Bypass to L2
                    └──→ Miscellaneous
```

**核心数据结构**:

**PendingRequestTable (PRT)**:
- 跟踪所有正在处理的访存请求
- 每个 entry 关联一条 `warp_inst_t`，记录待解决的访存次数
- 当所有子访存完成时释放 entry，指令可写回

```cpp
class PendingRequestTableEntry {
    void assign_entry(shared_ptr<warp_inst_t> &inst);
    void decrement_num_pending_accesses_to_solve();
    void release();
    bool is_free();
    bool is_all_accesses_solved();
};
```

**AccessQueue**:
- 有界队列，用于各子流水线的请求缓冲
- 每个 L1D bank 有独立的 preTLB 和 postTLB 队列

**InterWarpCoalescingUnit**:
- 可选的跨 warp 访存合并
- 支持多种选择策略（通过 `interwarp_coalescing_selection_policy` 配置）

**仲裁机制**:
- `m_writeback_arb_icnt_and_subcores`: 写回端口的轮询仲裁
- `m_dispatch_subpipeline_arb_between_icnt_and_subcores`: 子流水线分发仲裁
- 每周期限制: 每种子流水线（shared/texture/constant）每周期最多分发一次

**关键流程**:
1. Subcore 通过 reception port 发送访存指令
2. `ldst_unit_sm::issue()` 接收并分配 PRT entry
3. 地址合并后生成 `mem_access_t`，分发到对应子流水线队列
4. 各子流水线独立处理（L1D 需经过 TLB/GMMU）
5. 缓存响应后递减 PRT entry 的待解决计数
6. 全部完成后释放 PRT entry，结果写回 Subcore

---

### 2.4 functional_unit — 功能单元层次结构

**文件**: `functional_unit.h` / `functional_unit.cc`

**基类 `functional_unit`**:
- 建模通用执行单元：延迟流水线 + 可选指令队列
- 支持固定延迟（fixed latency）和可变延迟两种模式
- 通过 `m_pipeline[]` 数组建模多周期延迟

**关键属性**:
```cpp
class functional_unit {
    register_set_uniptr *m_result_port;           // 结果输出端口
    Register_file *m_regular_rf;                   // 关联的寄存器文件
    register_set_uniptr *m_fixed_latency_rf_write_queue; // 固定延迟写回队列
    unsigned int m_max_latency;                    // 最大流水线延迟
    operation_pipeline_t m_type_of_pipeline;       // 流水线类型
    bool m_can_set_wait_barriers;                  // 是否可设置 wait barrier
    bool m_has_queue;                              // 是否有指令队列
    std::queue<warp_inst_t*> m_queue;              // 指令队列
    warp_inst_t **m_pipeline;                      // 延迟流水线数组
    std::vector<bool> m_latency_available;         // 延迟槽可用性
};
```

**派生类**:

| 类 | 特化行为 |
|----|---------|
| `functional_unit_sfu` | 覆写 `can_issue()`，SFU 特有的发射条件 |
| `functional_unit_shared_sm_part` | 多端口接收（`m_reception_ports`），跨 Subcore 共享 |
| `ldst_unit_sm` | 继承 shared_sm_part，完整的访存子系统 |

**执行流程**:
```
issue() → 指令进入 m_pipeline[0] 或 m_queue
cycle() → 流水线推进: m_pipeline[i] → m_pipeline[i+1]
        → 到达 m_pipeline[max_latency-1] 时输出到 m_result_port
        → 如有 wait barrier 设置需求，生成 barrier modifier
```

---

### 2.5 Register_file — 分体式寄存器文件

**文件**: `register_file.h` / `register_file.cc`

建模真实 GPU 的分体式（banked）寄存器文件，支持 bank 冲突检测和可选的 RF Cache。

**4 种寄存器文件类型**（每 Subcore 各一套）:
- **Regular**: 通用寄存器（R0-R255），32-bit per lane
- **Uniform**: 统一寄存器（UR0-UR63），所有线程共享同一值
- **Predicate**: 谓词寄存器（P0-P7），1-bit per lane
- **Uniform Predicate**: 统一谓词寄存器（UP0-UP7）

**Bank 冲突建模**:
```cpp
class Register_file_bank {
    unsigned int m_num_read_ports;       // 每 bank 读端口数
    unsigned int m_num_write_ports;      // 每 bank 写端口数
    // 按周期追踪端口占用情况
    std::vector<unsigned int> m_read_ports_used;   // [latency] → 已用读端口数
    std::vector<unsigned int> m_write_ports_used;  // [latency] → 已用写端口数
};
```

- 寄存器到 bank 的映射: `bank_id = reg_id % num_banks`
- 读请求通过 `is_possible_to_read_*()` 检查 bank 端口可用性
- 写请求通过 `is_rf_bank_write_port_available_at_given_cycle()` 预留未来周期的写端口

**RF Cache**（可选）:
- `Register_file_cache`: 缓存最近读取的寄存器值
- 命中时绕过 bank 端口，减少 bank 冲突
- 通过 `is_possible_to_read_cacheable()` / `allocate_reads_cacheable()` 使用

**读请求结构**:
```cpp
struct RF_instruction_read_request {
    bool m_is_possible_to_read;
    vector<set<unsigned int>> m_requested_reads;        // [bank] → 请求的寄存器集合
    vector<RF_cache_action> m_rf_cache_read_requests;   // RF cache 读请求
    vector<RF_cache_action> m_rf_cache_allocate_requests; // RF cache 分配请求
    unsigned int max_slack_due_to_double_use_of_banks;  // bank 复用产生的额外延迟
};
```

---

### 2.6 IBuffer_Remodeled — 重构指令缓冲区

**文件**: `ibuffer_remodeled.h` / `ibuffer_remodeled.cc`

替代原始 GPGPU-Sim 的 2-entry IBuffer，实现可配置深度的指令缓冲区，支持更深的取指预取和取指/发射解耦。

**核心设计**:
- 底层数据结构: `std::deque<IBuffer_Entry>`
- 每个 warp 拥有独立的 IBuffer 实例
- 支持配置化的缓冲区大小（`ibuffer_remodeled_size`）和取指宽度（`fetch_decode_width`）

**IBuffer Entry**:
```cpp
struct IBuffer_Entry {
    bool m_valid;           // 有效位
    address_type m_pc;      // 程序计数器
    warp_inst_t *m_inst;    // 指令指针
};
```

**关键方法**:
| 方法 | 职责 |
|------|------|
| `push_back()` | Decode 阶段将译码后的指令压入缓冲区尾部 |
| `front()` | Issue 阶段读取缓冲区头部指令 |
| `pop_front()` | 指令成功发射后弹出 |
| `flush()` | 分支预测失败时清空缓冲区 |
| `is_full()` / `is_empty()` | 流控信号 |
| `get_next_pc_to_fetch_request()` | 向 Fetch 阶段提供下一个取指地址 |
| `set_next_pc_to_fetch_request()` | 更新下一个取指地址 |

**取指解耦机制**:
- `m_next_pc_to_fetch_request`: 维护独立的取指 PC，与发射 PC 解耦
- `m_is_ret_reached`: 标记是否已到达返回指令，停止取指
- 支持多条指令同时在缓冲区中等待发射

---

### 2.7 指令缓存层次结构

#### 2.7.1 first_level_instruction_cache — L0 指令缓存

**文件**: `first_level_instruction_cache.h` / `first_level_instruction_cache.cc`

每个 Subcore 拥有私有的 L0 指令缓存，继承自 `read_only_cache`。

**特性**:
- **IB Coalescing**: 可选的指令缓冲区合并（`m_is_IB_coalescing_enabled`），合并来自同一 Subcore 不同 warp 的相同地址请求
- **Stream Buffer 预取**: 集成 `multiple_stream_buffers` 进行指令预取
- **两种访问状态追踪**:
  - 有 IB Coalescing: `map<addr, status_element>` — 按地址去重
  - 无 IB Coalescing: `map<warp_id, map<addr, status_element>>` — 按 warp 分别追踪

**访问流程**:
```
1. access() → 查询 tag array
2. HIT → 直接返回数据
3. MISS → 检查 stream buffer
   3a. Stream buffer HIT → fill_from_stream_buffer()
   3b. Stream buffer MISS → 通过 L0_icnt 向 L1I 发请求
4. fill() → L1I 响应后填充 cache line
```

#### 2.7.2 L0_icnt — L0↔L1 指令缓存互连

**文件**: `l0_icnt.h` / `l0_icnt.cc`

建模 L0 指令缓存与 L1 指令缓存之间的互连网络，实现 `mem_fetch_interface` 接口。

**双向队列**:
```
L0 → L1 方向: m_icnt_to_L1_queue[port][cycle_latency]
L1 → L0 方向: m_L1_to_icnt_queue[port][entries]
```

- 可配置的请求/响应端口数和延迟
- 支持 TLB 旁路队列 (`m_icnt_L1_TLB_to_cache`)
- `cycle()` 方法每周期推进队列，处理请求和响应

#### 2.7.3 stream_buffer — 指令预取流缓冲

**文件**: `stream_buffer.h` / `stream_buffer.cc`

实现基于流的指令预取机制，每个 L0I 缓存关联多个 stream buffer。

**两级结构**:
- `single_stream_buffer`: 单个流缓冲，追踪一个连续的预取流
- `multiple_stream_buffers`: 管理多个 stream buffer 实例，提供统一的搜索和分配接口

**single_stream_buffer 状态**:
```cpp
class single_stream_buffer {
    bool m_is_currently_prefetching;          // 是否正在预取
    new_addr_type m_next_addr_to_prefetch;    // 下一个预取地址
    unsigned int m_current_unique_function_id; // 当前函数 ID（用于流识别）
    queue<new_addr_type> m_queue_ordered_prefetches;  // 有序预取队列
    map<addr, prefetch_element> m_all_prefetches;     // 所有预取条目
};
```

**预取流程**:
1. L0I miss 时，`search()` 在所有 stream buffer 中查找
2. 如果命中已有流 → 直接使用预取数据
3. 如果未命中 → `set_new_stream()` 分配新的 stream buffer
4. `do_prefetch()` 按顺序地址生成预取请求
5. `fill()` 接收预取响应，`send_to_cache()` 将就绪数据送入 L0I

---

### 2.8 Dependency_State — Warp 依赖状态追踪

**文件**: `warp_dependency_state.h` / `warp_dependency_state.cc`

建模 NVIDIA SASS 指令中的控制位依赖机制，这是区别于传统 Scoreboard 的关键特性。

**三种依赖机制**:

| 机制 | 说明 | 对应 SASS 控制位 |
|------|------|-----------------|
| **Stall Counter** | 指令发射后强制等待固定周期数 | `stall` 字段 |
| **Yield** | 让出调度权一个周期 | `yield` 位 |
| **Wait Barriers** | 等待特定 barrier 计数器归零 | `DEPBAR` / `LDGDEPBAR` |

**Wait Barrier 机制**:
```cpp
class Wait_Barrier {
    unsigned int m_counter;      // 计数器（递增/递减）
    unsigned int m_barrier_id;   // Barrier ID (0-5)
    bool is_ready(unsigned int min_val);  // 计数器 ≤ min_val 时就绪
};
```

- 每个 warp 拥有多个 wait barrier（通常 6 个，ID 0-5）
- 长延迟指令（如全局内存加载）发射时递增对应 barrier 计数器
- 指令完成时递减计数器
- 后续依赖指令通过 `DEPBAR` 等待计数器降至指定阈值

**Wait_Barrier_Entry_Modifier**:
```cpp
struct Wait_Barrier_Entry_Modifier {
    unsigned int sm_warp_id;
    unsigned int barrier_id;
    Wait_Barrier_Type barrier_type;    // READ / WRITE
    Wait_Barrier_Action barrier_action; // INCREASE / DECREASE
    new_addr_type pc;
};
```

- Barrier 的增减操作通过 SM 级的栈延迟处理，确保时序正确

**LDGSTS 追踪**:
- `m_num_pending_ldgsts`: 追踪异步全局到共享内存拷贝（LDGSTS）的待完成数
- `are_ldgsts_pending()`: 检查是否有未完成的 LDGSTS 操作

---

### 2.9 GMMU — GPU 内存管理单元

**文件**: `gmmu.h` / `gmmu.cc` / `page_table_walker.h` / `page_table_walker.cc`

建模 GPU 的地址翻译硬件，位于 L1D 缓存的 TLB miss 路径上。

**架构**:
```
请求 → FIFO_in → [PTW_0, PTW_1, ..., PTW_N-1] → FIFO_out → 返回
```

**PageTableWalker**:
```cpp
class PageTableWalker {
    unsigned int m_num_cycles_to_solve;  // 页表遍历延迟
    unsigned int m_remaining_cycles;     // 剩余周期
    mem_fetch *m_mf;                     // 当前处理的请求
    bool m_is_busy;                      // 忙碌标志
};
```

- 可配置的 PTW 数量（并行度）和遍历延迟
- FIFO_in/FIFO_out 有界队列提供流控
- `cycle()` 每周期推进所有 PTW 的状态

---

### 2.10 new_stats — 可扩展统计框架

**文件**: `new_stats.h`

提供类型安全的统计收集框架，支持动态注册和按名查询。

**类层次**:
```
Single_stat_abstract (纯虚接口)
  └── Single_stat_base (公共字段: name, suffix, reset 策略)
       ├── Single_stat_unsigned_long_long (整数计数器)
       └── Single_stat_double (浮点累加器)

Element_stats (统计容器)
  ├── m_stats_map: map<string, shared_ptr<Single_stat_abstract>>
  └── m_stats_name: vector<string> (保持插入顺序)
```

**特性**:
- 支持 SM 级聚合后擦除（`is_erase_after_gather_in_sm`）
- 可选的重置策略（`is_reset_allowed`）
- 区分 SM 级统计和 Subcore 级统计

---

### 2.11 coalescingStats — 访存合并统计

**文件**: `fusedMemory/coalescingStats.h` / `fusedMemory/coalescingStats.cc`

追踪和分析访存合并效率，支持 per-SM 和跨 SM 的统计聚合。

**主要类**:
- `coalescingAddressStats`: 单个访存子系统（L1D/L1C/SMEM）的合并统计
- `coalescingStatsPerSm`: 单个 SM 的所有访存子系统统计
- `coalescingStatsAcrossSms`: 全局跨 SM 统计聚合

---

## 3. 关键数据流

### 3.1 指令生命周期

```
                    ┌─────────────────────────────────────────────┐
                    │              Subcore 流水线                   │
                    │                                             │
L1I ← L0_icnt ← L0I ← Fetch                                     │
                    │     ↓                                       │
                    │   Decode → IBuffer                          │
                    │              ↓                               │
                    │            Issue (Scoreboard + Dep State 检查)│
                    │              ↓                               │
                    │           Control (stall/yield 处理)          │
                    │              ↓                               │
                    │           Allocate (RF 端口预留)              │
                    │              ↓                               │
                    │           Read RF (从 banked RF 读操作数)     │
                    │              ↓                               │
                    │           Execute (FU 流水线)                 │
                    │              ↓                               │
                    │           Writeback (结果写回 RF)             │
                    └─────────────────────────────────────────────┘
```

### 3.2 访存请求生命周期

```
Subcore Execute → ldst_unit_sm.issue()
                       ↓
                  PRT 分配 entry
                       ↓
                  地址合并 (intra-warp coalescing)
                       ↓
              ┌── 可选: InterWarp Coalescing ──┐
              ↓                                ↓
         分发到子流水线队列:
         ├── L1D: preTLB → GMMU/TLB → postTLB → L1D cache
         ├── L1T: queue → L1T cache
         ├── L1C: queue → L1C cache
         ├── SMEM: queue → shared memory
         └── Bypass: queue → L2
                       ↓
                  缓存响应 → PRT entry 递减
                       ↓
                  全部完成 → PRT 释放 → Writeback
```

### 3.3 Wait Barrier 数据流

```
长延迟指令发射 (Issue)
    ↓
SM::m_pending_wait_barrier_increments.push(barrier_id, INCREASE)
    ↓
SM::cycle() 末尾处理 → Dependency_State::increase_counter()
    ↓
指令完成 (Writeback / 缓存响应)
    ↓
SM::m_pending_wait_barrier_decrements.push(barrier_id, DECREASE)
    ↓
SM::cycle() 末尾处理 → Dependency_State::decrease_counter()
    ↓
后续指令 Issue 阶段检查 → is_wait_barriers_ready()
    ↓
barrier 计数器 ≤ 阈值 → 允许发射
```

---

## 4. 关键设计决策

### 4.1 Subcore 分区架构

Warp 以轮询方式静态分配到 Subcore（通常 4 个），每个 Subcore 拥有完全独立的流水线。这镜像了 NVIDIA Volta/Turing/Ampere 架构中的 sub-partition 设计，其中每个 SM 被划分为 4 个处理块，每块有独立的 warp 调度器、寄存器文件和执行单元。

### 4.2 基于控制位的依赖追踪

除传统 Scoreboard（RAW/WAW/WAR）外，引入了基于 SASS 控制位的依赖机制（stall counter、yield、wait barrier）。这些控制位从真实 GPU 的 SASS 二进制中提取，能更精确地建模编译器插入的依赖信息，避免了纯硬件 Scoreboard 的保守性。

### 4.3 分体式寄存器文件 + RF Cache

寄存器文件按 bank 划分，精确建模 bank 冲突对流水线的影响。可选的 RF Cache 缓存最近读取的寄存器值，减少重复读取对 bank 端口的压力——这与 NVIDIA 专利中描述的 operand collector 机制类似。

### 4.4 两级指令缓存层次

每 Subcore 私有 L0I 缓存 + SM 共享 L1I 缓存，通过建模的互连（L0_icnt）连接。配合 stream buffer 预取，精确建模指令供给带宽对性能的影响。

### 4.5 集中式访存单元 + PRT

所有 Subcore 的访存请求汇聚到共享的 ldst_unit_sm，通过 PendingRequestTable 追踪每条访存指令的所有子请求。这种设计允许建模跨 Subcore 的访存合并和共享缓存端口竞争。

### 4.6 Trace 驱动仿真

整个架构设计围绕增强 SASS trace（通过 NVBit 提取）展开，使用 `traced_instruction` 和 `traced_operand` 类型携带精确的操作数类型、寄存器编号、控制位等信息。这使得仿真器能够在不需要完整 ISA 解码器的情况下精确建模微架构行为。

---

## 5. 配置参数索引

所有配置通过 `shader_core_config` 传入，关键参数分类如下:

| 类别 | 参数 | 说明 |
|------|------|------|
| **SM 结构** | `num_subcores` | Subcore 数量（通常 4） |
| | `max_warps_per_shader` | SM 最大 warp 数 |
| | `warp_size` | Warp 大小（32） |
| **IBuffer** | `ibuffer_remodeled_size` | 重构 IBuffer 深度 |
| | `fetch_decode_width` | 每周期取指/译码宽度 |
| | `is_ibuffer_remodeled_enabled` | 是否启用重构 IBuffer |
| **寄存器文件** | `num_banks` | RF bank 数量 |
| | `num_read_ports_per_bank` | 每 bank 读端口数 |
| | `num_write_ports_per_bank` | 每 bank 写端口数 |
| | RF cache 相关参数 | RF Cache 启用/大小 |
| **功能单元** | 各 FU 的 latency/initiation interval | 执行延迟和发射间隔 |
| | `is_fp32_and_int_unified_pipeline` | FP32/INT 是否共享流水线 |
| | `is_dp_pipeline_shared_for_subcores` | FP64 是否跨 Subcore 共享 |
| **访存** | `m_L1D_config` / `m_L1T_config` / `m_L1C_config` | 各级缓存配置 |
| | PRT size | PendingRequestTable 大小 |
| | `is_interwarp_coalescing_enabled` | 跨 warp 合并开关 |
| | `interwarp_coalescing_selection_policy` | 合并选择策略 |
| **依赖** | `num_wait_barriers_per_warp` | 每 warp wait barrier 数量 |
| | `is_remodeling_scoreboarding_enabled` | 是否启用重构 Scoreboard |
| | `predicate_latency` | 谓词延迟 |
| **指令缓存** | L0I/L1I 缓存配置 | 大小/关联度/行大小 |
| | Stream buffer 参数 | 数量/大小/每周期预取数 |
| | L0_icnt 延迟/端口数 | 互连参数 |

---

## 6. 文件依赖关系图

```
sm.h
  ├── subcore.h
  │     ├── register_file.h
  │     └── (functional_unit.h via sm.h)
  ├── ldst_unit_sm.h
  │     └── functional_unit.h
  ├── new_stats.h
  └── (外部依赖)
       ├── abstract_hardware_model.h
       ├── shader.h / shader_core_wrapper.h
       └── constants.h

first_level_instruction_cache.h
  └── gpu-cache.h (read_only_cache 基类)

l0_icnt.h
  ├── abstract_hardware_model.h (mem_fetch_interface)
  └── constants.h

stream_buffer.h
  ├── abstract_hardware_model.h
  └── gpu-cache.h

ibuffer_remodeled.h
  └── abstract_hardware_model.h

warp_dependency_state.h
  └── abstract_hardware_model.h

gmmu.h
  └── page_table_walker.h

functional_unit.h
  └── traced_instruction.h (NVBit trace 格式)
```
