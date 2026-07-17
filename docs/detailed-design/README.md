# Detailed Design Index

Numbered developer-implementation-grade design docs for the remodeled SM
timing model (`simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/`).
For a short orientation before diving in, start at [`arch.md`](../../arch.md).

| Doc | Description |
| --- | --- |
| [00A-接口规范与映射约定.md](00A-接口规范与映射约定.md) | Writing baseline for this doc set: data types, interface protocol, granularity annotation, and the per-module chapter template. |
| [00-关键路径时序图.md](00-关键路径时序图.md) | End-to-end timing diagrams for the three core execution paths, for orientation before reading individual chapters. |
| [01-总览与假设集合.md](01-总览与假设集合.md) | Overview and the True-Path assumption set (13 boolean config flags), with paper-alignment validation status per flag. |
| [02-SM顶层设计.md](02-SM顶层设计.md) | SM top-level design: subcore array, SM-shared structures, per-cycle execution order. |
| [02A-SM论文对齐架构图详解-前端与执行.md](02A-SM论文对齐架构图详解-前端与执行.md) | Paper-aligned SM architecture diagram, front-end and execution side, in detail. |
| [02B-SM论文对齐架构图详解-访存与共享路径.md](02B-SM论文对齐架构图详解-访存与共享路径.md) | Paper-aligned SM architecture diagram, memory and shared-path side, in detail. |
| [03-Subcore流水线设计.md](03-Subcore流水线设计.md) | Subcore pipeline design overview: specification, class structure, unified entry point. |
| [03A-Subcore前端流水与调度设计.md](03A-Subcore前端流水与调度设计.md) | Subcore front-end detail: Fetch / Decode / Issue / Control stage implementation, state machines, stall propagation. |
| [03B-Subcore后端执行与写回设计.md](03B-Subcore后端执行与写回设计.md) | Subcore back-end detail: Allocate / Read_RF / Execute / Writeback resource modeling and execution semantics. |
| [03C-Subcore后端补充细节与诊断.md](03C-Subcore后端补充细节与诊断.md) | Subcore back-end supplementary detail: RF port windows, FU internal execution semantics, shared-pipeline throttling and backpressure diagnostics. |
| [04-依赖模型设计.md](04-依赖模型设计.md) | Dependency model design: control-bit stall counter, yield, and wait-barrier semantics. |
| [05-指令供给子系统设计.md](05-指令供给子系统设计.md) | Instruction supply subsystem overview: IBuffer and L0I design. |
| [05A-L0互连与流式预取子系统设计.md](05A-L0互连与流式预取子系统设计.md) | L0_icnt interconnect and stream-buffer prefetch subsystem, with interface timing. |
| [05B-Fetch-Decode实现与源码锚点.md](05B-Fetch-Decode实现与源码锚点.md) | Subcore Fetch/Decode stage implementation detail, key circuit description, stall-condition summary, and source anchors. |
| [06-执行资源设计.md](06-执行资源设计.md) | Execution resource design: functional units, register file, RF cache, result queues. |
| [07-共享访存单元设计.md](07-共享访存单元设计.md) | Shared memory unit design: `ldst_unit_sm`, Pending Request Table, interwarp coalescing. |
| [08-配置矩阵.md](08-配置矩阵.md) | Configuration matrix reference for the SM/subcore/RF/cache parameters. |
| [09-源码锚点索引.md](09-源码锚点索引.md) | Source anchor index: function-level cross-reference from design concepts to `remodeling/*.cc/.h`, including the `SM::cycle()` call chain. |

All source paths referenced by these docs are relative to
`simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/` unless stated
otherwise in the individual chapter.
