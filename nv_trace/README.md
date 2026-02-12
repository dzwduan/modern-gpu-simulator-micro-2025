c# NV Trace - Standalone GPU Trace Extraction Tool

A standalone NVBit-based GPU instruction trace extraction tool, extracted and adapted from the [modern-gpu-simulator](https://github.com/your-repo/modern-gpu-simulator-micro-2025) project. This tool captures detailed SASS instruction traces from real NVIDIA GPU execution using NVBit binary instrumentation.

## Supported GPUs

- **Ada Lovelace** (RTX 4090, sm_89)
- **Hopper** (H100, sm_90)
- **Ampere** (A100/RTX 3090, sm_80/86)
- **Turing** (RTX 2080, sm_75)
- **Volta** (V100, sm_70)
- **Pascal** (GTX 1080/P100, sm_60/61)
- **Kepler** (sm_35)

## Prerequisites

- NVIDIA GPU (see supported list above)
- CUDA Toolkit >= 10.1 (nvcc, cuobjdump, nvdisasm)
- Protocol Buffers compiler (protoc) and library (libprotobuf)
- g++ with C++17 support
- wget (for NVBit download)
- zlib (for compressed traces)

### Install dependencies (Ubuntu/Debian)

```bash
sudo apt-get install -y protobuf-compiler libprotobuf-dev zlib1g-dev wget
```

## Quick Start

### 1. Install NVBit

```bash
# Default: install NVBit v1.7.7.1 (recommended)
make install_nvbit

# Or specify a version
make install_nvbit NVBIT_VERSION=1.7.6
make install_nvbit NVBIT_VERSION=1.7.7.1
```

Supported NVBit versions:
- **v1.7.7.1** (default) - CUDA 13.1 headers, SM_120 (Blackwell) support, channel.hpp hotfix
- **v1.7.6** - CUDA 13.0 headers, SM_110 support

### 2. Build the tracer

```bash
make -j
```

### 3. Trace an application

**Method 1: Using the run script (recommended)**

```bash
# Basic usage
./run_trace.py --app /path/to/cuda_app --args "arg1 arg2" --device 0

# With kernel limits (trace only kernels 1-10)
./run_trace.py --app /path/to/cuda_app --kernel-start 1 --kernel-end 10

# Custom output directory
./run_trace.py --app /path/to/cuda_app --output /path/to/output

# Compressed mode with post-processing
./run_trace.py --app /path/to/cuda_app --compressed
```

**Method 2: Direct LD_PRELOAD (manual)**

```bash
export CUDA_VISIBLE_DEVICES=0
LD_PRELOAD=./tracer_tool/tracer_tool.so /path/to/cuda_app [args...]
```

The traces will be generated in the `traces/` folder in the current working directory.

### 4. Inspect traces (optional)

```bash
./tracer_tool/trace_printer traces/dynamic_trace.pb
```

## Output Format

The tracer generates the following outputs in the traces directory:

| File/Directory | Description |
|---|---|
| `dynamic_trace.pb` | Protocol Buffer binary containing the dynamic execution trace (device -> stream -> kernel -> threadblock -> warp -> instruction hierarchy) |
| `extra_info/enhanced_execution_info.json` | JSON file with static kernel information (SASS instructions, operands, register usage, control bits) |
| `threadblocks/` | Per-threadblock Protocol Buffer trace files organized by device/stream/kernel |
| `stats.csv` | Statistics for each traced kernel (grid dims, block dims, instruction counts) |

### Trace Data Structure

The dynamic trace uses Protocol Buffers with this hierarchy:

```
Trace
  └── gpu_device (per GPU)
       └── cuda_stream (per CUDA stream)
            ├── kernel (per kernel launch)
            │   ├── grid_dim, block_dim
            │   ├── shared_memory, registers
            │   └── function_unique_id
            └── ordered_cuda_events (kernel launches + memcpy)

threadblock (separate files per CTA)
  └── warp (per warp in the threadblock)
       └── instruction (per dynamic instruction)
            ├── pc, active_mask, predicate_mask
            └── addresses (memory addresses with compression)
```

The enhanced static trace (JSON) provides:
- Per-instruction: opcode, operands, control bits, register usage
- Per-kernel: architecture version, function address, instruction list

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | - | GPU device to trace on |
| `DYNAMIC_KERNEL_LIMIT_START` | 0 | Start tracing from this kernel ID (0 = beginning) |
| `DYNAMIC_KERNEL_LIMIT_END` | 0 | Stop after this kernel ID (0 = no limit) |
| `TERMINATE_UPON_LIMIT` | 0 | Exit process when kernel limit reached |
| `ACTIVE_FROM_START` | 1 | Set to 0 to use cuProfilerStart/Stop for region selection |
| `TOOL_VERBOSE` | 0 | Enable verbose tracer output |
| `TRACES_FOLDER` | ./traces | Custom output directory (requires `USER_DEFINED_FOLDERS=1`) |
| `USER_DEFINED_FOLDERS` | 0 | Use TRACES_FOLDER path |
| `INTERMEDIATE_EXTRA_FILES_PERSISTANCE` | 0 | Keep intermediate cubin/sass/rfu files |
| `THRESHOLD_UNIQUE_KERNEL_CHECKING` | 10 | Instructions used to distinguish same-named kernels |
| `EXCLUDE_PRED_OFF` | 1 | Exclude predicated-off instructions |

## Project Structure

```
nv_trace/
├── Makefile                     # Top-level build
├── install_nvbit.sh             # NVBit installer
├── run_trace.py                 # Trace runner script
├── README.md
├── ISA_Def/                     # NVIDIA ISA opcode definitions
│   ├── operation_type.h         # uarch operation types
│   ├── trace_opcode.h           # SASS opcode enum
│   ├── ampere_opcode.h          # Ampere opcode map
│   ├── volta_opcode.h           # Volta opcode map
│   ├── turing_opcode.h          # Turing opcode map
│   ├── pascal_opcode.h          # Pascal opcode map
│   ├── kepler_opcode.h          # Kepler opcode map
│   ├── hopper_opcode.h          # Hopper opcode map
│   └── blackwell_opcode.h       # Blackwell opcode map
├── traces_enhanced/             # Enhanced trace data library
│   ├── Makefile
│   ├── dynamic_trace/           # Protocol Buffer definitions
│   │   ├── trace.proto
│   │   ├── gpu_device.proto
│   │   ├── cuda_stream.proto
│   │   ├── kernel.proto
│   │   ├── threadblock.proto
│   │   ├── warp.proto
│   │   ├── instruction.proto
│   │   ├── address.proto
│   │   └── dim3d.proto
│   └── src/                     # C++ trace data structures
│       ├── traced_execution.h/cc
│       ├── traced_kernel.h/cc
│       ├── traced_instruction.h/cc
│       ├── traced_operand.h/cc
│       ├── traced_constants.h
│       ├── register_usage.h/cc
│       ├── control_bits.h/cc
│       ├── string_utilities.h/cc
│       ├── JSONBase.h/cc
│       └── rapidjson/           # JSON library (header-only)
├── tracer_tool/                 # NVBit tracer instrumentation
│   ├── Makefile
│   ├── tracer_tool.cu           # Main NVBit tool (host-side)
│   ├── inject_funcs.cu          # Device-side instrumentation
│   ├── common.h                 # Shared data structures
│   ├── trace_printer.cc         # Trace viewer utility
│   └── traces-processing/       # Post-processing tools
│       ├── Makefile
│       ├── post-traces-processing.cpp
│       └── post-traces-processing-compressed.cpp
└── others/                      # Additional NVBit analysis tools
    ├── spinlock_tool/           # Spinlock/non-deterministic instruction detection
    ├── bbv_tool/                # Basic Block Vector analysis
    │   ├── bbv_count/           # Per-warp BBV
    │   └── bbv_count_tb/        # Per-threadblock BBV
    ├── occupancy_calc_tool/     # GPU occupancy calculator
    └── silicon_checkpoint_tool/ # GPU memory state checkpointing
```

## Additional Tools

Build all additional tools:

```bash
make others
```

Or build individually:

```bash
make spinlock_tool
make bbv_tool
make occupancy_calc_tool
make silicon_checkpoint_tool
```

### Spinlock Tool

Detects non-deterministic (spinlock) instructions by comparing instruction execution histograms across two runs:

```bash
SPINLOCK_PHASE=0 LD_PRELOAD=./others/spinlock_tool/spinlock_tool.so /path/to/cuda_app
SPINLOCK_PHASE=1 LD_PRELOAD=./others/spinlock_tool/spinlock_tool.so /path/to/cuda_app
```

Output: `spinlock_detection/spinlock_instructions.txt`

### BBV Tool

Basic Block Vector analysis for kernel characterization:

```bash
# Per-warp BBV
LD_PRELOAD=./others/bbv_tool/bbv_count/bbv_count.so /path/to/cuda_app

# Per-threadblock BBV
LD_PRELOAD=./others/bbv_tool/bbv_count_tb/bbv_count_tb.so /path/to/cuda_app
```

### Occupancy Calculator

Reports maximum active blocks per SM for each kernel:

```bash
LD_PRELOAD=./others/occupancy_calc_tool/occupancy_calc/occupancy_calc.so /path/to/cuda_app
```

### Silicon Checkpoint Tool

Dumps GPU memory state after each kernel execution:

```bash
LD_PRELOAD=./others/silicon_checkpoint_tool/checkpoint/checkpoint.so /path/to/cuda_app
```

## RTX 4090 Notes

The RTX 4090 uses the Ada Lovelace architecture (sm_89). When building:

```bash
# The Makefile defaults to ARCH=all which builds for all architectures.
# To build specifically for RTX 4090:
export ARCH=sm_89
make
```

The tracer will automatically detect the GPU architecture at runtime and select the appropriate opcode map (Ampere_OpcodeMap for sm_89).

## License

BSD 3-Clause License. See source files for full license text.

Original work by Rodrigo Huerta, Mojtaba Abaie Shoushtary, Josep-Llorenç Cruz, Antonio González (Universitat Politecnica de Catalunya) and Mahmoud Khairy (Purdue University).
