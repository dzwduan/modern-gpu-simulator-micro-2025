# Welcome to repository modern-gpu-simulator-micro-2025

The structucture of this repository is the following one:
Simulator files in: `./simulator-remodeled`
Absolute Percentage Error for all the configurations and applications in: `./APEs`.


> [!IMPORTANT]  
> Main features of the simulator compared to Accel-sim

1. Redesigned SM model, including sub-core pipeline and memory pipeline.
2. Tracer that parses control bits.
3. Simulator that interprets control bits.
4. Configurable dependence handling: scoreboards or control bits.
5. Enhanced scoreboard detects dependencies in uniform, predicate, and uniform-predicate registers.
6. Additional scoreboard to protect against WAR hazards.
7. Correct per-kernel/function instruction addresses to prevent aliasing in memory requests.
8. Fix for non-contiguous traced instruction fetches causing false-positive I-cache hits.
9. Corrected fetch and decode stage timing (no longer both in a single cycle).
10. Fetch and decode now are integrated into the sub-core model properly.
11. Added L0 instruction cache.
12. Added stream-buffer instruction prefetcher.
13. Parallelized simulator with OpenMP.
14. AccelWattch energy reporting integrated.
15. Added static instruction metadata extraction, stored into JSON.
16. Traces stored using Google Protocol Buffers.

> [!IMPORTANT]
> This repository contains two major improvements to the Accel-Sim framework.
> Please cite the following resources appropriately.

First, an enhanced version of the simulator that models the architecture described in our MICRO 2025 paper. If you use any files related to this architectural model, please cite:

```
Rodrigo Huerta, Mojtaba Abaie Shoushtary, José-Lorenzo Cruz, Antonio González,
Dissecting and Modeling the Architecture of Modern GPU Cores,
in 2025 IEEE/ACM International Symposium on Microarchitecture (MICRO)
```

Second, this repository includes the implementation for parallelizing the simulator, described in ISPASS 2025 and CAMS 2024. If you use any of these files, please cite:

```
Rodrigo Huerta, Antonio González,
GPU Simulation Acceleration via Parallelization,
in 2025 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)
```

```
Rodrigo Huerta, Antonio González,
Parallelizing a modern GPU simulator,
in arXiv:2401.10082 
```

Finally, if you use this simulator, please also cite the Accel-Sim paper:

```
Mahmoud Khairy, Zhensheng Shen, Tor M. Aamodt, Timothy G. Rogers,
Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling,
in 2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA)
```

## Dependencies

This simulator builds on the original Accel-Sim. It requires all upstream [dependencies](https://github.com/accel-sim/accel-sim-framework/blob/main/README.md) plus Google Protocol Buffers (`sudo apt install protobuf-compiler` on Debian/Ubuntu).

Tested platforms:
- Ubuntu 20.04.6, 22.04.5, and 24.04
- g++/gcc ≤ 11 (CUDA 11.4 requires g++/gcc 9)
- CUDA 11.4 and CUDA 12.8

Note: Newer g++ versions may fail with RapidJSON.

## Simulator Components

> [!IMPORTANT]
> From here on, assume your working directory is `./simulator-remodeled/`.

1. **Tracer**: An NVBit tool for generating SASS traces from CUDA applications. While the implementation differs, usage is similar to the Accel-Sim tracer. Code lives in `./util/tracer_nvbit/`.

   ```bash
   export CUDA_INSTALL_PATH=<path-to-your-cuda>
   export PATH=$CUDA_INSTALL_PATH/bin:$PATH
   ./util/tracer_nvbit/install_nvbit.sh
   make -C ./util/tracer_nvbit/
   ```

   ---

   The following example demonstrates tracing the GPU_Microbenchmark suite. The
   application collection (`gpu-app-collection`) lives at the repository root
   under `../nv_trace/`, one level above `simulator-remodeled/`:

   ```bash
   # Ensure CUDA_INSTALL_PATH is set and PATH includes nvcc

   # Get applications, data files, and build them
   source ../nv_trace/gpu-app-collection/src/setup_environment
   make -j -C ../nv_trace/gpu-app-collection/src GPU_Microbenchmark \
       CUOPTS='-gencode=arch=compute_89,code="sm_89,compute_89"'

   # Run applications with the tracer (requires a real GPU)
   ./util/tracer_nvbit/run_hw_trace.py -B GPU_Microbenchmark -D <gpu-device-num>
   ```

   Traces will be generated in `./hw_run/traces/device-<device-num>/<cuda-version>/`.
   Important: Applications must be compiled using static libraries; otherwise, extracting static information from cubins may fail. Example Rodinia 2 traces for Turing, Ampere, and Blackwell are provided in `./exampleTraces/`. Uncompress with:
   `tar -xzvf <trace-archive>.tar.gz`

   Trace format:

   ```bash
   # Static metadata for all executed instructions in JSON
   ./app_name/app_parameters/traces/extra_info/enhanced_execution_info.json

   # Dynamic trace (Protocol Buffers)
   ./app_name/app_parameters/traces/dynamic_trace.pb

   # Per-kernel/threadblock dynamic info (e.g., per-warp PCs, memory addresses)
   ./app_name/app_parameters/traces/threadblocks
   ```

2. **Simulator**: The simulator consumes SASS traces. To build it:

   ```bash
   source ./gpu-simulator/setup_environment_no_git.sh
   make -j -C ./gpu-simulator/
   ```

   Optional: to additionally generate a `compile_commands.json` for editor/clangd
   tooling, wrap the build with `bear` instead of calling `make` directly
   (a plain `make -j -C ./gpu-simulator/` is sufficient otherwise):

   ```bash
   source ./gpu-simulator/setup_environment_no_git.sh
   make clean -C ./gpu-simulator/
   bear -- make -j -C ./gpu-simulator/
   ```

   This will produce an executable in:

   ```bash
   ./gpu-simulator/bin/release/accel-sim.out
   ```

   Running the simple example from item 1:

    ```bash
    ./util/job_launching/run_simulations.py \
       -B rodinia_2.0-ft \
       -C RTX3080-Accelwattch_SASS_SIM \
       -T ./hw_run/traces/device-<device-num>/<cuda-version>/ \
       -N myTestName
    ```

   After the jobs finish, collect stats with:

   ```bash
   ./util/job_launching/get_stats.py -N myTestName | tee stats.csv
   ```

   To run `accel-sim.out` directly for a specific workload:

    ```bash
    ./gpu-simulator/bin/release/accel-sim.out \
       -trace ./hw_run/Ampere/rodinia2/12.8/backprop-rodinia-2.0-ft/4096___data_result_4096_txt/traces/dynamic_trace.pb \
       -config ./gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTX3080/gpgpusim.config \
       -config ./gpu-simulator/configs/tested-cfgs/SM86_RTX3080/trace.config
    ```

   However, we encourage using the workload launch manager `run_simulations.py` as shown above, especially on clusters with SLURM.

   For a quick single-trace run with the OpenMP-parallel simulator (see feature
   13 above), run `accel-sim.out` directly with `OMP_NUM_THREADS` set:

   ```bash
   OMP_NUM_THREADS=32 OMP_PROC_BIND=spread ./gpu-simulator/bin/release/accel-sim.out \
       -config ./gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM89_RTX4090/gpgpusim.config \
       -config ./gpu-simulator/configs/tested-cfgs/SM89_RTX4090/trace.config \
       -trace ./hw_run/traces/device-0/12.6/l1_shared_bw/NO_ARGS/traces/dynamic_trace.pb
   ```

   Application definitions live in `./util/job_launching/apps/define-all-apps.yml`. Each application in each batch can configure RAM, CPU cores, and queue type to better match execution requirements and improve SLURM efficiency.

## Testing

> [!IMPORTANT]
> Run these commands from the repository root. The regression harness resolves
> and runs the simulator binary at
> `simulator-remodeled/gpu-simulator/bin/release/accel-sim.out`, so build the
> simulator first (see Simulator Components).

Unit tests:

```bash
python3 -m unittest discover -s tests -v
```

Remodeled trace regression harness:

```bash
# Validate the case manifest and list runnable cases
python3 tests/remodeled_trace/run_regression.py list

# Run every case and record deterministic observations
python3 tests/remodeled_trace/run_regression.py observe

# Compare every case against the approved goldens (exit 0 on full match)
python3 tests/remodeled_trace/run_regression.py check
```

## Relevant files with important changes respect Accel-sim

   The most important changes compared to Accel-Sim are:

1. **SM Model:**
   Located in `gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/`. It is focused on the SM implementation including sub-core pipelines, the SM memory unit, and new stats required to support parallelization.


2. **Instruction information:**
   Located in `util/traces_enhanced/`. It manages information about traced kernels and instructions, used during both tracing and simulation. It includes the Google Protocol Buffers implementation.
   


> [!WARNING]
> This repository shares our model and simulator parallelization with the community. It is not intended to be a long-term maintained repository.
