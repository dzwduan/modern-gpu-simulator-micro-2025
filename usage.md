```bash
export CUDA_INSTALL_PATH=<path-to-your-cuda>
export PATH=$CUDA_INSTALL_PATH/bin:$PATH
sudo apt install protobuf-compiler

# build tracer tool
cd simulator-remodeled && ./util/tracer_nvbit/install_nvbit.sh
cd ./util/tracer_nvbit && make clean && make -j && cd ../../

# build gpu simulator
source ./gpu-simulator/setup_environment_no_git.sh && make -C ./gpu-simulator clean && make -j -C ./gpu-simulator/ 

# trace a set of GPU app
source ./gpu-app-collection/src/setup_environment
make -j -C ./gpu-app-collection/src GPU_Microbenchmark CUOPTS='-gencode=arch=compute_89,code=\"sm_89,compute_89\"'
./util/tracer_nvbit/run_hw_trace.py -B GPU_Microbenchmark -D 0

# run simulation with openmp parallelism
OMP_NUM_THREADS=32 OMP_PROC_BIND=spread ./gpu-simulator/bin/release/accel-sim.out \
    -config /home/duanzhenwei/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM89_RTX4090/gpgpusim.config \
    -config /home/duanzhenwei/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/configs/tested-cfgs/SM89_RTX4090/trace.config \
    -trace /home/duanzhenwei/modern-gpu-simulator-micro-2025/simulator-remodeled/hw_run/traces/device-0/12.6/l1_shared_bw/NO_ARGS/traces/dynamic_trace.pb
```