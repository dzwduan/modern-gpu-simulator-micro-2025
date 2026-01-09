```bash
export CUDA_INSTALL_PATH=<path-to-your-cuda>
export PATH=$CUDA_INSTALL_PATH/bin:$PATH
sudo apt install protobuf-compiler

# build tracer tool
cd simulator-remodeled && ./util/tracer_nvbit/install_nvbit.sh
cd ./util/tracer_nvbit && make clean && make -j && cd ../../

# build gpu simulator
source ./gpu-simulator/setup_environment_no_git.sh && \
make -j -C ./gpu-simulator/

# trace a set of GPU app
source ./gpu-app-collection/src/setup_environment
make -j -C ./gpu-app-collection/src GPU_Microbenchmark CUOPTS='-gencode=arch=compute_80,code=\"sm_80,compute_80\"'
./util/tracer_nvbit/run_hw_trace.py -B GPU_Microbenchmark -D 0

# run simulation with openmp parallelism
OMP_NUM_THREADS=8 OMP_PROC_BIND=spread ./accel-sim.out \
    -config /path/to/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTX3080/gpgpusim.config \
    -config /path/to/gpu-simulator/configs/tested-cfgs/SM86_RTX3080/trace.config \
    -trace /path/to/traces/dynamic_trace.pb
```