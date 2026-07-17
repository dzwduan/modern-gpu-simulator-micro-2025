#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

namespace {

constexpr int kThreads = 32;
constexpr int kIterations = 64;

bool cuda_ok(cudaError_t status, const char *operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
  return false;
}

__global__ void fp64_dispatch_kernel(const double *input, double *output) {
  const int thread_id = threadIdx.x;
  double value = input[thread_id];

#pragma unroll 1
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    value = __dadd_rn(value, 0.125);
  }
  output[thread_id] = value;
}

}  // namespace

int main() {
  std::vector<double> input(kThreads, 1.0);
  std::vector<double> output(kThreads);
  double *device_input = nullptr;
  double *device_output = nullptr;

  if (!cuda_ok(cudaMalloc(&device_input, input.size() * sizeof(double)),
               "cudaMalloc input") ||
      !cuda_ok(cudaMalloc(&device_output, output.size() * sizeof(double)),
               "cudaMalloc output") ||
      !cuda_ok(cudaMemcpy(device_input, input.data(),
                          input.size() * sizeof(double), cudaMemcpyHostToDevice),
               "cudaMemcpy input")) {
    cudaFree(device_input);
    cudaFree(device_output);
    return 2;
  }

  fp64_dispatch_kernel<<<1, kThreads>>>(device_input, device_output);
  const bool completed = cuda_ok(cudaGetLastError(), "kernel launch") &&
                         cuda_ok(cudaDeviceSynchronize(), "kernel completion") &&
                         cuda_ok(cudaMemcpy(output.data(), device_output,
                                            output.size() * sizeof(double),
                                            cudaMemcpyDeviceToHost),
                                 "cudaMemcpy output");
  cudaFree(device_input);
  cudaFree(device_output);
  if (!completed) {
    return 2;
  }

  std::printf("fp64_dispatch_result=%.3f\n", output.front());
  return 0;
}
