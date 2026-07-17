#include <cuda_runtime.h>

#include <cstdint>
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

__global__ void half_pipeline_kernel(std::uint32_t *output) {
  const int thread_id = threadIdx.x;
  std::uint32_t value = 0x3C003C00U;
  const std::uint32_t increment = 0x38003800U;

#pragma unroll 1
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    asm volatile("add.rn.f16x2 %0, %1, %2;"
                 : "=r"(value)
                 : "r"(value), "r"(increment));
  }
  output[thread_id] = value;
}

}  // namespace

int main() {
  std::vector<std::uint32_t> output(kThreads);
  std::uint32_t *device_output = nullptr;

  if (!cuda_ok(
          cudaMalloc(&device_output, output.size() * sizeof(std::uint32_t)),
          "cudaMalloc output")) {
    cudaFree(device_output);
    return 2;
  }

  half_pipeline_kernel<<<1, kThreads>>>(device_output);
  const bool completed = cuda_ok(cudaGetLastError(), "kernel launch") &&
                         cuda_ok(cudaDeviceSynchronize(), "kernel completion") &&
                         cuda_ok(cudaMemcpy(output.data(), device_output,
                                            output.size() * sizeof(std::uint32_t),
                                            cudaMemcpyDeviceToHost),
                                 "cudaMemcpy output");
  cudaFree(device_output);
  if (!completed) {
    return 2;
  }

  std::printf("half_pipeline_result=0x%08x\n", output.front());
  return 0;
}
