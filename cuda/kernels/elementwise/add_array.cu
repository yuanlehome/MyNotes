#include <algorithm>
#include <cstdint>

#include "common.h"
#include "kernel_caller_declare.h"
#include "kernel_utils.cu.h"

constexpr DATA_TYPE a = 1.23;
constexpr DATA_TYPE b = 2.34;
constexpr DATA_TYPE c = 3.57;

void addArrayOnCPU(const DATA_TYPE* x,
                   const DATA_TYPE* y,
                   DATA_TYPE* z,
                   const uint32_t N) {
  for (size_t i = 0; i < N; i++) {
    z[i] = x[i] + y[i];
  }
}

__device__ void add(const DATA_TYPE a, const DATA_TYPE b, DATA_TYPE* c) {
  *c = a + b;
}

__global__ void addArrayOnGPU_V1(const DATA_TYPE* x,
                                 const DATA_TYPE* y,
                                 DATA_TYPE* z,
                                 const uint32_t N) {
  const uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N) {
    add(x[idx], y[idx], &z[idx]);
  }
}

__global__ void addArrayOnGPU_V2(const DATA_TYPE* x,
                                 const DATA_TYPE* y,
                                 DATA_TYPE* z,
                                 const uint32_t N) {
  const int stride = blockDim.x * gridDim.x;
  for (int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < N; idx += stride)
    add(x[idx], y[idx], &z[idx]);
}

void addArray() {
  constexpr uint32_t N = 1e8 + 1;
  constexpr uint32_t SIZE = sizeof(DATA_TYPE) * N;

  MallocWrapper cpu_allocator;
  DATA_TYPE* h_x = (DATA_TYPE*)cpu_allocator.allocate(SIZE);
  DATA_TYPE* h_y = (DATA_TYPE*)cpu_allocator.allocate(SIZE);
  DATA_TYPE* h_z = (DATA_TYPE*)cpu_allocator.allocate(SIZE);

  std::fill_n(h_x, N, a);
  std::fill_n(h_y, N, b);

  Timer cpu_timer;
  float total_time = 0.0;
  for (size_t i = 0; i < repeats; i++) {
    cpu_timer.start();
    addArrayOnCPU(h_x, h_y, h_z, N);
    cpu_timer.stop();
    total_time += cpu_timer.elapsedTime();
  }
  DBG(total_time, cpu_timer.totalTime());
  std::printf("addArrayOnCPU cost time: %f ms\n", total_time / repeats);
  DBG(checkEqual(h_z, N, c));

  GPUMallocWrapper gpu_allocator;
  DATA_TYPE* d_x = (DATA_TYPE*)gpu_allocator.allocate(SIZE);
  DATA_TYPE* d_y = (DATA_TYPE*)gpu_allocator.allocate(SIZE);
  DATA_TYPE* d_z = (DATA_TYPE*)gpu_allocator.allocate(SIZE);

  CUDA_CHECK(cudaMemcpy(d_x, h_x, SIZE, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, h_y, SIZE, cudaMemcpyHostToDevice));

  const uint32_t block_size = 512;
  const uint32_t grid_size = (N + block_size - 1) / block_size;
  DBG(block_size, grid_size);
  dim3 block(block_size);
  dim3 grid(grid_size);

  GPUTimer gpu_timer;
  total_time = 0.0;
  for (size_t i = 0; i < repeats; i++) {
    gpu_timer.start();
    addArrayOnGPU_V1<<<grid, block>>>(d_x, d_y, d_z, N);
    gpu_timer.stop();
    total_time += gpu_timer.elapsedTime();
  }
  std::printf("addArrayOnGPU_V1 cost time: %f ms\n", total_time / repeats);
  CUDA_CHECK(cudaMemcpy(h_z, d_z, SIZE, cudaMemcpyDeviceToHost));
  DBG(checkEqual(h_z, N, c));

  total_time = 0.0;
  for (size_t i = 0; i < repeats; i++) {
    CUDA_CHECK(cudaMemcpy(d_z, h_x, SIZE, cudaMemcpyHostToDevice));  // 清空结果
    gpu_timer.start();
    addArrayOnGPU_V2<<<10240, block_size>>>(d_x, d_y, d_z, N);
    gpu_timer.stop();
    total_time += gpu_timer.elapsedTime();
  }
  std::printf("addArrayOnGPU_V2 cost time: %f ms\n", total_time / repeats);
  CUDA_CHECK(cudaMemcpy(h_z, d_z, SIZE, cudaMemcpyDeviceToHost));
  DBG(checkEqual(h_z, N, c));
}
