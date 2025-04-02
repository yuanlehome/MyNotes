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

  CpuMallocWrapper cpu_allocator;
  DATA_TYPE* h_x = (DATA_TYPE*)cpu_allocator.allocate(SIZE);
  DATA_TYPE* h_y = (DATA_TYPE*)cpu_allocator.allocate(SIZE);
  DATA_TYPE* h_z = (DATA_TYPE*)cpu_allocator.allocate(SIZE);

  std::fill_n(h_x, N, a);
  std::fill_n(h_y, N, b);

  utils::performance<CpuTimer>(
      "addArrayOnCPU",
      repeats,
      [&] {},
      [&] { addArrayOnCPU(h_x, h_y, h_z, N); },
      [&] { dbg(utils::checkEqual(h_z, N, c)); });

  GpuMallocWrapper gpu_allocator;
  DATA_TYPE* d_x = (DATA_TYPE*)gpu_allocator.allocate(SIZE);
  DATA_TYPE* d_y = (DATA_TYPE*)gpu_allocator.allocate(SIZE);
  DATA_TYPE* d_z = (DATA_TYPE*)gpu_allocator.allocate(SIZE);

  CUDA_CHECK(cudaMemcpy(d_x, h_x, SIZE, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, h_y, SIZE, cudaMemcpyHostToDevice));

  const uint32_t block_size = 512;
  const uint32_t grid_size = (N + block_size - 1) / block_size;
  dbg(block_size, grid_size);
  dim3 block(block_size);
  dim3 grid(grid_size);

  utils::performance<GpuTimer>(
      "addArrayOnGPU_V1",
      repeats,
      [&] {},
      [&] { addArrayOnGPU_V1<<<grid, block>>>(d_x, d_y, d_z, N); },
      [&] {
        CUDA_CHECK(cudaMemcpy(h_z, d_z, SIZE, cudaMemcpyDeviceToHost));
        dbg(utils::checkEqual(h_z, N, c));
      });

  utils::performance<GpuTimer>(
      "addArrayOnGPU_V2",
      repeats,
      [&] {
        // 清空结果
        CUDA_CHECK(cudaMemcpy(d_z, h_x, SIZE, cudaMemcpyHostToDevice));
      },
      [&] { addArrayOnGPU_V2<<<10240, block_size>>>(d_x, d_y, d_z, N); },
      [&] {
        CUDA_CHECK(cudaMemcpy(h_z, d_z, SIZE, cudaMemcpyDeviceToHost));
        dbg(utils::checkEqual(h_z, N, c));
      });
}
