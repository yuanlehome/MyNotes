#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "dbg.h"

#include "common.h"
#include "kernel_caller_declare.h"
#include "kernel_utils.cuh"

constexpr DATA_TYPE a = 1.23;
constexpr DATA_TYPE b = 2.34;
constexpr DATA_TYPE c = 3.57;
constexpr DATA_TYPE EPSILON = 1.0e-8;

void checkData(const DATA_TYPE* z, const uint32_t N) {
  bool has_error = false;
  for (size_t i = 0; i < N; i++) {
    if (fabs(z[i] - c) > EPSILON) {
      has_error = true;
      break;
    }
  }
  std::printf("%s\n", has_error ? "has errors." : "no errors.");
}

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

__global__ void addArrayOnGPU(const DATA_TYPE* x,
                              const DATA_TYPE* y,
                              DATA_TYPE* z,
                              const uint32_t N) {
  const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < N) {
    add(x[tid], y[tid], &z[tid]);
  }
}

void addArray() {
  constexpr uint32_t N = 1e8 + 1;
  constexpr uint32_t M = sizeof(DATA_TYPE) * N;

  MallocWrapper cpu_allocator;
  DATA_TYPE* h_x = (DATA_TYPE*)cpu_allocator.allocate(M);
  DATA_TYPE* h_y = (DATA_TYPE*)cpu_allocator.allocate(M);
  DATA_TYPE* h_z = (DATA_TYPE*)cpu_allocator.allocate(M);

  std::fill_n(h_x, N, a);
  std::fill_n(h_y, N, b);

  for (size_t i = 0; i < warm_up; i++) {
    addArrayOnCPU(h_x, h_y, h_z, N);
  }
  Timer cpu_timer;
  cpu_timer.start();
  for (size_t i = 0; i < repeats; i++) {
    addArrayOnCPU(h_x, h_y, h_z, N);
  }
  cpu_timer.stop();
  std::printf("add_array_cpu cost time: %f ms\n",
              cpu_timer.elapsedTime() / repeats);
  checkData(h_z, N);

  GPUMallocWrapper gpu_allocator;
  DATA_TYPE* d_x = (DATA_TYPE*)gpu_allocator.allocate(M);
  DATA_TYPE* d_y = (DATA_TYPE*)gpu_allocator.allocate(M);
  DATA_TYPE* d_z = (DATA_TYPE*)gpu_allocator.allocate(M);

  CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));

  const uint32_t block_size = 128;
  const uint32_t grid_size = (N + block_size - 1) / block_size;
  dim3 block(block_size);
  dim3 grid(grid_size);

  for (size_t i = 0; i < warm_up; i++) {
    addArrayOnGPU<<<grid, block>>>(d_x, d_y, d_z, N);
  }
  GPUTimer gpu_timer;
  gpu_timer.start();
  for (size_t i = 0; i < repeats; i++) {
    addArrayOnGPU<<<grid, block>>>(d_x, d_y, d_z, N);
    // CHECK(cudaGetLastError());
    // CHECK(cudaDeviceSynchronize());
  }
  gpu_timer.stop();
  std::printf("add_array_gpu cost time: %f ms\n",
              gpu_timer.elapsedTime() / repeats);

  CHECK(cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost));
  checkData(h_z, N);
}
