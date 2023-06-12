#include <cstdint>
#include <cstdio>

#include "dbg.h"

#include "common.h"
#include "kernel_caller_declare.h"

constexpr DATA_TYPE a = 1.23;
constexpr DATA_TYPE b = 2.34;

constexpr int TILE_DIM = 32;

// Copy matrix A to B of size (N x N)
__global__ void matrixCopy(const DATA_TYPE* A, DATA_TYPE* B, const uint32_t N) {
  const int nx = threadIdx.x + blockIdx.x * TILE_DIM;
  const int ny = threadIdx.y + blockIdx.y * TILE_DIM;
  const int idx = nx + ny * N;
  if (nx < N && ny < N) {
    B[idx] = A[idx];
  }
}

__global__ void transposeSquareMatrix() {}

__global__ void transposeNonSquareMatrix() {}

void transposeMatrix() {
  constexpr uint32_t N = 1e4;
  constexpr uint32_t M = sizeof(DATA_TYPE) * N * N;

  MallocWrapper cpu_allocator;
  DATA_TYPE* h_x = (DATA_TYPE*)cpu_allocator.allocate(M);
  DATA_TYPE* h_y = (DATA_TYPE*)cpu_allocator.allocate(M);
  dbg(h_x, h_y);

  std::fill_n(h_x, N * N, a);
  std::fill_n(h_y, N * N, b);
  dbg(checkEqual(h_x, N * N, a),
      checkEqual(h_y, N * N, b),
      checkEqual(h_x, h_y, N * N));

  GPUMallocWrapper gpu_allocator;
  DATA_TYPE* d_x = (DATA_TYPE*)gpu_allocator.allocate(M);
  DATA_TYPE* d_y = (DATA_TYPE*)gpu_allocator.allocate(M);
  dbg(d_x, d_y);

  CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));

  const uint32_t block_size_x = TILE_DIM;
  const uint32_t block_size_y = block_size_x;
  const uint32_t grid_size_x = (N + TILE_DIM - 1) / TILE_DIM;
  const uint32_t grid_size_y = grid_size_x;
  dbg(block_size_x, block_size_y, grid_size_x, grid_size_y);
  dim3 block(block_size_x, block_size_y);
  dim3 grid(grid_size_x, grid_size_y);

  for (size_t i = 0; i < warm_up; i++) {
    matrixCopy<<<grid, block>>>(d_x, d_y, N);
  }
  GPUTimer gpu_timer;
  float total_time = 0.0;
  for (size_t i = 0; i < repeats; i++) {
    gpu_timer.start();
    matrixCopy<<<grid, block>>>(d_x, d_y, N);
    gpu_timer.stop();
    total_time += gpu_timer.elapsedTime();
  }
  std::printf("matrixCopy cost time: %f ms\n", total_time / repeats);

  CHECK(cudaMemcpy(h_y, d_y, M, cudaMemcpyDeviceToHost));
  dbg(checkEqual(h_x, h_y, N * N));
}
