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

// Transpose matrix A to B of size (N x N)
__global__ void transposeSquareMatrix_V1(const DATA_TYPE* A,
                                         DATA_TYPE* B,
                                         const uint32_t N) {
  const int nx = threadIdx.x + blockIdx.x * TILE_DIM;
  const int ny = threadIdx.y + blockIdx.y * TILE_DIM;
  if (nx < N && ny < N) {
    // 写非合并 读合并
    B[ny + nx * N] = A[nx + ny * N];
  }
}

// Transpose matrix A to B of size (N x N)
__global__ void transposeSquareMatrix_V2(const DATA_TYPE* A,
                                         DATA_TYPE* B,
                                         const uint32_t N) {
  const int nx = threadIdx.x + blockIdx.x * TILE_DIM;
  const int ny = threadIdx.y + blockIdx.y * TILE_DIM;
  if (nx < N && ny < N) {
    // 写合并 读非合并
    B[nx + ny * N] = A[ny + nx * N];
  }
}

// Transpose matrix A to B of size (N x N)
// 使用共享内存
__global__ void transposeSquareMatrix_V3(const DATA_TYPE* A,
                                         DATA_TYPE* B,
                                         const uint32_t N) {
  __shared__ DATA_TYPE S[TILE_DIM][TILE_DIM];
  int nx = threadIdx.x + blockIdx.x * TILE_DIM;
  int ny = threadIdx.y + blockIdx.y * TILE_DIM;
  if (nx < N && ny < N) {
    S[threadIdx.y][threadIdx.x] = A[nx + ny * N];
  }
  __syncthreads();

  nx = threadIdx.x + blockIdx.y * TILE_DIM;
  ny = threadIdx.y + blockIdx.x * TILE_DIM;
  if (nx < N && ny < N) {
    // 写合并 读(共享内存)非合并
    B[nx + ny * N] = S[threadIdx.x][threadIdx.y];
  }
}

// Transpose matrix A to B of size (N x N)
// 避免共享内存的 bank 冲突
__global__ void transposeSquareMatrix_V4(const DATA_TYPE* A,
                                         DATA_TYPE* B,
                                         const uint32_t N) {
  __shared__ DATA_TYPE S[TILE_DIM][TILE_DIM + 1];
  int nx = threadIdx.x + blockIdx.x * TILE_DIM;
  int ny = threadIdx.y + blockIdx.y * TILE_DIM;
  if (nx < N && ny < N) {
    S[threadIdx.y][threadIdx.x] = A[nx + ny * N];
  }
  __syncthreads();

  nx = threadIdx.x + blockIdx.y * TILE_DIM;
  ny = threadIdx.y + blockIdx.x * TILE_DIM;
  if (nx < N && ny < N) {
    // 写合并 读(共享内存)非合并
    B[nx + ny * N] = S[threadIdx.x][threadIdx.y];
  }
}

void transposeSquareMatrix() {
  constexpr uint32_t N = 1e4;
  constexpr uint32_t M = sizeof(DATA_TYPE) * N * N;

  MallocWrapper cpu_allocator;
  DATA_TYPE* h_x = (DATA_TYPE*)cpu_allocator.allocate(M);
  DATA_TYPE* h_y = (DATA_TYPE*)cpu_allocator.allocate(M);

  std::fill_n(h_x, N * N, a);
  std::fill_n(h_y, N * N, b);
  dbg(checkEqual(h_x, N * N, a),
      checkEqual(h_y, N * N, b),
      checkEqual(h_x, h_y, N * N));

  GPUMallocWrapper gpu_allocator;
  DATA_TYPE* d_x = (DATA_TYPE*)gpu_allocator.allocate(M);
  DATA_TYPE* d_y = (DATA_TYPE*)gpu_allocator.allocate(M);

  CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));

  const uint32_t block_size_x = TILE_DIM;
  const uint32_t block_size_y = block_size_x;
  const uint32_t grid_size_x = (N + TILE_DIM - 1) / TILE_DIM;
  const uint32_t grid_size_y = grid_size_x;
  dbg(block_size_x, block_size_y, grid_size_x, grid_size_y);
  dim3 block(block_size_x, block_size_y);
  dim3 grid(grid_size_x, grid_size_y);

  GPUTimer gpu_timer;
  float total_time = 0.0;
  for (size_t i = 0; i < repeats; i++) {
    gpu_timer.start();
    matrixCopy<<<grid, block>>>(d_x, d_y, N);
    gpu_timer.stop();
    total_time += gpu_timer.elapsedTime();
  }
  dbg(total_time, gpu_timer.totalTime());
  std::printf("matrixCopy cost time: %f ms\n", total_time / repeats);

  total_time = 0.0;
  for (size_t i = 0; i < repeats; i++) {
    gpu_timer.start();
    transposeSquareMatrix_V1<<<grid, block>>>(d_x, d_y, N);
    gpu_timer.stop();
    total_time += gpu_timer.elapsedTime();
  }
  dbg(total_time, gpu_timer.totalTime());
  std::printf("transposeSquareMatrix_V1 cost time: %f ms\n",
              total_time / repeats);

  total_time = 0.0;
  for (size_t i = 0; i < repeats; i++) {
    gpu_timer.start();
    transposeSquareMatrix_V2<<<grid, block>>>(d_x, d_y, N);
    gpu_timer.stop();
    total_time += gpu_timer.elapsedTime();
  }
  dbg(total_time, gpu_timer.totalTime());
  std::printf("transposeSquareMatrix_V2 cost time: %f ms\n",
              total_time / repeats);

  total_time = 0.0;
  for (size_t i = 0; i < repeats; i++) {
    gpu_timer.start();
    transposeSquareMatrix_V3<<<grid, block>>>(d_x, d_y, N);
    gpu_timer.stop();
    total_time += gpu_timer.elapsedTime();
  }
  dbg(total_time, gpu_timer.totalTime());
  std::printf("transposeSquareMatrix_V3 cost time: %f ms\n",
              total_time / repeats);

  total_time = 0.0;
  for (size_t i = 0; i < repeats; i++) {
    gpu_timer.start();
    transposeSquareMatrix_V4<<<grid, block>>>(d_x, d_y, N);
    gpu_timer.stop();
    total_time += gpu_timer.elapsedTime();
  }
  dbg(total_time, gpu_timer.totalTime());
  std::printf("transposeSquareMatrix_V4 cost time: %f ms\n",
              total_time / repeats);
}
