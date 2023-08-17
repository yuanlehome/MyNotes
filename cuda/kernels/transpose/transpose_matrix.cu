#include <cstdint>
#include <cstdio>

#include "dbg.h"

#include "common.h"
#include "kernel_caller_declare.h"

constexpr DATA_TYPE a = 1.23;
constexpr DATA_TYPE b = 2.34;

constexpr int BLOCK_DIM_X = 32;
constexpr int BLOCK_DIM_Y = 32;

// Print matrix with M x N
void printMatrix(const DATA_TYPE* A, const int M, const int N) {
  std::printf("\n");
  for (int ny = 0; ny < M; ny++) {
    for (int nx = 0; nx < N; nx++) {
      std::printf("%g\t", A[ny * N + nx]);
    }
    std::printf("\n");
  }
  std::printf("\n");
}

void transposeMatrixOnCPU(const DATA_TYPE* A,
                          DATA_TYPE* B,
                          const uint32_t M,
                          const uint32_t N) {
  for (int ny = 0; ny < M; ny++) {
    for (int nx = 0; nx < N; nx++) {
      B[nx * M + ny] = A[ny * N + nx];
    }
  }
}

// Copy matrix A to B of size (M x N) as row
__global__ void matrixCopyRow(const DATA_TYPE* A,
                              DATA_TYPE* B,
                              const uint32_t M,
                              const uint32_t N) {
  const int nx = threadIdx.x + blockIdx.x * BLOCK_DIM_X;
  const int ny = threadIdx.y + blockIdx.y * BLOCK_DIM_Y;
  const int idx_row = nx + ny * N;
  if (nx < N && ny < M) {
    B[idx_row] = A[idx_row];
  }
}

// Copy matrix A to B of size (M x N) as col
__global__ void matrixCopyCol(const DATA_TYPE* A,
                              DATA_TYPE* B,
                              const uint32_t M,
                              const uint32_t N) {
  const int nx = threadIdx.x + blockIdx.x * BLOCK_DIM_X;
  const int ny = threadIdx.y + blockIdx.y * BLOCK_DIM_Y;
  const int idx_col = ny + nx * M;
  if (nx < N && ny < M) {
    B[idx_col] = A[idx_col];
  }
}

// Transpose matrix A to B of size (M x N)
__global__ void transposeMatrix_V1(const DATA_TYPE* A,
                                   DATA_TYPE* B,
                                   const uint32_t M,
                                   const uint32_t N) {
  const int nx = threadIdx.x + blockIdx.x * BLOCK_DIM_X;
  const int ny = threadIdx.y + blockIdx.y * BLOCK_DIM_Y;
  if (nx < N && ny < M) {
    // 写非合并 读合并
    B[ny + nx * M] = A[nx + ny * N];
  }
}

// Transpose matrix A to B of size (M x N)
__global__ void transposeMatrix_V2(const DATA_TYPE* A,
                                   DATA_TYPE* B,
                                   const uint32_t M,
                                   const uint32_t N) {
  const int nx = threadIdx.x + blockIdx.x * BLOCK_DIM_X;
  const int ny = threadIdx.y + blockIdx.y * BLOCK_DIM_Y;
  if (nx < N && ny < M) {
    // 写合并 读非合并
    B[ny + nx * M] = A[nx + ny * N];
  }
}

// Transpose matrix A to B of size (M x N)
// 使用共享内存
__global__ void transposeMatrix_V3(const DATA_TYPE* A,
                                   DATA_TYPE* B,
                                   const uint32_t M,
                                   const uint32_t N) {
  __shared__ DATA_TYPE S[BLOCK_DIM_Y][BLOCK_DIM_X];
  int nx = threadIdx.x + blockIdx.x * BLOCK_DIM_X;
  int ny = threadIdx.y + blockIdx.y * BLOCK_DIM_Y;
  if (nx < N && ny < M) {
    // 写(共享内存)合并
    S[threadIdx.y][threadIdx.x] = A[nx + ny * N];
  }
  __syncthreads();

  nx = threadIdx.x + blockIdx.y * BLOCK_DIM_Y;
  ny = threadIdx.y + blockIdx.x * BLOCK_DIM_X;
  if (nx < M && ny < N) {
    // 写合并 读(共享内存)非合并 从而导致 bank conflict
    B[nx + ny * M] = S[threadIdx.x][threadIdx.y];
  }
}

// Transpose matrix A to B of size (M x N)
// 避免共享内存的 bank conflict
__global__ void transposeMatrix_V4(const DATA_TYPE* A,
                                   DATA_TYPE* B,
                                   const uint32_t M,
                                   const uint32_t N) {
  __shared__ DATA_TYPE S[BLOCK_DIM_Y][BLOCK_DIM_X + 1];
  int nx = threadIdx.x + blockIdx.x * BLOCK_DIM_X;
  int ny = threadIdx.y + blockIdx.y * BLOCK_DIM_Y;
  if (nx < N && ny < M) {
    S[threadIdx.y][threadIdx.x] = A[nx + ny * N];
  }
  __syncthreads();

  nx = threadIdx.x + blockIdx.y * BLOCK_DIM_Y;
  ny = threadIdx.y + blockIdx.x * BLOCK_DIM_X;
  if (nx < M && ny < N) {
    // 写合并 读(共享内存)非合并 但 bank conflict 已被避免
    B[nx + ny * M] = S[threadIdx.x][threadIdx.y];
  }
}

void transposeMatrix() {
  constexpr uint32_t M = 1e5;
  constexpr uint32_t N = 2e3;
  constexpr uint32_t SIZE = sizeof(DATA_TYPE) * M * N;

  MallocWrapper cpu_allocator;
  DATA_TYPE* h_x = (DATA_TYPE*)cpu_allocator.allocate(SIZE);
  DATA_TYPE* h_y = (DATA_TYPE*)cpu_allocator.allocate(SIZE);
  DATA_TYPE* h_y_base = (DATA_TYPE*)cpu_allocator.allocate(SIZE);

  std::fill_n(h_x, M * N / 2, a);
  std::fill_n(h_x + M * N / 2, M * N / 2, b);
  std::fill_n(h_y, M * N, -1);
  std::fill_n(h_y_base, M * N, -1);

  transposeMatrixOnCPU(h_x, h_y_base, M, N);

  GPUMallocWrapper gpu_allocator;
  DATA_TYPE* d_x = (DATA_TYPE*)gpu_allocator.allocate(SIZE);
  DATA_TYPE* d_y = (DATA_TYPE*)gpu_allocator.allocate(SIZE);

  CUDA_CHECK(cudaMemcpy(d_x, h_x, SIZE, cudaMemcpyHostToDevice));

  const uint32_t block_size_x = BLOCK_DIM_X;
  const uint32_t block_size_y = BLOCK_DIM_Y;
  const uint32_t grid_size_x = (N + BLOCK_DIM_X - 1) / BLOCK_DIM_X;
  const uint32_t grid_size_y = (M + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y;

  dbg(block_size_x, block_size_y, grid_size_x, grid_size_y);
  dim3 block(block_size_x, block_size_y);
  dim3 grid(grid_size_x, grid_size_y);

  GPUTimer gpu_timer;
  float total_time = 0.0;
  for (size_t i = 0; i < repeats; i++) {
    gpu_timer.start();
    matrixCopyRow<<<grid, block>>>(d_x, d_y, M, N);
    gpu_timer.stop();
    total_time += gpu_timer.elapsedTime();
  }
  dbg(total_time, gpu_timer.totalTime());
  std::printf("matrixCopyRow cost time: %f ms\n", total_time / repeats);
  CUDA_CHECK(cudaMemcpy(h_y, d_y, SIZE, cudaMemcpyDeviceToHost));
  dbg(checkEqual(h_x, h_y, M * N));

  total_time = 0.0;
  for (size_t i = 0; i < repeats; i++) {
    gpu_timer.start();
    matrixCopyCol<<<grid, block>>>(d_x, d_y, M, N);
    gpu_timer.stop();
    total_time += gpu_timer.elapsedTime();
  }
  dbg(total_time, gpu_timer.totalTime());
  std::printf("matrixCopyCol cost time: %f ms\n", total_time / repeats);
  CUDA_CHECK(cudaMemcpy(h_y, d_y, SIZE, cudaMemcpyDeviceToHost));
  dbg(checkEqual(h_x, h_y, M * N));

  total_time = 0.0;
  for (size_t i = 0; i < repeats; i++) {
    gpu_timer.start();
    transposeMatrix_V1<<<grid, block>>>(d_x, d_y, M, N);
    gpu_timer.stop();
    total_time += gpu_timer.elapsedTime();
  }
  dbg(total_time, gpu_timer.totalTime());
  std::printf("transposeMatrix_V1 cost time: %f ms\n", total_time / repeats);
  CUDA_CHECK(cudaMemcpy(h_y, d_y, SIZE, cudaMemcpyDeviceToHost));
  dbg(checkEqual(h_y_base, h_y, M * N));

  total_time = 0.0;
  for (size_t i = 0; i < repeats; i++) {
    gpu_timer.start();
    transposeMatrix_V2<<<grid, block>>>(d_x, d_y, M, N);
    gpu_timer.stop();
    total_time += gpu_timer.elapsedTime();
  }
  dbg(total_time, gpu_timer.totalTime());
  std::printf("transposeMatrix_V2 cost time: %f ms\n", total_time / repeats);
  CUDA_CHECK(cudaMemcpy(h_y, d_y, SIZE, cudaMemcpyDeviceToHost));
  dbg(checkEqual(h_y_base, h_y, M * N));

  total_time = 0.0;
  for (size_t i = 0; i < repeats; i++) {
    gpu_timer.start();
    transposeMatrix_V3<<<grid, block>>>(d_x, d_y, M, N);
    gpu_timer.stop();
    total_time += gpu_timer.elapsedTime();
  }
  dbg(total_time, gpu_timer.totalTime());
  std::printf("transposeMatrix_V3 cost time: %f ms\n", total_time / repeats);
  CUDA_CHECK(cudaMemcpy(h_y, d_y, SIZE, cudaMemcpyDeviceToHost));
  dbg(checkEqual(h_y_base, h_y, M * N));

  total_time = 0.0;
  for (size_t i = 0; i < repeats; i++) {
    gpu_timer.start();
    transposeMatrix_V4<<<grid, block>>>(d_x, d_y, M, N);
    gpu_timer.stop();
    total_time += gpu_timer.elapsedTime();
  }
  dbg(total_time, gpu_timer.totalTime());
  std::printf("transposeMatrix_V4 cost time: %f ms\n", total_time / repeats);

  CUDA_CHECK(cudaMemcpy(h_y, d_y, SIZE, cudaMemcpyDeviceToHost));
  dbg(checkEqual(h_y_base, h_y, M * N));
}
