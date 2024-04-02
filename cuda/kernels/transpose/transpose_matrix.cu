#include <cstdint>
#include <cstdio>

#include "common.h"
#include "kernel_caller_declare.h"

constexpr DATA_TYPE a = 1.23;
constexpr DATA_TYPE b = 2.34;

constexpr int kBlockDimX = 32;
constexpr int kBlockDimY = 32;

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

// Copy matrix A(M x N) to B(N x M) as row
__global__ void matrixCopyRow(const DATA_TYPE* A,
                              DATA_TYPE* B,
                              const uint32_t M,
                              const uint32_t N) {
  const int nx = threadIdx.x + blockIdx.x * kBlockDimX;
  const int ny = threadIdx.y + blockIdx.y * kBlockDimY;
  const int idx_row = nx + ny * N;
  if (nx < N && ny < M) {
    B[idx_row] = A[idx_row];
  }
}

// Copy matrix A(M x N) to B(N x M) as col
__global__ void matrixCopyCol(const DATA_TYPE* A,
                              DATA_TYPE* B,
                              const uint32_t M,
                              const uint32_t N) {
  const int nx = threadIdx.x + blockIdx.x * kBlockDimX;
  const int ny = threadIdx.y + blockIdx.y * kBlockDimY;
  const int idx_col = ny + nx * M;
  if (nx < N && ny < M) {
    B[idx_col] = A[idx_col];
  }
}

// Transpose matrix A(M x N) to B(N x M)
__global__ void transposeMatrix_V1(const DATA_TYPE* A,
                                   DATA_TYPE* B,
                                   const uint32_t M,
                                   const uint32_t N) {
  const int nx = threadIdx.x + blockIdx.x * kBlockDimX;
  const int ny = threadIdx.y + blockIdx.y * kBlockDimY;
  if (nx < N && ny < M) {
    // 写非合并 读合并
    B[ny + nx * M] = A[nx + ny * N];
  }
}

// Transpose matrix A(M x N) to B(N x M)
__global__ void transposeMatrix_V2(const DATA_TYPE* A,
                                   DATA_TYPE* B,
                                   const uint32_t M,
                                   const uint32_t N) {
  const int nx = threadIdx.x + blockIdx.y * kBlockDimY;
  const int ny = threadIdx.y + blockIdx.x * kBlockDimX;
  if (nx < M && ny < N) {
    // 写合并 读非合并
    B[nx + ny * M] = A[ny + nx * N];
  }
}

// Transpose matrix A(M x N) to B(N x M)
// 使用共享内存
__global__ void transposeMatrix_V3(const DATA_TYPE* A,
                                   DATA_TYPE* B,
                                   const uint32_t M,
                                   const uint32_t N) {
  __shared__ DATA_TYPE S[kBlockDimY][kBlockDimX];
  int nx = threadIdx.x + blockIdx.x * kBlockDimX;
  int ny = threadIdx.y + blockIdx.y * kBlockDimY;
  if (nx < N && ny < M) {
    // 读合并 写(共享内存)无 bank conflict
    S[threadIdx.y][threadIdx.x] = A[nx + ny * N];
  }
  __syncthreads();

  nx = threadIdx.x + blockIdx.y * kBlockDimY;
  ny = threadIdx.y + blockIdx.x * kBlockDimX;
  if (nx < M && ny < N) {
    // 写合并 读(共享内存)存在 bank conflict
    B[nx + ny * M] = S[threadIdx.x][threadIdx.y];
  }
}

// Transpose matrix A(M x N) to B(N x M)
// 避免共享内存的 bank conflict
__global__ void transposeMatrix_V4(const DATA_TYPE* A,
                                   DATA_TYPE* B,
                                   const uint32_t M,
                                   const uint32_t N) {
  __shared__ DATA_TYPE S[kBlockDimY][kBlockDimX + 1];
  int nx = threadIdx.x + blockIdx.x * kBlockDimX;
  int ny = threadIdx.y + blockIdx.y * kBlockDimY;
  if (nx < N && ny < M) {
    S[threadIdx.y][threadIdx.x] = A[nx + ny * N];
  }
  __syncthreads();

  nx = threadIdx.x + blockIdx.y * kBlockDimY;
  ny = threadIdx.y + blockIdx.x * kBlockDimX;
  if (nx < M && ny < N) {
    // 写合并 读(共享内存)的 bank conflict 已被避免
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

  const uint32_t block_size_x = kBlockDimX;
  const uint32_t block_size_y = kBlockDimY;
  const uint32_t grid_size_x = (N + block_size_x - 1) / block_size_x;
  const uint32_t grid_size_y = (M + block_size_y - 1) / block_size_y;

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
