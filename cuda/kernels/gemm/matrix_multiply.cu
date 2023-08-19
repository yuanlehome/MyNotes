#include "dbg.h"

#include "common.h"
#include "kernel_caller_declare.h"
#include "kernel_utils.cu.h"

constexpr int TILE_DIM = 32;

// Naive solution as baseline
// block(TILE_DIM, TILE_DIM)
// grid(N / TILE_DIM, M / TILE_DIM)
__global__ void matrixMultiplyKernel_V1(const float* A,
                                        const float* B,
                                        float* C,
                                        const int M,
                                        const int N,
                                        const int K) {
  const int col = threadIdx.x + blockIdx.x * blockDim.x;
  const int row = threadIdx.y + blockIdx.y * blockDim.y;

  if (col < N && row < M) {
    float sum = 0.0;
    for (int k = 0; k < K; k++) {
      sum += A[k + row * K] * B[col + k * N];
    }
    C[col + row * N] = sum;
  }
}

// Use shared memory
// block(TILE_DIM, TILE_DIM)
// grid(N / TILE_DIM, M / TILE_DIM)
__global__ void matrixMultiplyKernel_V2(const float* A,
                                        const float* B,
                                        float* C,
                                        const int M,
                                        const int N,
                                        const int K) {
  __shared__ float s_a[TILE_DIM][TILE_DIM];
  __shared__ float s_b[TILE_DIM][TILE_DIM + 1];
  const int col = threadIdx.x + blockIdx.x * blockDim.x;
  const int row = threadIdx.y + blockIdx.y * blockDim.y;

  if (col < N && row < M) {
    float sum = 0.0;
    for (int k = 0; k < K; k += TILE_DIM) {
      s_a[threadIdx.y][threadIdx.x] = A[k + threadIdx.x + row * K];
      s_b[threadIdx.y][threadIdx.x] = B[col + (k + threadIdx.y) * N];
      __syncthreads();

      for (int i = 0; i < TILE_DIM; i++) {
        sum += s_a[threadIdx.y][i] * s_b[i][threadIdx.x];
      }
      __syncthreads();
    }
    C[col + row * N] = sum;
  }
}
