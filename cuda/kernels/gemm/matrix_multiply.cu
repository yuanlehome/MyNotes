#include "dbg.h"

#include "common.h"
#include "kernel_caller_declare.h"
#include "kernel_utils.cuh"

constexpr int TILE_DIM = 32;

// Naive solution as baseline.
// dim3 block(TILE_DIM, TILE_DIM);
// dim3 grid(n / TILE_DIM, m / TILE_DIM);
__global__ void matrixMultiplyKernel_V1(const float* A,
                                        const float* B,
                                        float* C,
                                        const int M,
                                        const int N,
                                        const int K) {
  const int tile_x = threadIdx.x + blockIdx.x * blockDim.x;
  const int tile_y = threadIdx.y + blockIdx.y * blockDim.y;

  float sum = 0.0;
  if (tile_x < N && tile_y < M) {
    for (int k = 0; k < K; k++) {
      sum += A[k + tile_y * K] * B[tile_x + k * N];
    }
    C[tile_x + tile_y * N] = sum;
  }
}

// Use shared memory.
// dim3 block(TILE_DIM, TILE_DIM);
// dim3 grid(n / TILE_DIM, m / TILE_DIM);
__global__ void matrixMultiplyKernel_V2(const float* A,
                                        const float* B,
                                        float* C,
                                        const int M,
                                        const int N,
                                        const int K) {
  __shared__ float s_a[TILE_DIM][TILE_DIM];
  __shared__ float s_b[TILE_DIM][TILE_DIM];
  const int tile_x = threadIdx.x + blockIdx.x * blockDim.x;
  const int tile_y = threadIdx.y + blockIdx.y * blockDim.y;

  float sum = 0.0;
  if (tile_x < N && tile_y < M) {
    for (int k = 0; k < K; k += TILE_DIM) {
      s_a[threadIdx.y][threadIdx.x] = A[k + threadIdx.x + tile_y * K];
      s_b[threadIdx.y][threadIdx.x] = B[tile_x + (k + threadIdx.y) * N];
      __syncthreads();
      for (int i = 0; i < TILE_DIM; i++) {
        sum += s_a[threadIdx.y][i] * s_b[i][threadIdx.x];
      }
      __syncthreads();
    }
    C[tile_x + tile_y * N] = sum;
  }
}
