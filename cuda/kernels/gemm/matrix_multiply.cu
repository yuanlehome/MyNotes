#include "dbg.h"

#include "common.h"
#include "kernel_caller_declare.h"
#include "kernel_utils.cu.h"

constexpr int kTileDim = 32;

// Naive solution as baseline
// block(kTileDim, kTileDim)
// grid(N / kTileDim, M / kTileDim)
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
// block(kTileDim, kTileDim)
// grid(N / kTileDim, M / kTileDim)
__global__ void matrixMultiplyKernel_V2(const float* A,
                                        const float* B,
                                        float* C,
                                        const int M,
                                        const int N,
                                        const int K) {
  __shared__ float s_a[kTileDim][kTileDim];
  __shared__ float s_b[kTileDim][kTileDim + 1];
  const int col = threadIdx.x + blockIdx.x * blockDim.x;
  const int row = threadIdx.y + blockIdx.y * blockDim.y;

  if (col < N && row < M) {
    float sum = 0.0;
    for (int k = 0; k < K; k += kTileDim) {
      s_a[threadIdx.y][threadIdx.x] = A[k + threadIdx.x + row * K];
      s_b[threadIdx.y][threadIdx.x] = B[col + (k + threadIdx.y) * N];
      __syncthreads();

      for (int i = 0; i < kTileDim; i++) {
        sum += s_a[threadIdx.y][i] * s_b[i][threadIdx.x];
      }
      __syncthreads();
    }
    C[col + row * N] = sum;
  }
}
