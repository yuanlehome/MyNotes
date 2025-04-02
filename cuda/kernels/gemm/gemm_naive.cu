#include "all.h"
#include "common.h"
#include "kernel_utils.cu.h"

// Naive solution as baseline
__global__ void matrixMultiplyKernel_V1(const DATA_TYPE* __restrict__ A,
                                        const DATA_TYPE* __restrict__ B,
                                        DATA_TYPE* __restrict__ C,
                                        int M,
                                        int N,
                                        int K,
                                        DATA_TYPE alpha,
                                        DATA_TYPE beta) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  if (col < N && row < M) {
    DATA_TYPE sum = 0.0;
    for (int k = 0; k < K; k++) {
      sum += A[k + row * K] * B[col + k * N];
    }
    C[col + row * N] = alpha * sum + beta * C[col + row * N];
  }
}

// Use shared memory and avoid bank conflicts
template <int kTileDim>
__global__ void matrixMultiplyKernel_V2(const DATA_TYPE* __restrict__ A,
                                        const DATA_TYPE* __restrict__ B,
                                        DATA_TYPE* __restrict__ C,
                                        int M,
                                        int N,
                                        int K,
                                        DATA_TYPE alpha,
                                        DATA_TYPE beta) {
  __shared__ DATA_TYPE s_a[kTileDim][kTileDim];
  __shared__ DATA_TYPE s_b[kTileDim][kTileDim + 1];
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  if (col < N && row < M) {
    DATA_TYPE sum = 0.0;
    for (int k = 0; k < K; k += kTileDim) {
      s_a[threadIdx.y][threadIdx.x] = A[k + threadIdx.x + row * K];
      s_b[threadIdx.y][threadIdx.x] = B[col + (k + threadIdx.y) * N];
      __syncthreads();
#pragma unroll
      for (int i = 0; i < kTileDim; i++) {
        sum += s_a[threadIdx.y][i] * s_b[i][threadIdx.x];
      }
      __syncthreads();
    }
    C[col + row * N] = alpha * sum + beta * C[col + row * N];
  }
}

// Use vector 4 access
template <int kTileDim>
__global__ void matrixMultiplyKernel_V3(const DATA_TYPE* __restrict__ A,
                                        const DATA_TYPE* __restrict__ B,
                                        DATA_TYPE* __restrict__ C,
                                        int M,
                                        int N,
                                        int K,
                                        DATA_TYPE alpha,
                                        DATA_TYPE beta) {
  __shared__ DATA_TYPE s_a[kTileDim][kTileDim + 1];
  __shared__ DATA_TYPE s_b[kTileDim][kTileDim + 1];
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  if (col < N && row < M) {
    DATA_TYPE sum = 0.0;
    for (int k = 0; k < K; k += kTileDim) {
      s_a[threadIdx.y][threadIdx.x] = A[k + threadIdx.x + row * K];
      s_b[threadIdx.y][threadIdx.x] = B[col + (k + threadIdx.y) * N];
      __syncthreads();
#pragma unroll
      for (int i = 0; i < kTileDim; i++) {
        sum += s_a[threadIdx.y][i] * s_b[i][threadIdx.x];
      }
      __syncthreads();
    }
    C[col + row * N] = alpha * sum + beta * C[col + row * N];
  }
}

void gemm_naive() {
  constexpr uint32_t M = 1024;
  constexpr uint32_t N = 1024;
  constexpr uint32_t K = 512;

  constexpr uint32_t A_SIZE = sizeof(DATA_TYPE) * M * K;
  constexpr uint32_t B_SIZE = sizeof(DATA_TYPE) * K * N;
  constexpr uint32_t C_SIZE = sizeof(DATA_TYPE) * M * N;

  CpuMallocWrapper cpu_allocator;
  DATA_TYPE* h_a = (DATA_TYPE*)cpu_allocator.allocate(A_SIZE);
  DATA_TYPE* h_b = (DATA_TYPE*)cpu_allocator.allocate(B_SIZE);
  DATA_TYPE* h_c = (DATA_TYPE*)cpu_allocator.allocate(C_SIZE);
  std::fill_n(h_a, M * K, 0.5);
  std::fill_n(h_b, K * N, 0.3);
  std::fill_n(h_c, M * N, 0.0);

  GpuMallocWrapper gpu_allocator;
  DATA_TYPE* d_a = (DATA_TYPE*)gpu_allocator.allocate(A_SIZE);
  DATA_TYPE* d_b = (DATA_TYPE*)gpu_allocator.allocate(B_SIZE);
  DATA_TYPE* d_c = (DATA_TYPE*)gpu_allocator.allocate(C_SIZE);
  CUDA_CHECK(cudaMemcpy(d_a, h_a, A_SIZE, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, B_SIZE, cudaMemcpyHostToDevice));
  utils::fill_n(d_c, M * N, 0.0);

  // CPU results
  DATA_TYPE* real_c = (DATA_TYPE*)cpu_allocator.allocate(C_SIZE);
  std::fill_n(real_c, M * N, 0.0);
  utils::matrixMultiply(h_a, h_b, real_c, M, N, K);

  // GPU results
  utils::performance<GpuTimer>(
      "matrixMultiplyKernel_V1_16x16x16",
      repeats,
      [&] { utils::fill_n(d_c, M * N, 0.0); },
      [&] {
        constexpr int kTileDim = 16;
        dim3 block(kTileDim, kTileDim);
        dim3 grid((M + kTileDim - 1) / kTileDim, (N + kTileDim - 1) / kTileDim);
        matrixMultiplyKernel_V1<<<grid, block>>>(
            d_a, d_b, d_c, M, N, K, 1.0, 0.0);
      },
      [&] {
        CUDA_CHECK(cudaMemcpy(h_c, d_c, C_SIZE, cudaMemcpyDeviceToHost));
        dbg(utils::checkEqual(h_c, real_c, M * N));
      });

  utils::performance<GpuTimer>(
      "matrixMultiplyKernel_V1_32x32x32",
      repeats,
      [&] { utils::fill_n(d_c, M * N, 0.0); },
      [&] {
        constexpr int kTileDim = 32;
        dim3 block(kTileDim, kTileDim);
        dim3 grid((M + kTileDim - 1) / kTileDim, (N + kTileDim - 1) / kTileDim);
        matrixMultiplyKernel_V1<<<grid, block>>>(
            d_a, d_b, d_c, M, N, K, 1.0, 0.0);
      },
      [&] {
        CUDA_CHECK(cudaMemcpy(h_c, d_c, C_SIZE, cudaMemcpyDeviceToHost));
        dbg(utils::checkEqual(h_c, real_c, M * N));
      });

  utils::performance<GpuTimer>(
      "matrixMultiplyKernel_V2_16x16x16",
      repeats,
      [&] { utils::fill_n(d_c, M * N, 0.0); },
      [&] {
        constexpr int kTileDim = 16;
        dim3 block(kTileDim, kTileDim);
        dim3 grid((M + kTileDim - 1) / kTileDim, (N + kTileDim - 1) / kTileDim);
        matrixMultiplyKernel_V2<kTileDim>
            <<<grid, block>>>(d_a, d_b, d_c, M, N, K, 1.0, 0.0);
      },
      [&] {
        CUDA_CHECK(cudaMemcpy(h_c, d_c, C_SIZE, cudaMemcpyDeviceToHost));
        dbg(utils::checkEqual(h_c, real_c, M * N));
      });

  utils::performance<GpuTimer>(
      "matrixMultiplyKernel_V2_32x32x32",
      repeats,
      [&] { utils::fill_n(d_c, M * N, 0.0); },
      [&] {
        constexpr int kTileDim = 32;
        dim3 block(kTileDim, kTileDim);
        dim3 grid((M + kTileDim - 1) / kTileDim, (N + kTileDim - 1) / kTileDim);
        matrixMultiplyKernel_V2<kTileDim>
            <<<grid, block>>>(d_a, d_b, d_c, M, N, K, 1.0, 0.0);
      },
      [&] {
        CUDA_CHECK(cudaMemcpy(h_c, d_c, C_SIZE, cudaMemcpyDeviceToHost));
        dbg(utils::checkEqual(h_c, real_c, M * N));
      });

  // utils::performance<GpuTimer>(
  //     "matrixMultiplyKernel_V3_16x16x64",
  //     repeats,
  //     [&] { utils::fill_n(d_c, M * N, 0.0); },
  //     [&] {
  //       constexpr int kTileDim = 16;
  //       dim3 block(kTileDim, kTileDim);
  //       dim3 grid((M + kTileDim - 1) / kTileDim / 4,
  //                 (N + kTileDim - 1) / kTileDim);
  //       matrixMultiplyKernel_V3<kTileDim>
  //           <<<grid, block>>>(d_a, d_b, d_c, M, N, K, 1.0, 0.0);
  //     },
  //     [&] {
  //       CUDA_CHECK(cudaMemcpy(h_c, d_c, C_SIZE, cudaMemcpyDeviceToHost));
  //       dbg(utils::checkEqual(h_c, real_c, M * N));
  //     });
}
