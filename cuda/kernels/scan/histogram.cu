#include "common.h"
#include "kernel_caller_declare.h"
#include "kernel_utils.cu.h"

void histOnCpu(const int8_t* h_h, uint32_t* h_hist, const uint32_t N) {
  for (uint32_t i = 0; i < N; ++i) {
    h_hist[h_h[i]]++;
  }
}

__global__ void histOnGPU_V1(const int8_t* d_h,
                             uint32_t* d_hist,
                             const uint32_t N) {
  const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    atomicAdd(&d_hist[d_h[idx]], 1);
  }
}

__global__ void histOnGPU_V2(const int8_t* d_h, int* d_hist, const int N) {
  __shared__ uint32_t s_hist[256];
  s_hist[threadIdx.x] = 0;
  __syncthreads();

  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < N) {
    atomicAdd(&s_hist[d_h[idx]], 1);
  }
  __syncthreads();
  atomicAdd(&d_hist[threadIdx.x], s_hist[threadIdx.x]);
}

__global__ void histOnGPU_V3(const int8_t* d_h, int* d_hist, const int N) {
  __shared__ uint32_t s_hist[256];
  s_hist[threadIdx.x] = 0;
  __syncthreads();

  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t stride = gridDim.x * blockDim.x;
  while (idx < N) {
    atomicAdd(&s_hist[d_h[idx]], 1);
    idx += stride;
  }
  __syncthreads();
  atomicAdd(&d_hist[threadIdx.x], s_hist[threadIdx.x]);
}
