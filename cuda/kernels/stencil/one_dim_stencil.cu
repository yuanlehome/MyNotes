#include "dbg.h"

#include "common.h"
#include "kernel_caller_declare.h"
#include "kernel_utils.cu.h"

constexpr int kBlockSize = 128;
constexpr int kStencilSize = 9;
constexpr int kStencilPaddingSize = kStencilSize / 2;

__constant__ DATA_TYPE coef[kStencilPaddingSize];

__global__ void stencil_1d(const DATA_TYPE* in, DATA_TYPE* out) {
  __shared__ DATA_TYPE smem[kBlockSize + 2 * kStencilPaddingSize];
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int sidx = threadIdx.x + kStencilPaddingSize;
  smem[sidx] = in[idx];

  if (threadIdx.x < kStencilPaddingSize) {
    if (idx > kStencilPaddingSize) {
      smem[sidx - kStencilPaddingSize] = in[idx - kStencilPaddingSize];
    }
    if (idx < gridDim.x * blockDim.x - kBlockSize) {
      smem[sidx + kBlockSize] = in[idx + kBlockSize];
    }
  }
  __syncthreads();

  if (idx >= kStencilPaddingSize &&
      idx < gridDim.x * blockDim.x - kStencilPaddingSize) {
    DATA_TYPE value = 0.0;
    for (int i = 1; i <= kStencilPaddingSize; i++) {
      value += coef[i - 1] * (smem[sidx + i] - smem[sidx - i]);
    }
    out[idx] = value;
  }
}
