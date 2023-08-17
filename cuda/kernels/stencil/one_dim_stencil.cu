#include "dbg.h"

#include "common.h"
#include "kernel_caller_declare.h"
#include "kernel_utils.cu.h"

constexpr int STENCIL_SIZE = 9;
constexpr int STENCIL_PADDING_SIZE = STENCIL_SIZE / 2;

__constant__ DATA_TYPE coef[STENCIL_PADDING_SIZE];

__global__ void stencil_1d(const DATA_TYPE* in, DATA_TYPE* out) {
  __shared__ DATA_TYPE smem[BLOCK_SIZE + 2 * STENCIL_PADDING_SIZE];
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int sidx = threadIdx.x + STENCIL_PADDING_SIZE;
  smem[sidx] = in[idx];

  if (threadIdx.x < STENCIL_PADDING_SIZE) {
    if (idx > STENCIL_PADDING_SIZE) {
      smem[sidx - STENCIL_PADDING_SIZE] = in[idx - STENCIL_PADDING_SIZE];
    }
    if (idx < gridDim.x * blockDim.x - BLOCK_SIZE) {
      smem[sidx + BLOCK_SIZE] = in[idx + BLOCK_SIZE];
    }
  }
  __syncthreads();

  if (idx < STENCIL_PADDING_SIZE ||
      idx >= gridDim.x * blockDim.x - STENCIL_PADDING_SIZE)
    return;

  DATA_TYPE value = 0.0;
  for (int i = 1; i <= STENCIL_PADDING_SIZE; i++) {
    value += coef[i - 1] * (smem[sidx + i] - smem[sidx - i]);
  }
  out[idx] = value;
}
