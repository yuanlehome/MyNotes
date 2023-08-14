#pragma once

#define FULL_MASK 0xffffffff

template <typename T>
__forceinline__ __device__ T warpReduceSum(T value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(FULL_MASK, value, offset);
  }
  return value;
}

template <typename T>
__forceinline__ __device__ T blockReduceSum(T value) {
  __shared__ T shared[32];
  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;

  value = warpReduceSum<T>(value);
  __syncthreads();

  if (lane_id == 0) {
    shared[warp_id] = value;
  }
  __syncthreads();

  value = (threadIdx.x < blockDim.x / warpSize) ? shared[lane_id] : 0;

  if (warp_id == 0) {
    value = warpReduceSum<T>(value);
  }

  return value;
}
