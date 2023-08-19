#pragma once

#define FULL_MASK 0xffffffff

template <typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return a + b;
  }
};

template <typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return max(a, b);
  }
};

template <typename T>
struct MinOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return min(a, b);
  }
};

template <typename T, template <typename> class ReduceOp>
__forceinline__ __device__ T warpReduce(T value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = ReduceOp<T>()(value, __shfl_down_sync(FULL_MASK, value, offset));
  }
  return value;
}

template <typename T, template <typename> class ReduceOp>
__forceinline__ __device__ T blockReduce(T value) {
  __shared__ T shared[32];
  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;

  value = warpReduce<T, ReduceOp>(value);
  __syncthreads();

  if (lane_id == 0) {
    shared[warp_id] = value;
  }
  __syncthreads();

  value = (threadIdx.x < blockDim.x / warpSize) ? shared[lane_id] : 0;

  if (warp_id == 0) {
    value = warpReduce<T, ReduceOp>(value);
  }

  return value;
}
