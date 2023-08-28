#pragma once

constexpr unsigned int kMask = 0xffffffff;
constexpr unsigned int kWrapSize = 32;

template <typename T>
struct AddOp {
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

template <typename T, template <typename> class BinaryOp>
__forceinline__ __device__ T warpReduce(T value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = BinaryOp<T>()(value, __shfl_down_sync(kMask, value, offset));
  }
  return value;
}

template <typename T, template <typename> class BinaryOp>
__forceinline__ __device__ T blockReduce(T value) {
  __shared__ T shared[kWrapSize];
  int lane = threadIdx.x % warpSize;
  int warp = threadIdx.x / warpSize;

  value = warpReduce<T, BinaryOp>(value);
  __syncthreads();

  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();

  value = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (warp == 0) {
    value = warpReduce<T, BinaryOp>(value);
  }

  return value;
}
