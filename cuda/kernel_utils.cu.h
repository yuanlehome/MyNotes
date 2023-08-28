#pragma once

#include "common.h"

constexpr unsigned int kMask = 0xffffffff;
constexpr unsigned int kWrapSize = 32;
constexpr unsigned int kNumWaves = 32;

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
  __shared__ T smem[kWrapSize];
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;

  value = warpReduce<T, BinaryOp>(value);
  __syncthreads();

  if (lane == 0) {
    smem[warp] = value;
  }
  __syncthreads();

  value = (threadIdx.x < blockDim.x / warpSize) ? smem[threadIdx.x] : 0;

  if (warp == 0) {
    value = warpReduce<T, BinaryOp>(value);
  }

  return value;
}

/**
 * @brief Get the Number of Blocks
 *
 * @param n, number of elements
 * @param block_size, size of block
 * @return number of blocks
 */
inline uint32_t getNumBlocks(uint64_t n, uint32_t block_size) {
  // which device is currently being used.
  int device_id;
  CUDA_CHECK(cudaGetDevice(&device_id));

  // Number of multiprocessors on the device.
  int sm_count;
  CUDA_CHECK(cudaDeviceGetAttribute(
      &sm_count, cudaDevAttrMultiProcessorCount, device_id));

  // Maximum resident threads per multiprocessor.
  int tpm;
  CUDA_CHECK(cudaDeviceGetAttribute(
      &tpm, cudaDevAttrMaxThreadsPerMultiProcessor, device_id));

  return std::max<uint32_t>(
      1,
      std::min<uint64_t>((n + block_size - 1) / block_size,
                         sm_count * tpm / block_size * kNumWaves));
}
