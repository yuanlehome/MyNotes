#define FULL_MASK 0xffffffff

template <typename T>
__forceinline__ __device__ T warpReduceSum(T val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(FULL_MASK, val, offset);
  }
  return val;
}

template <typename T>
__forceinline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum<T>(val);
  __syncthreads();
  if (lane == 0) shared[wid] = val;

  __syncthreads();
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid == 0) val = warpReduceSum<T>(val);

  return val;
}
