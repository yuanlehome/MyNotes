#include <cuda_fp16.h>
#include <stdint.h>
#include <cassert>
#include <cmath>
#include "stdio.h"

/////////////////////////////////////////////////////////////////////
__device__ inline void fast_cvt_4_packed_signed_i8s_to_2_half2s(
    half halves[4], int8_t signed_chars[4]) {
  uint32_t *h = reinterpret_cast<uint32_t *>(halves);
  uint32_t i8s = *reinterpret_cast<uint32_t *>(signed_chars);

  static constexpr uint32_t mask_for_elt_01 = 0x5150;
  static constexpr uint32_t mask_for_elt_23 = 0x5352;
  static constexpr uint32_t start_byte_for_fp16 = 0x64646464;
  asm volatile("prmt.b32 %0,%1,%2,%3;\n"
               : "=r"(h[0])
               : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
  asm volatile("prmt.b32 %0,%1,%2,%3;\n"
               : "=r"(h[1])
               : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(h[0])
               : "r"(h[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(h[1])
               : "r"(h[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
}

/* Gelu Activation */

__forceinline__ __device__ float copysignf_pos(float a, float b) {
  float r;
  r = __int_as_float(__float_as_int(a) | (__float_as_int(b) & 0x80000000));
  return r;
}

__inline__ __device__ float tanh_opt(float x) {
#if (__CUDA_ARCH__ >= 750 && CUDART_VERSION >= 11000)
  float r;
  asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
  return r;
#else
  const float exp_val = -1.f * fabs(2 * x);
  return copysignf_pos((1.0f - __expf(exp_val)) / (__expf(exp_val) + 1.0f), x);
#endif
}

template <typename T, bool Enable>
struct GeluActivation {
  using return_type = T;
  static __device__ __forceinline__ T apply(const T &val) {
    if (!Enable) return val;
    const float cdf =
        0.5f * (1.0f + tanh_opt((0.7978845608028654f *
                                 (val + 0.044715f * val * val * val))));
    return val * cdf;
  }
};
template <bool Bias, bool Gelu>
__global__ void int8_weight_only_gemv(const int8_t *weight,
                                      const half *input,
                                      const float *scale_list,
                                      const half *bias,
                                      half *output,
                                      const int k,
                                      const int n) {
  constexpr int kWarpSize = 32;
  constexpr int kVecSize = 8;
  half vec_input[kVecSize];
  int8_t vec_weight[kVecSize];
  half vec_weight_f16[kVecSize];

  const int lane_id = threadIdx.x % kWarpSize;
  const int row_id = (blockIdx.x * blockDim.x + threadIdx.x) / kWarpSize;
  if (row_id >= n) return;
  weight += row_id * k;
  float v = 0.f, scale = scale_list[row_id], v_bias;
  if (Bias) {
    v_bias = __half2float(bias[row_id]);
  }
#pragma unroll
  for (int i = lane_id; i < k / kVecSize; i += kWarpSize) {
    *(float4 *)vec_input = *(float4 *)(input + i * kVecSize);
    *(int2 *)vec_weight = *(int2 *)(weight + i * kVecSize);
#pragma unroll
    for (int j = 0; j < kVecSize; j += 4) {
      fast_cvt_4_packed_signed_i8s_to_2_half2s(vec_weight_f16 + j,
                                               vec_weight + j);
#pragma unroll
      for (int p = j; p < j + 4; ++p) {
        v += __half2float(__hmul(vec_input[p], vec_weight_f16[p]));
      }
    }
  }
  v += __shfl_xor_sync(0xffffffff, v, 16);
  v += __shfl_xor_sync(0xffffffff, v, 8);
  v += __shfl_xor_sync(0xffffffff, v, 4);
  v += __shfl_xor_sync(0xffffffff, v, 2);
  v += __shfl_xor_sync(0xffffffff, v, 1);
  if (lane_id == 0) {
    if (Bias) {
      output[row_id] = __float2half_rn(
          GeluActivation<float, Gelu>::apply(v * scale + v_bias));
    } else {
      output[row_id] =
          __float2half_rn(GeluActivation<float, Gelu>::apply(v * scale));
    }
  }
}

template <bool Bias, bool Gelu>
__global__ void int8_weight_only_gemv_smem_cached(const int8_t *weight,
                                                  const half *input,
                                                  const float *scale_list,
                                                  const half *bias,
                                                  half *output,
                                                  const int k,
                                                  const int n) {
  extern __shared__ __align__(sizeof(double)) unsigned char smem_buf[];
  half *input_buf = reinterpret_cast<half *>(smem_buf);

  constexpr int kWarpSize = 32;
  constexpr int kVecSize = 8;

  for (int idx = (blockIdx.x * blockDim.x + threadIdx.x) * kVecSize,
           step = blockDim.x * gridDim.x * kVecSize;
       idx < k;
       idx += step) {
    *reinterpret_cast<float4 *>(input_buf + idx) =
        *reinterpret_cast<const float4 *>(input + idx);
  }
  __syncthreads();

  half vec_input[kVecSize];
  int8_t vec_weight[kVecSize];
  half vec_weight_f16[kVecSize];

  const int lane_id = threadIdx.x % kWarpSize;
  const int row_id = (blockIdx.x * blockDim.x + threadIdx.x) / kWarpSize;
  if (row_id >= n) return;
  weight += row_id * k;
  float v = 0.f, scale = scale_list[row_id], v_bias;
  if (Bias) {
    v_bias = __half2float(bias[row_id]);
  }
#pragma unroll
  for (int i = lane_id; i < k / kVecSize; i += kWarpSize) {
    // *(float4*)vec_input = *(float4*)(input + i * kVecSize);
    *(float4 *)vec_input = *(float4 *)(input_buf + i * kVecSize);

    *(int2 *)vec_weight = *(int2 *)(weight + i * kVecSize);
#pragma unroll
    for (int j = 0; j < kVecSize; j += 4) {
      fast_cvt_4_packed_signed_i8s_to_2_half2s(vec_weight_f16 + j,
                                               vec_weight + j);
#pragma unroll
      for (int p = j; p < j + 4; ++p) {
        v += __half2float(__hmul(vec_input[p], vec_weight_f16[p]));
      }
    }
  }
  v += __shfl_xor_sync(0xffffffff, v, 16);
  v += __shfl_xor_sync(0xffffffff, v, 8);
  v += __shfl_xor_sync(0xffffffff, v, 4);
  v += __shfl_xor_sync(0xffffffff, v, 2);
  v += __shfl_xor_sync(0xffffffff, v, 1);
  if (lane_id == 0) {
    if (Bias) {
      output[row_id] = __float2half_rn(
          GeluActivation<float, Gelu>::apply(v * scale + v_bias));
    } else {
      output[row_id] =
          __float2half_rn(GeluActivation<float, Gelu>::apply(v * scale));
    }
  }
}

void int8_weight_only_gemv_smem_cached_launcher(const int8_t *weight,
                                                const half *input,
                                                const float *scale_list,
                                                const half *bias,
                                                half *output,
                                                const int k,
                                                const int n,
                                                const bool gelu,
                                                cudaStream_t stream) {
  dim3 block(512);
  dim3 grid(n / 16);
  const int smem_size = (sizeof(half) * k + 4 - 1) / 4 * 4;

  if (bias) {
    if (gelu) {
      int8_weight_only_gemv_smem_cached<true, true>
          <<<grid, block, smem_size, stream>>>(
              weight, input, scale_list, bias, output, k, n);
    } else {
      int8_weight_only_gemv_smem_cached<true, false>
          <<<grid, block, smem_size, stream>>>(
              weight, input, scale_list, bias, output, k, n);
    }
  } else {
    if (gelu) {
      int8_weight_only_gemv_smem_cached<false, true>
          <<<grid, block, smem_size, stream>>>(
              weight, input, scale_list, bias, output, k, n);
    } else {
      int8_weight_only_gemv_smem_cached<false, false>
          <<<grid, block, smem_size, stream>>>(
              weight, input, scale_list, bias, output, k, n);
    }
  }
}

void int8_weight_only_gemv_launcher(const int8_t *weight,
                                    const half *input,
                                    const float *scale_list,
                                    const half *bias,
                                    half *output,
                                    const int k,
                                    const int n,
                                    const bool gelu,
                                    cudaStream_t stream) {
  dim3 block(512);
  dim3 grid(n / 16);
  if (bias) {
    if (gelu) {
      int8_weight_only_gemv<true, true><<<grid, block, 0, stream>>>(
          weight, input, scale_list, bias, output, k, n);
    } else {
      int8_weight_only_gemv<true, false><<<grid, block, 0, stream>>>(
          weight, input, scale_list, bias, output, k, n);
    }
  } else {
    if (gelu) {
      int8_weight_only_gemv<false, true><<<grid, block, 0, stream>>>(
          weight, input, scale_list, bias, output, k, n);
    } else {
      int8_weight_only_gemv<false, false><<<grid, block, 0, stream>>>(
          weight, input, scale_list, bias, output, k, n);
    }
  }
}

int main(int argc, char *argv[]) {
  size_t m = strtol(argv[1], nullptr, 0);
  size_t n = strtol(argv[2], nullptr, 0);
  size_t k = strtol(argv[3], nullptr, 0);

  half *input;
  int8_t *weight;
  float *weight_scale;
  half *bias;
  half *output;
  cudaStream_t stream;
  cudaMalloc(&input, sizeof(half) * m * k);
  cudaMalloc(&weight, sizeof(int8_t) * k * n);
  cudaMalloc(&bias, sizeof(half) * n);
  cudaMalloc(&weight_scale, sizeof(float) * k * n);
  cudaMalloc(&output, sizeof(half) * m * n);
  cudaStreamCreate(&stream);

  int8_weight_only_gemv_launcher(
      weight, input, weight_scale, nullptr, output, k, n, false, stream);

  int8_weight_only_gemv_smem_cached_launcher(
      weight, input, weight_scale, nullptr, output, k, n, false, stream);
  cudaDeviceSynchronize();
  cudaFree(input);
  cudaFree(weight);
  cudaFree(bias);
  cudaFree(weight_scale);
  cudaFree(output);
  cudaStreamDestroy(stream);
  return 0;
}
