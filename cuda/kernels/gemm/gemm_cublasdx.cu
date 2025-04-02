// https://docs.nvidia.com/cuda/cublasdx/index.html

#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

#include <cuda_runtime_api.h>
#include <cublasdx.hpp>

#define CUDA_CHECK_AND_EXIT(error)                                      \
  {                                                                     \
    auto status = static_cast<cudaError_t>(error);                      \
    if (status != cudaSuccess) {                                        \
      std::cout << cudaGetErrorString(status) << " " << __FILE__ << ":" \
                << __LINE__ << std::endl;                               \
      std::exit(status);                                                \
    }                                                                   \
  }

template <typename T>
std::vector<T> get_random_data(const float min,
                               const float max,
                               const size_t size) {
  std::default_random_engine e;
  std::uniform_real_distribution<float> u(min, max);

  std::vector<T> ret(size);
  for (auto& v : ret) {
    v = static_cast<T>(u(e));
  }
  return ret;
}

namespace example {
template <class T>
inline __device__ void naive_copy(T* dest, const T* src, unsigned int size) {
  if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {
    // Note: This copies values in padding too
    for (unsigned int idx = 0; idx < size; ++idx) {
      dest[idx] = src[idx];
    }
  }
}
template <class BLAS, class ValueType = typename BLAS::value_type>
struct io {
  using value_type = ValueType;

  template <class T>
  static inline __device__ void load(T* shared_output,
                                     const value_type* global_input,
                                     const unsigned int size) {
    naive_copy(
        reinterpret_cast<value_type*>(shared_output), global_input, size);
  }

  template <class T>
  static inline __device__ void store(value_type* global_output,
                                      const T* shared_input,
                                      const unsigned int size) {
    naive_copy(
        global_output, reinterpret_cast<const value_type*>(shared_input), size);
  }
};
}  // namespace example

template <class BLAS1,
          class BLAS2,
          class ValueType = typename BLAS1::value_type>
__launch_bounds__(BLAS1::max_threads_per_block)  //
    __global__                                   //
    void gemm_kernel(const ValueType alpha1,
                     const ValueType* a,
                     const ValueType* b,
                     const ValueType beta1,
                     const ValueType* c,
                     const ValueType alpha2,
                     const ValueType* d,
                     const ValueType beta2,
                     const ValueType* f,
                     ValueType* output) {
  using value_type = ValueType;
  extern __shared__ __align__(16) char smem[];

  // Matrix C is the first in shared memory, because it's reused in the 2nd
  // GEMM. Moreover, matrices A and B might have different sizes than F and D.
  value_type* smem_c = reinterpret_cast<value_type*>(smem);
  value_type* smem_a = reinterpret_cast<value_type*>(smem) + BLAS1::c_size;
  value_type* smem_b =
      reinterpret_cast<value_type*>(smem) + BLAS1::c_size + BLAS1::a_size;

  example::io<BLAS1>::load(smem_a, a, BLAS1::a_size);
  example::io<BLAS1>::load(smem_b, b, BLAS1::b_size);
  example::io<BLAS1>::load(smem_c, c, BLAS1::c_size);
  __syncthreads();

  BLAS1().execute(alpha1, smem_a, smem_b, beta1, smem_c);
  __syncthreads();

  static_assert((BLAS1::c_size == BLAS2::a_size),
                "The sizes of C matrix are different in BLAS1 and BLAS2");
  value_type* smem_d = smem_c + BLAS2::a_size;
  value_type* smem_f = smem_c + BLAS2::a_size + BLAS2::b_size;
  example::io<BLAS2>::load(smem_d, d, BLAS2::b_size);
  example::io<BLAS2>::load(smem_f, f, BLAS2::c_size);
  __syncthreads();

  BLAS2().execute(alpha2, smem_c, smem_d, beta2, smem_f);

  __syncthreads();
  example::io<BLAS2>::store(output, smem_f, BLAS2::c_size);
}

//             1) C = alpha1 * (A * B) + beta1 * C
//             2) F = alpha2 * (C * D) + beta2 * F
int simple_gemm() {
  // Parameters m1, n1, k1 define the dimensions of matrices A, B, and C
  constexpr unsigned int m1 = 64;
  constexpr unsigned int n1 = 64;
  constexpr unsigned int k1 = 64;

  // Parameters m2, n2, k2 define the dimensions of matrices C, D and F
  // Note: (m1, n1) and (m2, k2) must be equal as describe the same matrix
  // (matrix C)
  constexpr unsigned int m2 = m1;
  constexpr unsigned int n2 = 128;
  constexpr unsigned int k2 = n1;

  // The logical dimensions of matrix A are: [m1, k1] (m rows, k columns)
  // The logical dimensions of matrix B are: [k1, n1]
  // The logical dimensions of matrix C are: [m1, n1]
  constexpr auto a_transpose_mode = cublasdx::transpose_mode::non_transposed;
  constexpr auto b_transpose_mode = cublasdx::transpose_mode::non_transposed;

  // The logical dimensions of matrix C are: [m2, k2] == [m1, n1]
  // The logical dimensions of matrix D are: [k2, n2]
  // The logical dimensions of matrix F are: [m2, n2]
  constexpr auto c_transpose_mode = cublasdx::transpose_mode::non_transposed;
  constexpr auto d_transpose_mode = cublasdx::transpose_mode::non_transposed;

  // Use the same block size for both GEMM operations, so BLAS1::block_dim ==
  // BLAS2::block_dim which simplifies the example.
  constexpr unsigned int block_size = 128;

  using BLAS1 =
      decltype(cublasdx::Size<m1, n1, k1>() + cublasdx::Precision<__half>() +
               cublasdx::Type<cublasdx::type::real>() +
               cublasdx::Function<cublasdx::function::MM>() +
               cublasdx::TransposeMode<a_transpose_mode, b_transpose_mode>() +
               cublasdx::Block() + cublasdx::BlockDim<block_size>() +
               cublasdx::SM<800>());
  using BLAS2 =
      decltype(cublasdx::Size<m2, n2, k2>() + cublasdx::Precision<__half>() +
               cublasdx::Type<cublasdx::type::real>() +
               cublasdx::Function<cublasdx::function::MM>() +
               cublasdx::TransposeMode<c_transpose_mode, d_transpose_mode>() +
               cublasdx::Block() + cublasdx::BlockDim<block_size>() +
               cublasdx::SM<800>());
  using value_type = typename BLAS1::value_type;

  // alpha and beta for the first GEMM
  value_type alpha1 = 1.0;
  value_type beta1 = 0.0;

  // alpha and beta for the 2nd GEMM
  value_type alpha2 = 1.0;
  value_type beta2 = 1.0;

  // Allocate managed memory for a, b, c, d, f and output
  value_type* inputs;
  value_type* output;
  auto inputs_size = BLAS1::a_size + BLAS1::b_size + BLAS1::c_size +
                     BLAS2::b_size + BLAS2::c_size;
  auto inputs_size_bytes = inputs_size * sizeof(value_type);
  CUDA_CHECK_AND_EXIT(cudaMalloc(&inputs, inputs_size_bytes));
  CUDA_CHECK_AND_EXIT(cudaMalloc(&output, BLAS2::c_size * sizeof(value_type)));

  value_type* a = inputs;
  value_type* b = a + (BLAS1::a_size);
  value_type* c =
      b + (BLAS1::b_size);  // C matrix for BLAS1, A matrix for BLAS2
  value_type* d = c + (BLAS1::c_size);  // D is B matrix for BLAS2
  value_type* f = d + (BLAS2::b_size);  // F is C matrix for BLAS2

  // Fill the A, B, C matrices with random values
  auto host_a = get_random_data<value_type>(0.1, 1.0, BLAS1::a_size);
  auto host_b = get_random_data<value_type>(0.1, 1.0, BLAS1::b_size);
  auto host_c = get_random_data<value_type>(0.1, 1.0, BLAS1::c_size);
  auto host_d = get_random_data<value_type>(1.0, 2.0, BLAS2::b_size);
  auto host_f = get_random_data<value_type>(1.0, 10.0, BLAS2::c_size);
  CUDA_CHECK_AND_EXIT(cudaMemcpy(a,
                                 host_a.data(),
                                 BLAS1::a_size * sizeof(value_type),
                                 cudaMemcpyHostToDevice));
  CUDA_CHECK_AND_EXIT(cudaMemcpy(b,
                                 host_b.data(),
                                 BLAS1::b_size * sizeof(value_type),
                                 cudaMemcpyHostToDevice));
  CUDA_CHECK_AND_EXIT(cudaMemcpy(c,
                                 host_c.data(),
                                 BLAS1::c_size * sizeof(value_type),
                                 cudaMemcpyHostToDevice));
  CUDA_CHECK_AND_EXIT(cudaMemcpy(d,
                                 host_d.data(),
                                 BLAS2::b_size * sizeof(value_type),
                                 cudaMemcpyHostToDevice));
  CUDA_CHECK_AND_EXIT(cudaMemcpy(f,
                                 host_f.data(),
                                 BLAS2::c_size * sizeof(value_type),
                                 cudaMemcpyHostToDevice));
  CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

  // Increase max dynamic shared memory for the kernel if needed
  const auto shared_memory =
      std::max<size_t>(BLAS1::shared_memory_size, BLAS2::shared_memory_size);
  CUDA_CHECK_AND_EXIT(
      cudaFuncSetAttribute(gemm_kernel<BLAS1, BLAS2>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           shared_memory));

  // Execute kernel
  gemm_kernel<BLAS1, BLAS2><<<1, BLAS1::block_dim, shared_memory>>>(
      alpha1, a, b, beta1, c, alpha2, d, beta2, f, output);
  CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

  // Copy results back to host
  std::vector<value_type> host_output(BLAS2::c_size);
  CUDA_CHECK_AND_EXIT(cudaMemcpy(host_output.data(),
                                 output,
                                 BLAS2::c_size * sizeof(value_type),
                                 cudaMemcpyDeviceToHost));
  CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

  // Free device memory
  CUDA_CHECK_AND_EXIT(cudaFree(inputs));
  CUDA_CHECK_AND_EXIT(cudaFree(output));
  return 1;
}

int main(int, char**) {
  simple_gemm();
  printf("run success!\n");
}
