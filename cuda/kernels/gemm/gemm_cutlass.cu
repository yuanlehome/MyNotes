#include "all.h"
#include "common.h"
#include "kernel_utils.cu.h"

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

// C = alpha * A * B + beta * C
cutlass::Status matrixMultiplyKernel_CUTLASS(int M,
                                             int N,
                                             int K,
                                             DATA_TYPE alpha,
                                             DATA_TYPE const* A,
                                             int lda,
                                             DATA_TYPE const* B,
                                             int ldb,
                                             DATA_TYPE beta,
                                             DATA_TYPE* C,
                                             int ldc) {
  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size (chosen by default).
  //
  // To keep the interface manageable, several helpers are defined for plausible
  // compositions
  // including the following example for single-precision GEMM. Typical values
  // are used as
  // default template arguments. See
  // `cutlass/gemm/device/default_gemm_configuration.h` for more details.
  //
  // To view the full gemm device API interface, see
  // `cutlass/gemm/device/gemm.h`

  using ColumnMajor = cutlass::layout::ColumnMajor;
  using RowMajor = cutlass::layout::RowMajor;

  using EpilogueOutputOp =
      cutlass::epilogue::thread::LinearCombination<float, 1, float, float>;

  using Gemm = cutlass::gemm::device::Gemm<
      DATA_TYPE,  // Data-type of A matrix
      RowMajor,   // Layout of A matrix
      DATA_TYPE,  // Data-type of B matrix
      RowMajor,   // Layout of B matrix
      DATA_TYPE,  // Data-type of C matrix
      RowMajor,
      DATA_TYPE,
      cutlass::arch::OpClassSimt,
      cutlass::arch::Sm86,
      cutlass::gemm::GemmShape<128, 128, 8>,
      cutlass::gemm::GemmShape<64, 64, 8>,
      cutlass::gemm::GemmShape<1, 1, 1>,
      EpilogueOutputOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      2,
      1,
      1>;

  // Define a CUTLASS GEMM type
  Gemm gemm;

  // Construct the CUTLASS GEMM arguments object.
  //
  // One of CUTLASS's design patterns is to define gemm argument objects that
  // are constructible
  // in host code and passed to kernels by value. These may include pointers,
  // strides, scalars,
  // and other arguments needed by Gemm and its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy for
  // passing host-constructible
  // arguments to kernels and (2.) minimized initialization overhead on kernel
  // entry.
  //
  Gemm::Arguments args({M, N, K},  // Gemm Problem dimensions
                       {A, lda},   // Tensor-ref for source matrix A
                       {B, ldb},   // Tensor-ref for source matrix B
                       {C, ldc},   // Tensor-ref for source matrix C
                       {C, ldc},   // Tensor-ref for destination matrix D
                       // (may be different memory than source
                       // C matrix)
                       {alpha, beta});  // Scalars used in the Epilogue

  cutlass::Status status = gemm(args);

  return status;
}

void gemm_cutlass() {
  constexpr uint32_t M = 1024;
  constexpr uint32_t N = 1024;
  constexpr uint32_t K = 512;

  constexpr uint32_t A_SIZE = sizeof(DATA_TYPE) * M * K;
  constexpr uint32_t B_SIZE = sizeof(DATA_TYPE) * K * N;
  constexpr uint32_t C_SIZE = sizeof(DATA_TYPE) * M * N;

  CPUMallocWrapper cpu_allocator;
  DATA_TYPE* h_a = (DATA_TYPE*)cpu_allocator.allocate(A_SIZE);
  DATA_TYPE* h_b = (DATA_TYPE*)cpu_allocator.allocate(B_SIZE);
  DATA_TYPE* h_c = (DATA_TYPE*)cpu_allocator.allocate(C_SIZE);
  std::fill_n(h_a, M * K, 0.5);
  std::fill_n(h_b, K * N, 0.3);
  std::fill_n(h_c, M * N, 0.0);

  GPUMallocWrapper gpu_allocator;
  DATA_TYPE* d_a = (DATA_TYPE*)gpu_allocator.allocate(A_SIZE);
  DATA_TYPE* d_b = (DATA_TYPE*)gpu_allocator.allocate(B_SIZE);
  DATA_TYPE* d_c = (DATA_TYPE*)gpu_allocator.allocate(C_SIZE);
  CUDA_CHECK(cudaMemcpy(d_a, h_a, A_SIZE, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, B_SIZE, cudaMemcpyHostToDevice));
  utils::fill_n(d_c, M * N, 0.0);

  // CPU results
  DATA_TYPE* real_c = (DATA_TYPE*)cpu_allocator.allocate(C_SIZE);
  std::fill_n(real_c, M * N, 0.0);
  utils::matrixMultiply(h_a, h_b, real_c, M, N, K);

  utils::performance<GPUTimer>(
      "matrixMultiplyKernel_CUTLASS",
      repeats,
      [&] { utils::fill_n(d_c, M * N, 0.0); },
      [&] {
        matrixMultiplyKernel_CUTLASS(M, N, K, 1.0, d_a, K, d_b, N, 0.0, d_c, N);
      },
      [&] {
        utils::fill_n(d_c, M * N, 0.0);
        cutlass::Status status;
        status = matrixMultiplyKernel_CUTLASS(
            M, N, K, 1.0, d_a, K, d_b, N, 0.0, d_c, N);
        dbg(cutlass::cutlassGetStatusString(status));
        CUDA_CHECK(cudaMemcpy(h_c, d_c, C_SIZE, cudaMemcpyDeviceToHost));
        dbg(utils::checkEqual(h_c, real_c, M * N));
      });
}
