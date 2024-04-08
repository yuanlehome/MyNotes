#include "all.h"
#include "common.h"
#include "kernel_utils.cu.h"

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

cudaError_t cutlassKernel(int M,
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
      cutlass::arch::Sm80,
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

  //
  // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
  //

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

void gemm_cutlass() {
  constexpr uint32_t M = 1024;
  constexpr uint32_t N = 2048;
  constexpr uint32_t K = 512;

  constexpr uint32_t A_SIZE = sizeof(DATA_TYPE) * M * K;
  constexpr uint32_t B_SIZE = sizeof(DATA_TYPE) * K * N;
  constexpr uint32_t C_SIZE = sizeof(DATA_TYPE) * M * N;

  MallocWrapper cpu_allocator;
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

  // CPU results
  DATA_TYPE* real_c = (DATA_TYPE*)cpu_allocator.allocate(C_SIZE);
  std::fill_n(real_c, M * N, 0.0);
  matrixMultiplyOnCPU(h_a, h_b, real_c, M, N, K);

  GPUTimer gpu_timer;
  float total_time = 0.0;
  for (size_t i = 0; i < repeats; i++) {
    fillNKernel<<<1024, (M * N + 1024 - 1) / 1024>>>(d_c, M * N, 0.0);
    gpu_timer.start();
    CUDA_CHECK(cutlassKernel(M, N, K, 1.0, d_a, K, d_b, N, 0.0, d_c, N));
    gpu_timer.stop();
    total_time += gpu_timer.elapsedTime();
  }
  dbg(total_time, gpu_timer.totalTime());
  CUDA_CHECK(cudaMemcpy(h_c, d_c, C_SIZE, cudaMemcpyDeviceToHost));
  dbg(checkEqual(h_c, real_c, M * N));
  std::printf("cutlassKernel cost time: %f ms\n", total_time / repeats);
}
