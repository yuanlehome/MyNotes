#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include "kernel_caller_declare.h"

constexpr DATA_TYPE a = 1.23;
constexpr DATA_TYPE b = 2.34;
constexpr DATA_TYPE c = 3.57;
constexpr DATA_TYPE EPSILON = 1.0e-15;

void check_data(const DATA_TYPE* z, const int N) {
  bool has_error = false;
  for (size_t i = 0; i < N; i++) {
    if (fabs(z[i] - c) > EPSILON) {
      has_error = true;
      break;
    }
  }
  std::printf("%s\n", has_error ? "Has errors." : "No errors.");
}

void add_array_cpu(const DATA_TYPE* x,
                   const DATA_TYPE* y,
                   DATA_TYPE* z,
                   const int N) {
  for (size_t i = 0; i < N; i++) {
    z[i] = x[i] + y[i];
  }
}

__global__ void add_array_gpu(const DATA_TYPE* x,
                              const DATA_TYPE* y,
                              DATA_TYPE* z,
                              const int N) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < N) {
    z[tid] = x[tid] + y[tid];
  }
}

void add_array() {
  constexpr int N = 1e8;
  constexpr int M = sizeof(DATA_TYPE) * N;

  DATA_TYPE* h_x = reinterpret_cast<DATA_TYPE*>(std::malloc(M));
  DATA_TYPE* h_y = reinterpret_cast<DATA_TYPE*>(std::malloc(M));
  DATA_TYPE* h_z = reinterpret_cast<DATA_TYPE*>(std::malloc(M));

  std::fill_n(h_x, N, a);
  std::fill_n(h_y, N, b);

  DATA_TYPE *d_x, *d_y, *d_z;
  cudaMalloc((void**)&d_x, M);
  cudaMalloc((void**)&d_y, M);
  cudaMalloc((void**)&d_z, M);

  cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);

  add_array_cpu(h_x, h_y, h_z, N);
  check_data(h_z, N);

  const int block_size(128);
  const int grid_size = (N + block_size - 1) / block_size;
  add_array_gpu<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
  cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
  check_data(h_z, N);

  std::free(h_x);
  std::free(h_y);
  std::free(h_z);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
}
