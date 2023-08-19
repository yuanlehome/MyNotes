#include "common.h"
#include "kernel_caller_declare.h"

__global__ void helloWorld() {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  std::printf("Hello world from gpu int block %d and thread %d.\n", bid, tid);
}

__global__ void helloWorldV2() {
  const int bid_x = blockIdx.x;
  const int bid_y = blockIdx.y;
  const int tid_x = threadIdx.x;
  const int tid_y = threadIdx.y;
  std::printf("Hello world from gpu int block (%d, %d) and thread (%d, %d).\n",
         bid_x,
         bid_y,
         tid_x,
         tid_y);
}

void printHelloWorld() {
  helloWorld<<<2, 4>>>();

  dim3 block_size(2, 4);
  dim3 grid_size(1, 1);
  helloWorldV2<<<grid_size, block_size>>>();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}
