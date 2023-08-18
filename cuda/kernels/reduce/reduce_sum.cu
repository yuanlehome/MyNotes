#include "dbg.h"

#include "common.h"
#include "kernel_caller_declare.h"
#include "kernel_utils.cu.h"

constexpr int BLOCK_SIZE = 128;

constexpr DATA_TYPE a = 1.23;

// 数值错误
// Add x[start...end] 左闭右闭
DATA_TYPE reduceSumOnCPU_V1(const DATA_TYPE* x,
                            const int start,
                            const int end) {
  DATA_TYPE sum{0.0};
  for (int i = start; i <= end; i++) {
    // 大数加小数
    sum += x[i];
  }
  return sum;
}

// 数值正确 不修改原数组
// Add x[start...end] 左闭右闭
DATA_TYPE reduceSumOnCPU_V2(const DATA_TYPE* x,
                            const int start,
                            const int end) {
  if (start > end) return DATA_TYPE{};
  if (start == end) return x[start];
  const int p = (start + end) / 2;
  // 递归
  return reduceSumOnCPU_V2(x, start, p) + reduceSumOnCPU_V2(x, p + 1, end);
}

// 数值正确 修改原数组
// Add x[start...end] 左闭右闭
DATA_TYPE reduceSumOnCPU_V3(DATA_TYPE* x, int start, int end) {
  while (start < end) {
    // 双指针
    int i = start, j = end;
    while (i < j) {
      x[i++] += x[j--];
    }
    end = i == j ? i : i - 1;
  }
  return x[start];
}

// 数值错误 要求数据个数为 BLOCK_SIZE 的整数倍 改变原数组
// 每个 block 负责一块内存数据的 reduce
__global__ void reduceSumOnGPU_V1(DATA_TYPE* d_x, DATA_TYPE* d_y) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  DATA_TYPE* x = d_x + blockDim.x * bid;

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      x[tid] += x[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(d_y, x[0]);
  }
}

// 数值错误 不要求数据个数为 BLOCK_SIZE 的整数倍 不改变原数组
// 每个 block 负责一块内存数据 使用动态共享内存
__global__ void reduceSumOnGPU_V2(const DATA_TYPE* d_x,
                                  DATA_TYPE* d_y,
                                  const int N) {
  extern __shared__ DATA_TYPE s_y[];
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int idx = tid + blockDim.x * bid;
  s_y[tid] = idx < N ? d_x[idx] : 0.0;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_y[tid] += s_y[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(d_y, s_y[0]);
  }
}

// 数值错误 不要求数据个数为 BLOCK_SIZE 的整数倍 不改变原数组
// 每个 block 负责一块内存数据 使用动态共享内存 使用 __syncwarp 替换
// __syncthreads
__global__ void reduceSumOnGPU_V3(const DATA_TYPE* d_x,
                                  DATA_TYPE* d_y,
                                  const int N) {
  extern __shared__ DATA_TYPE s_y[];
  const int tid = threadIdx.x;
  const int idx = tid + blockDim.x * blockIdx.x;
  s_y[tid] = idx < N ? d_x[idx] : 0.0;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
    if (tid < offset) {
      s_y[tid] += s_y[tid + offset];
    }
    __syncthreads();
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_y[tid] += s_y[tid + offset];
    }
    __syncwarp();
  }

  if (tid == 0) {
    atomicAdd(d_y, s_y[0]);
  }
}

// 数值错误 不要求数据个数为 BLOCK_SIZE 的整数倍 不改变原数组
// 每个 block 负责一块内存数据的 reduce 调用 block reduce function
__global__ void reduceSumOnGPU_V4(const DATA_TYPE* d_x,
                                  DATA_TYPE* d_y,
                                  const int N) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  DATA_TYPE val = idx < N ? d_x[idx] : 0.0;
  val = blockReduceSum(val);
  if (threadIdx.x == 0) {
    atomicAdd(d_y, val);
  }
}

// V5 kernel 的计算有问题 但没想明白哪里的问题 待定!!!
// 数值错误 不要求数据个数为 BLOCK_SIZE 的整数倍 不改变原数组
// 每个 block 负责连续的两块内存数据的 reduce 调用 block reduce function
__global__ void reduceSumOnGPU_V5(const DATA_TYPE* d_x,
                                  DATA_TYPE* d_y,
                                  const int N) {
  const int idx = threadIdx.x + 2 * blockDim.x * blockIdx.x;
  DATA_TYPE val = 0.0;
  if (idx < N) {
    val += d_x[idx];
  }
  if (idx + blockDim.x < N) {
    val += d_x[idx + blockDim.x];
    // val += d_x[idx + 2 * blockDim.x];
    // val += d_x[idx + 3 * blockDim.x];
  }
  val = blockReduceSum(val);
  if (threadIdx.x == 0) {
    atomicAdd(d_y, val);
  }
}

// 数值正确 需二次 reduce 不要求数据个数为 BLOCK_SIZE 的整数倍 不改变原数组
// 每个 block 负责不连续的多块内存数据的 reduce 其中每个线程负责跨度为整个 grid
// 的内存 调用 warp/block reduce function
__global__ void reduceSumOnGPU_V6(const DATA_TYPE* d_x,
                                  DATA_TYPE* d_y,
                                  const int N) {
  DATA_TYPE val = 0.0;
  const int stride = gridDim.x * blockDim.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (idx < N) {
    val += d_x[idx];
    idx += stride;
  }
  val = blockReduceSum(val);

  if (threadIdx.x == 0) {
    d_y[blockIdx.x] = val;
  }
}

void reduceSum() {
  constexpr uint32_t N = 1e8;
  constexpr uint32_t SIZE = sizeof(DATA_TYPE) * N;

  MallocWrapper cpu_allocator;
  DATA_TYPE* h_x = (DATA_TYPE*)cpu_allocator.allocate(SIZE);
  std::fill_n(h_x, N, a);

  // Timer cpu_timer;
  // float total_time = 0.0;
  // DATA_TYPE sum_on_cpu;
  // for (size_t i = 0; i < repeats; i++) {
  //   cpu_timer.start();
  //   sum_on_cpu = reduceSumOnCPU_V1(h_x, 0, N - 1);
  //   cpu_timer.stop();
  //   total_time += cpu_timer.elapsedTime();
  // }
  // dbg(sum_on_cpu);
  // std::printf("reduceSumOnCPU_V1 cost time: %f ms\n", total_time / repeats);

  // total_time = 0.0;
  // for (size_t i = 0; i < repeats; i++) {
  //   cpu_timer.start();
  //   sum_on_cpu = reduceSumOnCPU_V2(h_x, 0, N - 1);
  //   cpu_timer.stop();
  //   total_time += cpu_timer.elapsedTime();
  // }
  // dbg(sum_on_cpu);
  // std::printf("reduceSumOnCPU_V2 cost time: %f ms\n", total_time / repeats);

  // total_time = 0.0;
  // for (size_t i = 0; i < repeats; i++) {
  //   cpu_timer.start();
  //   sum_on_cpu = reduceSumOnCPU_V3(h_x, 0, N - 1);
  //   cpu_timer.stop();
  //   total_time += cpu_timer.elapsedTime();
  //   std::fill_n(h_x, N, a);  // 恢复原数组
  // }
  // dbg(sum_on_cpu);
  // std::printf("reduceSumOnCPU_V3 cost time: %f ms\n", total_time / repeats);

  const uint32_t block_size = BLOCK_SIZE;
  const uint32_t grid_size = (N + block_size - 1) / block_size;
  dbg(block_size, grid_size);
  dim3 block(block_size);
  dim3 grid(grid_size);

  GPUMallocWrapper gpu_allocator;
  DATA_TYPE* d_x = (DATA_TYPE*)gpu_allocator.allocate(SIZE);
  CUDA_CHECK(cudaMemcpy(d_x, h_x, SIZE, cudaMemcpyHostToDevice));

  DATA_TYPE* d_y = (DATA_TYPE*)gpu_allocator.allocate(sizeof(DATA_TYPE));

  GPUTimer gpu_timer;
  DATA_TYPE y = 0.0;
  float total_time = 0.0;
  for (size_t i = 0; i < repeats; i++) {
    CUDA_CHECK(cudaMemcpy(
        d_y, &y, sizeof(DATA_TYPE), cudaMemcpyHostToDevice));  // 清空结果
    gpu_timer.start();
    reduceSumOnGPU_V1<<<grid, block>>>(d_x, d_y);
    gpu_timer.stop();
    total_time += gpu_timer.elapsedTime();
    CUDA_CHECK(
        cudaMemcpy(d_x, h_x, SIZE, cudaMemcpyHostToDevice));  // 恢复原数组
  }
  dbg(total_time, gpu_timer.totalTime());
  CUDA_CHECK(cudaMemcpy(&y, d_y, sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));
  dbg(y);
  std::printf("reduceSumOnGPU_V1 cost time: %f ms\n", total_time / repeats);

  y = 0.0;
  total_time = 0.0;
  for (size_t i = 0; i < repeats; i++) {
    CUDA_CHECK(cudaMemcpy(
        d_y, &y, sizeof(DATA_TYPE), cudaMemcpyHostToDevice));  // 清空结果
    gpu_timer.start();
    reduceSumOnGPU_V2<<<grid, block, sizeof(DATA_TYPE) * block_size>>>(
        d_x, d_y, N);
    gpu_timer.stop();
    total_time += gpu_timer.elapsedTime();
  }
  dbg(total_time, gpu_timer.totalTime());
  CUDA_CHECK(cudaMemcpy(&y, d_y, sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));
  dbg(y);
  std::printf("reduceSumOnGPU_V2 (dynamic shared memory) cost time: %f ms\n",
              total_time / repeats);

  y = 0.0;
  total_time = 0.0;
  for (size_t i = 0; i < repeats; i++) {
    CUDA_CHECK(cudaMemcpy(
        d_y, &y, sizeof(DATA_TYPE), cudaMemcpyHostToDevice));  // 清空结果
    gpu_timer.start();
    reduceSumOnGPU_V3<<<grid, block, sizeof(DATA_TYPE) * block_size>>>(
        d_x, d_y, N);
    gpu_timer.stop();
    total_time += gpu_timer.elapsedTime();
  }
  dbg(total_time, gpu_timer.totalTime());
  CUDA_CHECK(cudaMemcpy(&y, d_y, sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));
  dbg(y);
  std::printf("reduceSumOnGPU_V3 (__syncwarp) cost time: %f ms\n",
              total_time / repeats);

  y = 0.0;
  total_time = 0.0;
  for (size_t i = 0; i < repeats; i++) {
    CUDA_CHECK(cudaMemcpy(
        d_y, &y, sizeof(DATA_TYPE), cudaMemcpyHostToDevice));  // 清空结果
    gpu_timer.start();
    reduceSumOnGPU_V4<<<grid, block>>>(d_x, d_y, N);
    gpu_timer.stop();
    total_time += gpu_timer.elapsedTime();
  }
  dbg(total_time, gpu_timer.totalTime());
  CUDA_CHECK(cudaMemcpy(&y, d_y, sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));
  dbg(y);
  std::printf("reduceSumOnGPU_V4 (warp/block reduce) cost time: %f ms\n",
              total_time / repeats);

  y = 0.0;
  total_time = 0.0;
  for (size_t i = 0; i < repeats; i++) {
    CUDA_CHECK(cudaMemcpy(
        d_y, &y, sizeof(DATA_TYPE), cudaMemcpyHostToDevice));  // 清空结果
    gpu_timer.start();
    // 注意这里 grid 大小减半 但要向上取整
    reduceSumOnGPU_V5<<<(grid.x + 1) / 2, block>>>(d_x, d_y, N);
    gpu_timer.stop();
    total_time += gpu_timer.elapsedTime();
  }
  dbg(total_time, gpu_timer.totalTime());
  CUDA_CHECK(cudaMemcpy(&y, d_y, sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));
  dbg(y);
  std::printf("reduceSumOnGPU_V5 (连续两块) cost time: %f ms\n",
              total_time / repeats);

  y = 0.0;
  total_time = 0.0;
  for (size_t i = 0; i < repeats; i++) {
    CUDA_CHECK(cudaMemcpy(
        d_y, &y, sizeof(DATA_TYPE), cudaMemcpyHostToDevice));  // 清空结果
    gpu_timer.start();
    // 注意这里传的 d_x 和 d_y 相同 如果 block 之间没有同步的话 会有问题
    reduceSumOnGPU_V6<<<10240, block_size>>>(d_x, d_x, N);
    reduceSumOnGPU_V6<<<1, 1024>>>(d_x, d_y, 10240);
    gpu_timer.stop();
    total_time += gpu_timer.elapsedTime();
    CUDA_CHECK(
        cudaMemcpy(d_x, h_x, SIZE, cudaMemcpyHostToDevice));  // 恢复原数组
  }
  dbg(total_time, gpu_timer.totalTime());
  CUDA_CHECK(cudaMemcpy(&y, d_y, sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));
  dbg(y);
  std::printf("reduceSumOnGPU_V6 (提高线程利用率) cost time: %f ms\n",
              total_time / repeats);
}
