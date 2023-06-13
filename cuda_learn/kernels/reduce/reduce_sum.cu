#include "dbg.h"

#include "common.h"
#include "kernel_caller_declare.h"

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

// Add x[start...end] 左闭右闭
DATA_TYPE reduceSumOnCPU_V3(const DATA_TYPE* x, int start, int end) {
  MallocWrapper cpu_allocator;
  DATA_TYPE* x_copy =
      (DATA_TYPE*)cpu_allocator.allocate(sizeof(DATA_TYPE) * (end - start) + 1);
  std::copy(x, x + end + 1, x_copy);
  while (start < end) {
    // 双指针
    int i = start, j = end;
    while (i < j) {
      x_copy[i++] += x_copy[j--];
    }
    end = i == j ? i : i - 1;
  }
  return x_copy[start];
}

void reduceSum() {
  constexpr uint32_t N = 1e8;
  constexpr uint32_t M = sizeof(DATA_TYPE) * N;

  MallocWrapper cpu_allocator;
  DATA_TYPE* h_x = (DATA_TYPE*)cpu_allocator.allocate(M);

  std::fill_n(h_x, N, a);

  Timer cpu_timer;
  float total_time = 0.0;
  DATA_TYPE sum_on_cpu;
  for (size_t i = 0; i < repeats; i++) {
    cpu_timer.start();
    sum_on_cpu = reduceSumOnCPU_V1(h_x, 0, N - 1);
    cpu_timer.stop();
    total_time += cpu_timer.elapsedTime();
  }
  dbg(sum_on_cpu);
  std::printf("reduceSumOnCPU_V1 cost time: %f ms\n", total_time / repeats);

  total_time = 0.0;
  for (size_t i = 0; i < repeats; i++) {
    cpu_timer.start();
    sum_on_cpu = reduceSumOnCPU_V2(h_x, 0, N - 1);
    cpu_timer.stop();
    total_time += cpu_timer.elapsedTime();
  }
  dbg(sum_on_cpu);
  std::printf("reduceSumOnCPU_V2 cost time: %f ms\n", total_time / repeats);

  total_time = 0.0;
  for (size_t i = 0; i < repeats; i++) {
    cpu_timer.start();
    sum_on_cpu = reduceSumOnCPU_V3(h_x, 0, N - 1);
    cpu_timer.stop();
    total_time += cpu_timer.elapsedTime();
  }
  dbg(sum_on_cpu);
  std::printf("reduceSumOnCPU_V3 cost time: %f ms\n", total_time / repeats);
}
