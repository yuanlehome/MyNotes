#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ostream>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#ifdef WITH_DOUBLE
using DATA_TYPE = double;
#else
using DATA_TYPE = float;
#endif

constexpr size_t warm_up = 1;
constexpr size_t repeats = 10;

#define CHECK(call)                                \
  do {                                             \
    const cudaError_t error_code = call;           \
    if (error_code != cudaSuccess) {               \
      std::printf("at %s:%d - %s.\n",              \
                  __FILE__,                        \
                  __LINE__,                        \
                  cudaGetErrorString(error_code)); \
      exit(1);                                     \
    }                                              \
  } while (0)

static void print_header(const std::string& header, std::ostream& os) {
  unsigned padding = (80 - header.size()) / 2;
  os << "===" << std::string(73, '-') << "===\n";
  os << std::string(padding, ' ') << header << "\n";
  os << "===" << std::string(73, '-') << "===\n";
}

class MallocWraper {
 public:
  MallocWraper() = default;

  void* allocate(size_t size) {
    void* ptr = std::malloc(size);
    ptrs_.push_back(ptr);
    return ptr;
  }

  ~MallocWraper() {
    for (void* ptr : ptrs_) {
      if (ptr) std::free(ptr);
    }
  }

 private:
  std::vector<void*> ptrs_;
};

class GPUMallocWraper {
 public:
  GPUMallocWraper() = default;

  void* allocate(size_t size) {
    void* ptr{nullptr};
    CHECK(cudaMalloc(&ptr, size));
    ptrs_.push_back(ptr);
    return ptr;
  }

  ~GPUMallocWraper() {
    for (void* ptr : ptrs_) {
      if (ptr) CHECK(cudaFree(ptr));
    }
  }

 private:
  std::vector<void*> ptrs_;
};

class Timer {
 public:
  Timer() = default;

  ~Timer() = default;

  void start() { start_time_ = std::chrono::steady_clock::now(); }

  void stop() {
    elapsed_time_ = std::chrono::steady_clock::now() - start_time_;
  }

  float elapsed_time() const {
    return 1000.0 * elapsed_time_.count();  // ms
  }

 private:
  std::chrono::time_point<std::chrono::steady_clock> start_time_;

  std::chrono::duration<float> elapsed_time_{0};
};

class GPUTimer {
 public:
  GPUTimer() {
    CHECK(cudaEventCreate(&start_));
    CHECK(cudaEventCreate(&stop_));
  }

  ~GPUTimer() {
    CHECK(cudaEventDestroy(start_));
    CHECK(cudaEventDestroy(stop_));
  }

  void start() {
    CHECK(cudaEventRecord(start_));
    cudaEventQuery(start_);
  }

  void stop() {
    CHECK(cudaEventRecord(stop_));
    CHECK(cudaEventSynchronize(stop_));
  }

  float elapsed_time() const {
    float elapsed_time;
    CHECK(cudaEventElapsedTime(&elapsed_time, start_, stop_));  // ms
    return elapsed_time;
  }

 private:
  cudaEvent_t start_;

  cudaEvent_t stop_;
};
