#pragma once

#include <chrono>
#include <cmath>
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

constexpr size_t repeats = 10;

constexpr DATA_TYPE EPSILON = 1.0e-8;

constexpr int BLOCK_SIZE = 128;

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

static void printHeader(const std::string& header, std::ostream& os) {
  unsigned padding = (80 - header.size()) / 2;
  os << "===" << std::string(73, '-') << "===\n";
  os << std::string(padding, ' ') << header << "\n";
  os << "===" << std::string(73, '-') << "===\n";
}

//===----------------------------------------------------------------------===//
// MallocWrapper
//===----------------------------------------------------------------------===//
class MallocWrapper {
 public:
  MallocWrapper() = default;

  void* allocate(size_t size) {
    void* ptr = std::malloc(size);
    ptrs_.push_back(ptr);
    return ptr;
  }

  ~MallocWrapper() {
    for (void* ptr : ptrs_) {
      if (ptr) std::free(ptr);
    }
  }

 private:
  std::vector<void*> ptrs_;
};

//===----------------------------------------------------------------------===//
// GPUMallocWrapper
//===----------------------------------------------------------------------===//
class GPUMallocWrapper {
 public:
  GPUMallocWrapper() = default;

  void* allocate(size_t size) {
    void* ptr{nullptr};
    CHECK(cudaMalloc(&ptr, size));
    ptrs_.push_back(ptr);
    return ptr;
  }

  ~GPUMallocWrapper() {
    for (void* ptr : ptrs_) {
      if (ptr) CHECK(cudaFree(ptr));
    }
  }

 private:
  std::vector<void*> ptrs_;
};

//===----------------------------------------------------------------------===//
// Timer
//===----------------------------------------------------------------------===//
class Timer {
 public:
  Timer() = default;

  ~Timer() = default;

  void start() { start_time_ = std::chrono::steady_clock::now(); }

  void stop() {
    elapsed_time_ = std::chrono::steady_clock::now() - start_time_;
    total_time_ += elapsed_time_;
  }

  float elapsedTime() const {
    return 1000.0 * elapsed_time_.count();  // ms
  }

  float totalTime() const { return 1000.0 * total_time_.count(); }

 private:
  std::chrono::duration<float> total_time_{0};

  std::chrono::time_point<std::chrono::steady_clock> start_time_;

  std::chrono::duration<float> elapsed_time_{0};
};

//===----------------------------------------------------------------------===//
// GPUTimer
//===----------------------------------------------------------------------===//
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

  float elapsedTime() {
    float elapsed_time;
    CHECK(cudaEventElapsedTime(&elapsed_time, start_, stop_));  // ms
    total_time_ += elapsed_time;
    return elapsed_time;
  }

  float totalTime() const { return total_time_; }

 private:
  float total_time_{0.0};

  cudaEvent_t start_;

  cudaEvent_t stop_;
};

// Check x[i] == y
static bool checkEqual(const DATA_TYPE* x,
                       const uint32_t N,
                       const DATA_TYPE y) {
  bool has_error = false;
  for (size_t i = 0; i < N; i++) {
    if (fabs(x[i] - y) > EPSILON) {
      has_error = true;
      break;
    }
  }
  return !has_error;
}

// Check x[i] == y[i]
static bool checkEqual(const DATA_TYPE* x,
                       const DATA_TYPE* y,
                       const uint32_t N) {
  bool has_error = false;
  for (size_t i = 0; i < N; i++) {
    if (fabs(x[i] - y[i]) > EPSILON) {
      has_error = true;
      break;
    }
  }
  return !has_error;
}
