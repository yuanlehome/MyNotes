#pragma once

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ostream>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "dbg.h"

using DATA_TYPE = float;

constexpr size_t repeats = 50;

constexpr DATA_TYPE EPSILON = 1.0e-8;

#define CUDA_CHECK(call)                           \
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

//===----------------------------------------------------------------------===//
// CPUMallocWrapper
//===----------------------------------------------------------------------===//
class CPUMallocWrapper {
 public:
  CPUMallocWrapper() = default;

  void* allocate(size_t size) {
    ptrs_.push_back(std::malloc(size));
    return ptrs_.back();
  }

  ~CPUMallocWrapper() {
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
    CUDA_CHECK(cudaMalloc(&ptr, size));
    ptrs_.push_back(ptr);
    return ptr;
  }

  ~GPUMallocWrapper() {
    for (void* ptr : ptrs_) {
      if (ptr) CUDA_CHECK(cudaFree(ptr));
    }
  }

 private:
  std::vector<void*> ptrs_;
};

//===----------------------------------------------------------------------===//
// CPUTimer
//===----------------------------------------------------------------------===//
class CPUTimer {
 public:
  CPUTimer() = default;

  ~CPUTimer() = default;

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
  std::chrono::duration<double> total_time_{0};

  std::chrono::time_point<std::chrono::steady_clock> start_time_;

  std::chrono::duration<double> elapsed_time_{0};
};

//===----------------------------------------------------------------------===//
// GPUTimer
//===----------------------------------------------------------------------===//
class GPUTimer {
 public:
  GPUTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }

  ~GPUTimer() {
    CUDA_CHECK(cudaEventDestroy(start_));
    CUDA_CHECK(cudaEventDestroy(stop_));
  }

  void start() {
    CUDA_CHECK(cudaEventRecord(start_));
    cudaEventQuery(start_);
  }

  void stop() {
    CUDA_CHECK(cudaEventRecord(stop_));
    CUDA_CHECK(cudaEventSynchronize(stop_));
  }

  float elapsedTime() {
    float elapsed_time{0.0};
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start_, stop_));  // ms
    total_time_ += elapsed_time;
    return elapsed_time;
  }

  float totalTime() const { return total_time_; }

 private:
  float total_time_{0.0};

  cudaEvent_t start_;

  cudaEvent_t stop_;
};

namespace utils {

static void printHeader(const std::string& header, std::ostream& os) {
  unsigned padding = (80 - header.size()) / 2;
  os << "===" << std::string(73, '-') << "===\n";
  os << std::string(padding, ' ') << header << "\n";
  os << "===" << std::string(73, '-') << "===\n";
}

// Check x[i] == y
static bool checkEqual(const DATA_TYPE* x,
                       const uint32_t N,
                       const DATA_TYPE y) {
  bool has_error = false;
  for (size_t i = 0; i < N; i++) {
    if (fabs(x[i] - y) > EPSILON) {
      has_error = true;
      std::cout << "Error[" << i << "/" << N << "]: " << x[i] << " vs " << y
                << std::endl;
      break;
    }
  }
  return !has_error;
}

// Check x[i] == y[i]
static bool checkEqual(const DATA_TYPE* x, const DATA_TYPE* y, const size_t N) {
  bool has_error = false;
  for (size_t i = 0; i < N; i++) {
    if (fabs(x[i] - y[i]) > EPSILON) {
      has_error = true;
      std::cout << "Error[" << i << "/" << N << "]: " << x[i] << " vs " << y[i]
                << std::endl;
      break;
    }
  }
  return !has_error;
}

static void matrixMultiply(
    const float* A, const float* B, float* C, int M, int N, int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < K; ++k) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

}  // namespace utils
