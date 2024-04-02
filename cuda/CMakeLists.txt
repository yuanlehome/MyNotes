cmake_minimum_required(VERSION 3.18)
project(liuyuanle CXX CUDA)

include_directories(${PROJECT_SOURCE_DIR})

set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")
include_directories("${CUDA_INCLUDE_DIRS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++17")
message(STATUS "CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}")

# setting nvcc arch flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_80")
message(STATUS "CMAKE_CUDA_FLAGS:${CMAKE_CUDA_FLAGS}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
MESSAGE(STATUS "CMAKE_EXPORT_COMPILE_COMMANDS: ${CMAKE_EXPORT_COMPILE_COMMANDS}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(third_party)

add_subdirectory(kernels)

add_executable(kernel_caller kernel_caller.cc)

set(KERNEL_LIB trivial_kernel elementwise_kernel transpose_kernel reduce_kernel)

target_link_libraries(kernel_caller PRIVATE ${KERNEL_LIB})
