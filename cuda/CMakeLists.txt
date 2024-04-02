cmake_minimum_required(VERSION 3.18)
project(liuyuanle CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
# setting nvcc arch flags
set(CUDA_NVCC_FLAGS -std=c++17 -arch=sm_80)

include_directories(${PROJECT_SOURCE_DIR})

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")
include_directories("${CUDA_INCLUDE_DIRS}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
MESSAGE(STATUS "CMAKE_EXPORT_COMPILE_COMMANDS: ${CMAKE_EXPORT_COMPILE_COMMANDS}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(third_party)

add_subdirectory(kernels)

add_executable(kernel_caller kernel_caller.cc)

set(KERNEL_LIB trivial_kernel elementwise_kernel transpose_kernel reduce_kernel gemm_kernel)

target_link_libraries(kernel_caller PRIVATE ${KERNEL_LIB})
