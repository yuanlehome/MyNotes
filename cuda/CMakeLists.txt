cmake_minimum_required(VERSION 3.20)

project(cuda_learn CXX CUDA)

include_directories(${PROJECT_SOURCE_DIR})

option(WITH_DOUBLE "use double or float precision" OFF)
if(WITH_DOUBLE)
  message(STATUS "WITH_DOUBLE: ${WITH_DOUBLE}")
  add_definitions(-DWITH_DOUBLE_PRECISION)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
# target_compile_features(hello_world PUBLIC cxx_std_14)

# setting nvcc arch flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75")
message(STATUS "CMAKE_CUDA_FLAGS:${CMAKE_CUDA_FLAGS}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
MESSAGE(STATUS "CMAKE_EXPORT_COMPILE_COMMANDS: ${CMAKE_EXPORT_COMPILE_COMMANDS}")

add_subdirectory(hello_world)
add_subdirectory(add)

add_executable(kernel_caller kernel_caller.cc)

set(kernels hello_world_kernel add_kernel)

target_link_libraries(kernel_caller ${kernels})
