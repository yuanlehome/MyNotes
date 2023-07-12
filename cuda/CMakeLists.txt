cmake_minimum_required(VERSION 3.18)
project(cuda_learn CXX CUDA)

include_directories(${PROJECT_SOURCE_DIR})

set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")
include_directories("${CUDA_INCLUDE_DIRS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++17")
# target_compile_features(hello_world PUBLIC cxx_std_17)
message(STATUS "CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}")

# setting nvcc arch flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_86")
message(STATUS "CMAKE_CUDA_FLAGS:${CMAKE_CUDA_FLAGS}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
MESSAGE(STATUS "CMAKE_EXPORT_COMPILE_COMMANDS: ${CMAKE_EXPORT_COMPILE_COMMANDS}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(WITH_DOUBLE "use double or float precision" OFF)
if(WITH_DOUBLE)
  message(STATUS "WITH_DOUBLE: ${WITH_DOUBLE}")
  add_definitions(-DWITH_DOUBLE)
endif()

option(WITH_DEBUG "use dbg_macro" OFF)
if(NOT WITH_DEBUG)
  message(STATUS "WITH_DEBUG: ${WITH_DEBUG}")
  add_definitions(-DDBG_MACRO_DISABLE)
endif()

include(third_party)

add_subdirectory(kernels)

add_executable(kernel_caller kernel_caller.cc)

set(KERNEL_LIB general_kernel element_wise_kernel transpose_kernel reduce_kernel)

target_link_libraries(kernel_caller PRIVATE ${KERNEL_LIB})
