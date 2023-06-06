set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")

file(GLOB cu_files "*.cu")

add_library(reduce_kernel STATIC ${cu_files})


include_directories(${PROJECT_SOURCE_DIR})

include_directories(${CUDA_INCLUDE_DIRS})
