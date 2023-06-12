file(GLOB cu_files "*.cu")

add_library(transpose_kernel STATIC ${cu_files})
