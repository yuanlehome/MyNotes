file(GLOB cu_files "*.cu")

add_library(add_kernel STATIC ${cu_files})
