file(GLOB cu_files "*.cu")

add_library(trivial_kernel STATIC ${cu_files})
