file(GLOB cu_files "*.cu")

add_library(hello_world_kernel STATIC ${cu_files})
