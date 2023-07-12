file(GLOB cu_files "*.cu")

add_library(general_kernel STATIC ${cu_files})
