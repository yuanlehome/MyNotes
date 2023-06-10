file(GLOB cu_files "*.cu")

add_library(element_wise_kernel STATIC ${cu_files})
