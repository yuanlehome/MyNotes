file(GLOB cu_files "*.cu")

add_library(elementwise_kernel STATIC ${cu_files})

add_dependencies(elementwise_kernel dbg_macro)
