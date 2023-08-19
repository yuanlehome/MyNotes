file(GLOB cu_files "*.cu")

add_library(softmax_kernel STATIC ${cu_files})

add_dependencies(softmax_kernel dbg_macro)
