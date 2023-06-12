file(GLOB cu_files "*.cu")

add_library(transpose_kernel STATIC ${cu_files})

add_dependencies(transpose_kernel dbg_macro)
