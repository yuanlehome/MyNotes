file(GLOB cu_files "*.cu")

add_library(reduce_kernel STATIC ${cu_files})

add_dependencies(reduce_kernel dbg_macro)
