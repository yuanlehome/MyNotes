file(GLOB cu_files "*.cu")

add_library(trivial_kernel STATIC ${cu_files})

add_dependencies(trivial_kernel dbg_macro)