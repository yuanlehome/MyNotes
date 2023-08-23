file(GLOB cu_files "*.cu")

add_library(scan_kernel STATIC ${cu_files})

add_dependencies(scan_kernel dbg_macro)
