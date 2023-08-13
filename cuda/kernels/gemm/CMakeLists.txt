file(GLOB cu_files "*.cu")

add_library(gemm_kernel STATIC ${cu_files})

add_dependencies(gemm_kernel dbg_macro)
