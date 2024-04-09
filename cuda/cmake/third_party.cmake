set(THIRD_PARTY_PATH "${CMAKE_CURRENT_BINARY_DIR}/third_party")

# dbg-macro
option(WITH_DEBUG_MACRO "use dbg_macro" OFF)
include(dbg_macro)
if(NOT WITH_DEBUG_MACRO)
  add_definitions(-DDBG_MACRO_DISABLE)
endif()

# cutlass
option(WITH_CUTLASS "use cutlass" OFF)
if(WITH_CUTLASS)
  include(cutlass)
endif()
