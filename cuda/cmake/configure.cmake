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

option(WITH_DOUBLE "use double or float precision" OFF)
if(WITH_DOUBLE)
  message(STATUS "WITH_DOUBLE: ${WITH_DOUBLE}")
  add_definitions(-DWITH_DOUBLE)
endif()
