set(DEBUG_MACRO_PREFIX_DIR ${THIRD_PARTY_PATH}/dbg_macro)
set(DEBUG_MACRO_INSTALL_DIR ${THIRD_PARTY_PATH}/dbg_macro)
set(DEBUG_MACRO_SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/dbg_macro)

include_directories(${DEBUG_MACRO_INSTALL_DIR}/include)

add_definitions(-DDBG_MACRO_NO_WARNING)

include(ExternalProject)
ExternalProject_Add(
  extern_dbg_macro
  SOURCE_DIR ${DEBUG_MACRO_SOURCE_DIR}
  PREFIX ${DEBUG_MACRO_PREFIX_DIR}
  UPDATE_COMMAND ""
  CMAKE_ARGS -DDBG_MACRO_ENABLE_TESTS=OFF
             -DDBG_MACRO_NO_WARNING=OFF
             -DCMAKE_INSTALL_PREFIX=${DEBUG_MACRO_INSTALL_DIR}
  CMAKE_CACHE_ARGS -DDBG_MACRO_ENABLE_TESTS=OFF
                   -DDBG_MACRO_NO_WARNING=OFF
                   -DCMAKE_INSTALL_PREFIX=${DEBUG_MACRO_INSTALL_DIR}
)

add_library(dbg_macro INTERFACE)
add_dependencies(dbg_macro extern_dbg_macro)
