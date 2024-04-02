set(CUTLASS_PREFIX_DIR ${THIRD_PARTY_PATH}/cutlass)
set(CUTLASS_INCLUDE_DIR ${CUTLASS_SOURCE_DIR}/include)
set(CUTLASS_SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/cutlass)

include_directories(${CUTLASS_INCLUDE_DIR})

set(CUTLASS_TAG v3.4.1)

include(ExternalProject)
ExternalProject_Add(
  extern_cutlass
  SOURCE_DIR ${CUTLASS_SOURCE_DIR}
  PREFIX ${CUTLASS_PREFIX_DIR}
  COMMAND cd ${CUTLASS_SOURCE_DIR} && git checkout ${CUTLASS_TAG}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND "")

add_library(cutlass INTERFACE)
add_dependencies(cutlass extern_cutlass)
