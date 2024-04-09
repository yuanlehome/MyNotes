#include "all.h"

void matrixMultiply() {
  gemm_naive();
  gemm_cutlass();
}
