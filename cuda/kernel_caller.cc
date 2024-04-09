#include <iostream>

#include "common.h"
#include "kernel_caller_declare.h"

int main(void) {
  utils::printHeader("from printHelloWorld", std::cout);
  printHelloWorld();

  utils::printHeader("from addArray", std::cout);
  addArray();

  utils::printHeader("from deviceQuery", std::cout);
  deviceQuery();

  utils::printHeader("from transposeMatrix", std::cout);
  transposeMatrix();

  utils::printHeader("from reduceSum", std::cout);
  reduceSum();

  utils::printHeader("from matrixMultiply", std::cout);
  matrixMultiply();

  return 0;
}
