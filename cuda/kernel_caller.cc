#include <iostream>

#include "common.h"
#include "kernel_caller_declare.h"

int main(void) {
  printHeader("from printHelloWorld", std::cout);
  printHelloWorld();

  printHeader("from addArray", std::cout);
  addArray();

  printHeader("from deviceQuery", std::cout);
  deviceQuery();

  printHeader("from transposeSquareMatrix", std::cout);
  transposeSquareMatrix();

  printHeader("from reduceSum", std::cout);
  reduceSum();

  return 0;
}
