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

  printHeader("from copyMatrix", std::cout);
  copyMatrix();

  printHeader("from transposeSquareMatrix", std::cout);
  transposeSquareMatrix();

  return 0;
}
