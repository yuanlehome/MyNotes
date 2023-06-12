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

  printHeader("from transposeMatrix", std::cout);
  transposeMatrix();

  return 0;
}
