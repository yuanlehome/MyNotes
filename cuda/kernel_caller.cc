#include <iostream>

#include "common.h"
#include "kernel_caller_declare.h"

int main(void) {
  // print_header("from print_hello_world", std::cout);
  // print_hello_world();

  print_header("from add_array", std::cout);
  add_array();
  return 0;
}
