#pragma once

#ifdef WITH_DOUBLE_PRECISION
using DATA_TYPE = double;
#else
using DATA_TYPE = float;
#endif

void print_hello_world();

void add_array();
