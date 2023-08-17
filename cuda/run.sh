#!/bin/bash

# clean build directory
mkdir -p build
cd build
rm -rf *

# compile target
cmake .. -G Ninja -DWITH_DOUBLE=OFF -DWITH_DEBUG=ON
ninja -j8

# cuda-memcheck ./kernel_caller
# nvprof --profile-child-processes ./kernel_caller

./kernel_caller

# nvprof --metrics shared_load_transactions_per_request,shared_store_transactions_per_request ./kernel_caller
