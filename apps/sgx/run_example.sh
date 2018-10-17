#!/bin/bash

sgx_sdk=${SGX_SDK:=/opt/sgxsdk}
<<<<<<< HEAD
make
echo "========================="
LD_LIBRARY_PATH="$sgx_sdk/lib64":${LD_LIBRARY_PATH} TVM_CACHE_DIR=/tmp python test_addone.py
=======

export LD_LIBRARY_PATH="$sgx_sdk/lib64":${LD_LIBRARY_PATH}
export CC=clang-6.0
export AR=llvm-ar-6.0
export TVM_CACHE_DIR=/tmp

make && printf "\n" && python3 run_model.py
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
