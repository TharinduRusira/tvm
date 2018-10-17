# Makefile Example to deploy TVM modules.
ROCM_PATH=/opt/rocm

TVM_ROOT=$(shell cd ../..; pwd)
NNVM_PATH=nnvm
<<<<<<< HEAD
DMLC_CORE=${TVM_ROOT}/dmlc-core
=======
DMLC_CORE=${TVM_ROOT}/3rdparty/dmlc-core
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

PKG_CFLAGS = -std=c++11 -O2 -fPIC\
	-I${TVM_ROOT}/include\
	-I${DMLC_CORE}/include\
<<<<<<< HEAD
	-I${TVM_ROOT}/dlpack/include\
=======
	-I${TVM_ROOT}/3rdparty/dlpack/include\
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
	-I${ROCM_PATH}/include

PKG_LDFLAGS = -L${ROCM_PATH}/lib -L${TVM_ROOT}/lib -ldl -lpthread -lhip_hcc -lMIOpen

.PHONY: clean all

all: lib/libtvm_runtime_rocm.so

# Build rule for all in one TVM package library
lib/libtvm_runtime_rocm.so: rocm_runtime_pack.cc
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -shared -o $@ $(filter %.cc %.o %.a, $^) $(PKG_LDFLAGS)
