<<<<<<< HEAD
# Makefile for example to deploy TVM modules in SGX.

TVM_ROOT := $(shell cd ../..; pwd)
NNVM_PATH := nnvm
DMLC_CORE := ${TVM_ROOT}/dmlc-core

SGX_SDK ?= /opt/sgxsdk
SGX_MODE ?= SIM
SGX_ARCH ?= x64
SGX_DEBUG ?= 1
=======
SGX_SDK ?= /opt/sgxsdk
RUST_SGX_SDK ?= /opt/rust-sgx-sdk
SGX_MODE ?= SIM
DEBUG ?= true
NUM_THREADS ?= 4

TVM_DIR ?= $(shell git rev-parse --show-toplevel)

export
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

sgx_edger8r := $(SGX_SDK)/bin/x64/sgx_edger8r
sgx_enclave_signer := $(SGX_SDK)/bin/x64/sgx_sign

ifneq ($(SGX_MODE), HW)
	sgx_sim := _sim
endif
urts_library_name := sgx_urts$(sgx_sim)
trts_library_name := sgx_trts$(sgx_sim)
tservice_library_name := sgx_tservice$(sgx_sim)
uservice_library_name := sgx_uae_service$(sgx_sim)

<<<<<<< HEAD
pkg_cflags := -std=c++11 -O2 -fPIC\
	-I${TVM_ROOT}/include\
	-I${DMLC_CORE}/include\
	-I${TVM_ROOT}/dlpack/include\
	-I.\
	-DDMLC_LOG_STACK_TRACE=0\
	-fmax-errors=4

pkg_ldflags := -L${TVM_ROOT}/lib

enclave_include_paths := -I$(SGX_SDK)/include\
	-I$(SGX_SDK)/include/tlibc\
	-I$(SGX_SDK)/include/libcxx\
	-I$(SGX_SDK)/include/stdc++\

enclave_cflags := -static -nostdinc\
	-fvisibility=hidden -fpie -fstack-protector-strong\
	-ffunction-sections -fdata-sections\
	-DDMLC_CXX11_THREAD_LOCAL=0\
	-include "lib/tvm_t.h"\
	$(enclave_include_paths)\

enclave_cxxflags := -nostdinc++ $(enclave_cflags) -DTVM_SGX_MAX_CONCURRENCY=4

enclave_ldflags :=\
=======
pkg_cflags := -std=c++11 -fPIC \
	-I$(SGX_SDK)/include \
	-I$(TVM_DIR)/include \
	-I$(TVM_DIR)/dlpack/include \
	-I$(TVM_DIR)/dmlc-core/include

pkg_ldflags := -L$(TVM_DIR)/build -ltvm_runtime

ifneq ($(DEBUG), false)
	debug := debug
	enclave_cflags += -Og -g
	pkg_cflags += -Og -g
else
	debug := release
	enclave_cflags += -O2
	pkg_cflags += -O2
endif

build_dir := build

enclave_cflags := \
	-I$(SGX_SDK)/include \
	-I$(SGX_SDK)/include/tlibc \
	-I$(SGX_SDK)/include/stdport \
	-I$(SGX_SDK)/include/epid \
	-I$(TVM_DIR)/include \
	-I$(TVM_DIR)/dlpack/include \
	-I$(TVM_DIR)/dmlc-core/include

enclave_ldflags :=\
	-L$(build_dir) -L$(TVM_DIR)/build \
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
	-Wl,--no-undefined -nostdlib -nodefaultlibs -nostartfiles -L$(SGX_SDK)/lib64\
	-Wl,--whole-archive -l$(trts_library_name) -Wl,--no-whole-archive\
	-Wl,--start-group\
	-lsgx_tstdc -lsgx_tstdcxx -lsgx_tcxx -lsgx_tcrypto -lsgx_tkey_exchange -l$(tservice_library_name)\
<<<<<<< HEAD
	-Wl,--end-group\
	-Wl,-Bstatic -Wl,-Bsymbolic -Wl,--no-undefined\
	-Wl,-pie,-eenclave_entry -Wl,--export-dynamic\
	-Wl,--defsym,__ImageBase=0 -Wl,--gc-sections

.PHONY: clean all

all: lib/test_addone.signed.so

# The code library built by TVM
lib/test_addone_sys.o: prepare_test_libs.py
	python prepare_test_libs.py

lib/tvm_t.h: ../../src/runtime/sgx/tvm.edl
	$(sgx_edger8r) --trusted $< --trusted-dir lib --search-path $(SGX_SDK)/include
	mv $@ $@.in
	awk 'NR==4{print "#include <tvm/runtime/c_runtime_api.h>"}1' $@.in > $@

lib/tvm_t.c: lib/tvm_t.h

lib/tvm_t.o: lib/tvm_t.c
	$(CC) $(enclave_cflags) $(pkg_cflags) -c $< -o $@ -include $(TVM_ROOT)/include/tvm/runtime/c_runtime_api.h

# The enclave library
lib/test_addone.so: $(TVM_ROOT)/src/runtime/sgx/trusted/runtime.cc lib/tvm_t.o lib/test_addone_sys.o
	$(CXX) $^ -o $@ $(pkg_cflags) $(pkg_ldflags) $(enclave_cxxflags) $(enclave_ldflags) -g

# The signed enclave
lib/test_addone.signed.so: lib/test_addone.so enclave_config.xml
	$(sgx_enclave_signer) sign -key lib/enclave.pem -enclave $< -out $@ -config enclave_config.xml

clean:
	rm -rf lib
=======
	-lenclave -ltvm_t\
	-Wl,--end-group\
	-Wl,-Bstatic -Wl,-Bsymbolic -Wl,--no-undefined\
	-Wl,-pie,-eenclave_entry -Wl,--export-dynamic\
	-Wl,--defsym,__ImageBase=0 -Wl,--gc-sections\
	-Wl,--version-script=enclave/enclave.lds

.PHONY: enclave clean

enclave: $(build_dir)/enclave.signed.so

$(build_dir)/enclave.signed.so: $(build_dir)/enclave.so build/enclave_config.xml enclave/enclave.pem
	$(sgx_enclave_signer) sign -key enclave/enclave.pem -enclave $< -out $@ -config build/enclave_config.xml

enclave/enclave.pem:
	curl -sSo $@ 'https://gist.githubusercontent.com/nhynes/8a2d80068a92e672f8b0b7d710ceb404/raw/2d5ae5fbe83198ede49465fdc6535065e093543b/tvm_sgx_demo.pem'

build/enclave_config.xml: enclave/enclave_config.xml.in
	cpp $^ -P -o $@ -DNUM_THREADS=$$(( $(NUM_THREADS) + 1 ))

$(build_dir)/enclave.so: $(build_dir)/libenclave.a $(TVM_DIR)/build/libtvm_t.a
	$(CXX) $< -o $@ $(enclave_ldflags) $(enclave_cflags) -ltvm_t

$(build_dir)/libenclave.a: enclave/target/x86_64-unknown-linux-sgx/$(debug)/libmodel_enclave.a
	@mkdir -p $(@D)
	@cp $< $@

enclave/target/x86_64-unknown-linux-sgx/$(debug)/libmodel_enclave.a: enclave/**/*
	$(MAKE) -C enclave

clean:
	$(MAKE) -s -C enclave clean
	rm -rf build
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
