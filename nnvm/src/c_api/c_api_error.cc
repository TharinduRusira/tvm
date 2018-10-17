/*!
 *  Copyright (c) 2016 by Contributors
 * \file c_api_error.cc
 * \brief C error handling
 */
#include <dmlc/thread_local.h>
<<<<<<< HEAD
#include "./c_api_common.h"
=======
#include "c_api_common.h"
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

struct ErrorEntry {
  std::string last_error;
};

typedef dmlc::ThreadLocalStore<ErrorEntry> NNAPIErrorStore;

const char *NNGetLastError() {
  return NNAPIErrorStore::Get()->last_error.c_str();
}

void NNAPISetLastError(const char* msg) {
  NNAPIErrorStore::Get()->last_error = msg;
}
