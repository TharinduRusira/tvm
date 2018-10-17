/*!
 *  Copyright (c) 2017 by Contributors
 * \file intrin_rule_opencl.cc
 * \brief OpenCL intrinsic rules.
 */
<<<<<<< HEAD
#include "./intrin_rule.h"
=======
#include "intrin_rule.h"
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

namespace tvm {
namespace codegen {
namespace intrin {

<<<<<<< HEAD
=======
TVM_REGISTER_GLOBAL("tvm.intrin.rule.opengl.floor")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.opengl.ceil")
.set_body(DispatchExtern<Direct>);

>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
TVM_REGISTER_GLOBAL("tvm.intrin.rule.opengl.exp")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.opengl.log")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.opengl.tanh")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.opengl.sqrt")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.opengl.pow")
.set_body(DispatchExtern<Direct>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.opengl.popcount")
.set_body(DispatchExtern<Direct>);

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
