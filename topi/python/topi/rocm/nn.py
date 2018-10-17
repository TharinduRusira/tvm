"""scheduler for normalization functions on rocm backend"""
from __future__ import absolute_import as _abs

<<<<<<< HEAD
import topi
from .. import generic

@generic.schedule_lrn.register(["rocm", "gpu"])
def schedule_lrn(outs):
    return topi.cuda.schedule_lrn(outs)

@generic.schedule_l2norm.register(["rocm", "gpu"])
def schedule_l2norm(outs):
    return topi.cuda.schedule_l2norm(outs)
=======
import tvm
from .. import generic
from .. import cpp

@generic.schedule_lrn.register(["rocm", "gpu"])
def schedule_lrn(outs):
    target = tvm.target.current_target(allow_none=False)
    cpp_target = cpp.TEST_create_target(target.target_name)
    return cpp.rocm.schedule_lrn(cpp_target, outs)

@generic.schedule_l2_normalize.register(["rocm", "gpu"])
def schedule_l2_normalize(outs):
    target = tvm.target.current_target(allow_none=False)
    cpp_target = cpp.TEST_create_target(target.target_name)
    return cpp.rocm.schedule_l2_normalize(cpp_target, outs)
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
