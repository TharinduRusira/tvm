"""Configuration about tests"""
from __future__ import absolute_import as _abs

import os
import tvm

def ctx_list():
    """Get context list for testcases"""
    device_list = os.environ.get("NNVM_TEST_TARGETS", "")
    device_list = (device_list.split(",") if device_list
                   else ["llvm", "cuda"])
    device_list = set(device_list)
<<<<<<< HEAD
    res = [("llvm", tvm.cpu(0)), ("cuda", tvm.gpu(0))]
    return [x for x in res if x[1].exist and x[0] in device_list]
=======
    res = [(device, tvm.context(device, 0)) for device in device_list]
    return [x for x in res if x[1].exist]
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
