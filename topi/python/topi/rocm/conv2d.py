<<<<<<< HEAD
# pylint: disable=invalid-name, no-member, too-many-locals, too-many-statements, too-many-arguments, too-many-branches, line-too-long
"""Compute and schedule for rocm conv2d_nchw with auto fusion"""
import tvm
from tvm.contrib import miopen
import topi
from .. import generic
from ..nn.conv2d import conv2d
from ..util import get_const_int


@conv2d.register("rocm")
def conv2d_rocm(data, kernel, stride, padding, layout='NCHW', out_dtype='float32'):
=======
# pylint: disable=invalid-name
"""Compute definition for conv2d with rocm backend"""
import tvm
from tvm import autotvm
from tvm.contrib import miopen

from .. import nn, generic
from ..util import get_const_int, get_const_tuple
from ..cuda.conv2d import conv2d_cuda, schedule_conv2d_nchw_cuda

@autotvm.register_topi_compute(nn.conv2d, 'rocm', ['direct', 'winograd'])
def conv2d_rocm(cfg, data, kernel, strides, padding, layout='NCHW', out_dtype='float32'):
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
    """Conv2D operator for rocm backend.

    Parameters
    ----------
<<<<<<< HEAD
=======
    cfg: ConfigEntity
        The config for this template

>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
    input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    filter : tvm.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

<<<<<<< HEAD
    stride : int or a list/tuple of two ints
=======
    strides : int or a list/tuple of two ints
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    layout : str
        layout of data

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
<<<<<<< HEAD
    assert layout == 'NCHW', "Only NCHW layout is supported."
    assert isinstance(stride, int) or len(stride) == 2
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding
    # handle dilation
    dilation_h = dilation_w = 1
    kernel_tvm = kernel
    kernel_cudnn = kernel
    if isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag:
        kernel_before_dilation = kernel.op.input_tensors[0]
        kernel_cudnn = kernel_before_dilation
        dilation_h = (get_const_int(kernel.shape[2]) + get_const_int(kernel_before_dilation.shape[2]) - 1) \
            // get_const_int(kernel_before_dilation.shape[2])
        dilation_w = (get_const_int(kernel.shape[3]) + get_const_int(kernel_before_dilation.shape[3]) - 1) \
            // get_const_int(kernel_before_dilation.shape[2])
    target = tvm.target.current_target()
    if "miopen" in target.libs:
        return miopen.conv2d_forward(data,
                                     kernel_cudnn,
=======

    target = tvm.target.current_target()
    if "miopen" in target.libs:
        assert layout == 'NCHW', "Only NCHW layout is supported."
        CO, CI, KH, KW = get_const_tuple(kernel.shape)
        N, _, H, W = get_const_tuple(data.shape)

        # handle dilation
        stride_h, stride_w = (strides, strides) if isinstance(strides, int) else strides
        pad_h, pad_w = (padding, padding) if isinstance(padding, int) else padding

        OH = (H + 2 * pad_h - KH) // stride_h + 1
        OW = (W + 2 * pad_w - KW) // stride_w + 1
        cfg.add_flop(2 * N * OH * OW * CO * CI * KH * KW)

        dilation_h = dilation_w = 1
        kernel_before_dilation = kernel
        if isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag:
            kernel_before_dilation = kernel.op.input_tensors[0]
            if layout == 'NCHW':
                dilation_h = (get_const_int(kernel.shape[2]) +
                              get_const_int(kernel_before_dilation.shape[2]) - 1) \
                             // get_const_int(kernel_before_dilation.shape[2])
                dilation_w = (get_const_int(kernel.shape[3]) +
                              get_const_int(kernel_before_dilation.shape[3]) - 1) \
                             // get_const_int(kernel_before_dilation.shape[2])
            elif layout == 'NHWC':
                dilation_h = (get_const_int(kernel.shape[1]) +
                              get_const_int(kernel_before_dilation.shape[1]) - 1) \
                             // get_const_int(kernel_before_dilation.shape[1])
                dilation_w = (get_const_int(kernel.shape[2]) +
                              get_const_int(kernel_before_dilation.shape[2]) - 1) \
                             // get_const_int(kernel_before_dilation.shape[2])

        return miopen.conv2d_forward(data,
                                     kernel_before_dilation,
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
                                     stride_h,
                                     stride_w,
                                     pad_h,
                                     pad_w,
                                     dilation_h,
                                     dilation_w,
                                     conv_mode=0)
<<<<<<< HEAD
    return topi.nn.conv2d_nchw(data, kernel_tvm, stride, padding, out_dtype)


@generic.schedule_conv2d_nchw.register(["rocm"])
def schedule_conv2d_nchw(outs):
    """Schedule for conv2d_nchw with rocm backend.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of conv2d_nchw
=======

    return conv2d_cuda(cfg, data, kernel, strides, padding, layout, out_dtype)


@autotvm.register_topi_schedule(generic.schedule_conv2d_nchw, 'rocm', ["direct", 'winograd'])
def schedule_conv2d_nchw_rocm(cfg, outs):
    """TOPI schedule callback of conv2d for rocm

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    outs: Array of Tensor
        The computation graph description of conv2d
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
<<<<<<< HEAD
        The computation schedule for conv2d_nchw.
    """
    target = tvm.target.current_target()
    if target and "miopen" in target.libs:
        return topi.generic.schedule_extern(outs)
    return topi.cuda.schedule_conv2d_nchw(outs)
=======
        The computation schedule for conv2d.
    """
    target = tvm.target.current_target()
    if target and "miopen" in target.libs:
        return generic.schedule_extern(outs)

    return schedule_conv2d_nchw_cuda(cfg, outs)
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
