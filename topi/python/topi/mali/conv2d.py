# pylint: disable=invalid-name,unused-variable,unused-argument,no-else-return
"""conv2d schedule on ARM Mali GPU"""
<<<<<<< HEAD

from __future__ import absolute_import as _abs

import numpy as np
import tvm

from .. import generic
from .. import util
from .. import tag
from ..nn import pad
from ..nn.conv2d import conv2d
from ..nn.util import get_pad_tuple

##### SCHEDULE UTILITIES #####
def fuse_and_bind(s, tensor, axis=None, num_thread=None):
    """ fuse all the axis and bind to GPU threads """
    axis = axis or s[tensor].op.axis
    fused = s[tensor].fuse(*axis)
    max_threads = tvm.target.current_target(allow_none=False).max_num_threads
    bx, tx = s[tensor].split(fused, num_thread or max_threads)
    s[tensor].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[tensor].bind(tx, tvm.thread_axis("threadIdx.x"))
    return bx, tx

def tile_and_bind(s, tensor, y, x, y_factor, x_factor=None):
    """ tile and bind to GPU threads """
    x_factor = x_factor or y_factor
    yo, xo, yi, xi = s[tensor].tile(y, x, y_factor, x_factor)
    s[tensor].bind(xo, tvm.thread_axis("blockIdx.x"))
    s[tensor].bind(xi, tvm.thread_axis("threadIdx.x"))
    s[tensor].bind(yo, tvm.thread_axis("blockIdx.y"))
    s[tensor].bind(yi, tvm.thread_axis("threadIdx.y"))
    return yo, xo, yi, xi

def tile_and_bind3d(s, tensor, z, y, x, z_factor=2, y_factor=None, x_factor=None):
    """ tile and bind 3d """
    y_factor = y_factor or z_factor
    x_factor = x_factor or y_factor
    zo, zi = s[tensor].split(z, z_factor)
    yo, yi = s[tensor].split(y, y_factor)
    xo, xi = s[tensor].split(x, x_factor)
    s[tensor].bind(zo, tvm.thread_axis("blockIdx.z"))
    s[tensor].bind(zi, tvm.thread_axis("threadIdx.z"))
    s[tensor].bind(yo, tvm.thread_axis("blockIdx.y"))
    s[tensor].bind(yi, tvm.thread_axis("threadIdx.y"))
    s[tensor].bind(xo, tvm.thread_axis("blockIdx.x"))
    s[tensor].bind(xi, tvm.thread_axis("threadIdx.x"))

def pack_tensor(s, tensor, factor, readers):
    """ do transform X[n, m] -> X[n / factor, m, factor] """
    tmp = s.cache_read(tensor, 'global', readers)
    y, x = s[tmp].op.axis
    yo, yi = s[tmp].split(y, factor)
    s[tmp].reorder(yo, x, yi)
    s[tmp].compute_inline()
    return s.cache_write(tmp, 'global')

def transpose(s, tensor, readers):
    """ do transform X[n, m] -> X[m, n] """
    tmp = s.cache_read(tensor, 'global', readers)
    y, x = s[tmp].op.axis
    s[tmp].reorder(x, y)
    s[tmp].compute_inline()
    return s.cache_write(tmp, "global"), tmp

def const_array(data, name):
    """ convert an const array to tvm tensor"""
    row, col = data.shape
    dtype = str(data.dtype)

    def select_array(i, j):
        now = tvm.const(0.0, dtype)
        for ii in range(row):
            for jj in range(col):
                now = tvm.select(tvm.all(i % row == ii, j % col == jj),
                                 tvm.const(data[ii][jj], dtype),
                                 now)
        return now
    return tvm.compute(data.shape, select_array, name=name)


@conv2d.register(["mali"])
def decl_conv2d(data, kernel, stride, padding, layout='NCHW', out_dtype='float32'):
    """Conv2D operator for ARM Mali GPU backend.

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    kernel : tvm.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    stride : int or a list/tuple of two ints
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
    assert layout == 'NCHW', "only support NCHW convolution on mali"
    assert data.shape[0].value == 1, "only support batch size=1 convolution on mali"
    assert data.dtype == kernel.dtype, "Do not support inputs with different data types now."

    out_dtype = data.dtype
    HPAD, WPAD, _, _ = get_pad_tuple(padding, kernel)
    kernel_shape = util.get_const_tuple(kernel.shape)
    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride

    if (kernel_shape[2:4] == (3, 3) and (HPAD, WPAD) == (1, 1) and kernel_shape[0] >= 64 and
            (HSTR, WSTR) == (1, 1)):
        return _decl_winograd(data, kernel, stride, padding, layout, out_dtype)
    elif kernel_shape[2:4] == (1, 1):
        return _decl_im2col(data, kernel, stride, padding, layout, out_dtype)
    else:
        return _decl_spatialpack(data, kernel, stride, padding, layout, out_dtype)

@generic.schedule_conv2d_nchw.register(["mali"])
def schedule_conv2d_nchw(outs):
    """Schedule for conv2d_nchw for ARM Mali GPU

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of conv2d_nchw
=======
import numpy as np

import tvm
from tvm import autotvm
from tvm.autotvm.task.space import get_factors

from ..generic import schedule_conv2d_nchw, schedule_conv2d_winograd_without_weight_transform
from ..util import traverse_inline, get_const_int, get_const_tuple, const_matrix
from ..nn import conv2d, conv2d_winograd_without_weight_transform, \
    get_pad_tuple, pad, conv2d_alter_layout

# reuse some compute declarations from ARM CPU
from ..arm_cpu.conv2d import _conv_arg_to_workload, _decl_spatial_pack,\
    _winograd_conv_arg_to_workload, _alter_conv2d_layout_arm


@conv2d.register('mali')
@autotvm.task.dispatcher
def conv2d_mali(data, kernel, strides, padding, layout, out_dtype):
    """TOPI compute callback. Mark this function as a dispatcher, so
    this template can assign config according to workload

    Returns
    -------
    workload: Tuple
        Dispatcher will use this workload to query corresponding config.
        Then use cfg.template_key to call a registered template.
    """
    return _conv_arg_to_workload(data, kernel, strides, padding, layout, out_dtype)

@conv2d_mali.register(['direct'])
def decl_spatial_pack(cfg, data, kernel, strides, padding, layout, out_dtype):
    """spatial packing template"""
    return _decl_spatial_pack(cfg, data, kernel, strides, padding, layout, out_dtype, num_tile=3)

@autotvm.register_topi_schedule(schedule_conv2d_nchw, 'mali', ['direct', 'winograd'])
def schedule_conv2d_nchw_mali(cfg, outs):
    """TOPI schedule callback for conv2d

    Parameters
    ----------
    cfg: ConfigEntity
        The configuration of this template
    outs: Array of Tensor
        The computation graph description of convolution2d
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
<<<<<<< HEAD
        The computation schedule for conv2d_nchw.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def traverse(op):
        """inline all one-to-one-mapping operators except the last stage (output)"""
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)

        if 'im2col_conv_output' in op.tag:
            _schedule_im2col_conv2d(s, op)

        if 'spatialpack_conv_output' in op.tag:
            _schedule_spatialpack_conv2d(s, op)

        if 'winograd_conv_output' in op.tag:
            _schedule_winograd(s, op)

    traverse(outs[0].op)
    return s

def _decl_spatialpack(data, kernel, stride, padding, layout, out_dtype):
    """declare the spatialpack method (spatial packing) for conv2d"""
    _, CI, IH, IW = [util.get_const_int(x) for x in data.shape]
    CO, _, KH, KW = [util.get_const_int(x) for x in kernel.shape]
    HPAD, WPAD, _, _ = get_pad_tuple(padding, kernel)
    HCAT, WCAT = KH - 1, KW - 1

    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride

    N = 1
    TH = IH + 2*HPAD
    TW = IW + 2*WPAD
    OH = (IH + 2*HPAD - KH) // HSTR + 1
    OW = (IW + 2*WPAD - KW) // WSTR + 1

    DO_PAD = (HPAD != 0 and WPAD != 0)
    if DO_PAD:
        data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")
    else:
        data_pad = data

    # set tunable parameters (tile factor, ...)
    tune_config = getattr(tvm.target.current_target(), "tune_config", None)
    if tune_config is None:
        VH = 1
        VW, VC = 4, 4
        # correct tile factor
        if OW % VW != 0:
            if OW == 14:
                VW = 2
                VC = 8
            elif OW == 7:
                VW = 7
    else:
        VH = tune_config['VH']
        VW = tune_config['VW']
        VC = tune_config['VC']

    if data.dtype == 'float16':
        VC *= 2

    assert CO % VC == 0
    assert OH % VH == 0, "OH: %d  VH : %d" % (OH, VH)
    assert OW % VW == 0, "OW: %d  VW : %d" % (OW, VW)

    dvshape = (N, TH//(VH*HSTR), TW//(VW*WSTR), CI, VH*HSTR+HCAT, VW*WSTR+WCAT)
    kvshape = (CO // VC, CI, KH, KW, VC)
    ovshape = (N, CO // VC, OH // VH, OW // VW, VH, VW, VC)
    oshape = (N, CO, OH, OW)

    data_vec = tvm.compute(dvshape, lambda n, h, w, ci, vh, vw:
                           data_pad[n][ci][h*VH*HSTR+vh][w*VW*WSTR+vw],
                           name='data_vec')

    kernel_vec = tvm.compute(kvshape, lambda co, ci, kh, kw, vc:
                             kernel[co*VC+vc][ci][kh][kw],
                             name='kernel_vec')

    ci = tvm.reduce_axis((0, CI), name='ci')
    kh = tvm.reduce_axis((0, KH), name='kh')
    kw = tvm.reduce_axis((0, KW), name='kw')

    conv = tvm.compute(ovshape, lambda n, co, h, w, vh, vw, vc:\
                tvm.sum(data_vec[n, h, w, ci, vh*HSTR+kh, vw*WSTR+kw].astype(out_dtype) *
                        kernel_vec[co, ci, kh, kw, vc].astype(out_dtype),
                        axis=[ci, kh, kw]), name='conv')

    output = tvm.compute(oshape, lambda n, co, h, w:
                         conv[n][co//VC][h/VH][w//VW][h%VH][w%VW][co%VC],
                         name='output_unpack', tag='spatialpack_conv_output')

    return output

def _schedule_spatialpack_conv2d(s, op):
    """schedule the spatialpack method (spatial packing) for conv2d"""
    # get ops and tensors
    output = op.output(0)
    output_height = util.get_const_int(output.shape[2])

    conv = op.input_tensors[0]
    data_vec = s[conv].op.input_tensors[0]
    kernel_vec = s[conv].op.input_tensors[1]
    data = s[data_vec].op.input_tensors[0]
    kernel = s[kernel_vec].op.input_tensors[0]

    # set tunable parameters (tile factor, ...)
    tune_config = getattr(tvm.target.current_target(), "tune_config", None)
    if tune_config is None:
        num_thread = 8

        out_channel = util.get_const_int(kernel.shape[0])
        in_channel = util.get_const_int(kernel.shape[1])
        in_width = util.get_const_int(data.shape[2])

        if in_width >= 224:
            pass
        elif in_width >= 112:
            pass
        elif in_width >= 56:
            if out_channel != in_channel:
                num_thread = 16
        elif in_width >= 28:
            if out_channel >= 256:
                num_thread = 16
        elif in_width >= 14:
            if in_channel == out_channel:
                num_thread = 8
            else:
                num_thread = 4
    else:
        num_thread = tune_config["num_thread"]

    last = 1
    if output_height == 28:
        last = 7
        num_thread = 32

    if data.dtype == 'float16' and (util.get_const_int(conv.shape[1]) == 4 or output_height == 28):
        num_thread //= 2

    # schedule dilation
    if isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag:
        s[kernel].compute_inline()
=======
        The computation schedule for conv2d
    """
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        # schedule conv2d
        if 'spatial_conv2d_output' in op.tag:
            output = op.output(0)
            conv = op.input_tensors[0]

            data_vec = conv.op.input_tensors[0]
            data_pad = data_vec.op.input_tensors[0]
            s[data_pad].compute_inline()

            kernel_vec = conv.op.input_tensors[1]
            if kernel_vec.op.name == 'kernel_vec':
                kernel = kernel_vec.op.input_tensors[0]
            else:
                kernel = kernel_vec
            if isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag:
                s[kernel].compute_inline()

            _schedule_spatial_pack(cfg, s, output, conv, data_vec, kernel_vec)

        if 'winograd_conv2d_output' in op.tag:
            _schedule_winograd(cfg, s, op)

    traverse_inline(s, outs[0].op, _callback)
    return s


def _schedule_spatial_pack(cfg, s, output, conv, data_vec, kernel_vec):
    """schedule the spatial packing for conv2d"""
    data = s[data_vec].op.input_tensors[0]

    max_unroll = 16
    vec_size = [1, 2, 4, 8, 16]
    # get tunable parameters (they are defined in compute)
    BC, TC, VC = cfg["tile_co"].size
    BH, TH, VH = cfg["tile_oh"].size
    BW, TW, VW = cfg["tile_ow"].size
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

    # schedule padding
    if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
        data_pad = data
<<<<<<< HEAD
        data = data_pad.op.input_tensors[0]
=======
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
        s[data_pad].compute_inline()

    # schedule data packing
    _, h, w, ci, vh, vw = s[data_vec].op.axis
    tile_and_bind3d(s, data_vec, h, w, ci, 1)
<<<<<<< HEAD
    s[data_vec].unroll(vw)

    # schedule kernel packing
    co, ci, kh, kw, vc = s[kernel_vec].op.axis
    tile_and_bind(s, kernel_vec, co, ci, 1)
    s[kernel_vec].unroll(kh)
    s[kernel_vec].unroll(kw)
    s[kernel_vec].vectorize(vc)

    # schedule convolution
    _, c, h, w, vh, vw, vc = s[conv].op.axis
    kc, kh, kw = s[conv].op.reduce_axis
    s[conv].reorder(_, c, h, w, vh, kc, kh, kw, vw, vc)
    tile_and_bind3d(s, conv, c, h, w, num_thread, 1, last)
    s[conv].unroll(kh)
    s[conv].unroll(kw)
    s[conv].unroll(vw)
    s[conv].vectorize(vc)
=======
    if vh.dom.extent.value < max_unroll:
        s[data_vec].unroll(vh)
    if vw.dom.extent.value < max_unroll:
        s[data_vec].unroll(vw)

    if isinstance(kernel_vec.op, tvm.tensor.ComputeOp) and kernel_vec.name == 'kernel_vec':
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # kernel packing will be pre-computed during compilation, so we skip
            # this part to make tuning records correct
            s[kernel_vec].pragma(s[kernel_vec].op.axis[0], 'debug_skip_region')
        else:
            max_threads = tvm.target.current_target(allow_none=False).max_num_threads
            co, ci, kh, kw, vc = s[kernel_vec].op.axis
            fused = s[kernel_vec].fuse(co, ci, kh, kw, vc)
            fused, vec = s[kernel_vec].split(fused, VC)
            bb, tt = s[kernel_vec].split(fused, max_threads)
            s[kernel_vec].bind(bb, tvm.thread_axis("blockIdx.x"))
            s[kernel_vec].bind(tt, tvm.thread_axis("threadIdx.x"))
            if VC in vec_size:
                s[kernel_vec].vectorize(vec)

    # schedule convolution
    n, c, h, w, vh, vw, vc = s[conv].op.axis
    kc, kh, kw = s[conv].op.reduce_axis

    cfg["reorder_0"].apply(s, conv, [n, c, h, w, kc, kh, kw, vh, vw, vc])
    tile_and_bind3d(s, conv, c, h, w, TC, TH, TW)

    cfg["ann_reduce"].apply(s, conv, [kh, kw],
                            axis_lens=[get_const_int(kernel_vec.shape[2]),
                                       get_const_int(kernel_vec.shape[3])],
                            max_unroll=max_unroll)

    cfg["ann_spatial"].apply(s, conv, [vh, vw, vc],
                             axis_lens=[VH, VW, VC],
                             max_unroll=max_unroll,
                             vec_size=vec_size,
                             cfg=cfg)
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

    # schedule output
    if output.op not in s.outputs:  # has bias
        s[output].compute_inline()
        output = s.outputs[0]

    _, co, oh, ow = s[output].op.axis
<<<<<<< HEAD
    tile_and_bind3d(s, output, co, oh, ow, num_thread, 1, last)

def _decl_im2col(data, kernel, stride, padding, layout='NCHW', out_dtype='float32'):
    """declare the Im2Col method for conv2d"""
    _, CI, IH, IW = [x.value for x in data.shape]
    CO, _, KH, KW = [x.value for x in kernel.shape]
    HPAD, WPAD, _, _ = get_pad_tuple(padding, kernel)

    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride

    N = 1
    OH = (IH + 2*HPAD - KH) // HSTR + 1
    OW = (IW + 2*WPAD - KW) // WSTR + 1

    DO_PAD = (HPAD != 0 and WPAD != 0)
    if DO_PAD:
        data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")
    else:
        data_pad = data

    ALIGN = 16
    def upround(x, align):
        return (x + align - 1) // align * align

    # A [CO, CI * KH * KW]
    reduce_len = upround(CI * KH * KW, ALIGN)
    A = tvm.compute((upround(CO, ALIGN), reduce_len), lambda i, j:
                    kernel[i][j // KW // KH][j // KW % KH][j % KW], name='A')

    # B [CI * KH * KW, N * OH * OW]
    B = tvm.compute((reduce_len, upround(N * OH * OW, ALIGN)), lambda i, j:\
            tvm.select(tvm.all(i < CI * KH * KW, j < N * OH * OW),
                       data_pad[j // (OH*OW)][i // (KH*KW)][j // OW % OH*HSTR + i // KW % KH]
                       [j % OW*WSTR + i % KW],
                       tvm.const(0, data_pad.dtype)), name='B')

    gemm_n, gemm_l, gemm_m = A.shape[0], reduce_len, B.shape[1]

    # C [CO, N * OH * OW]
    k = tvm.reduce_axis((0, gemm_l), name='k')
    C = tvm.compute((gemm_n, gemm_m), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k), name='C')

    # output
    # the last term C[gemm_n-1, gemm_m-1] is for enabling the alignment,
    # otherwise the alignment above will be eliminated by bound inference
    output = tvm.compute((N, CO, OH, OW), lambda n, co, h, w:\
                 C[co][n * OW * OW + h * OW + w] + tvm.const(0, C.dtype) * C[gemm_n-1, gemm_m-1],
                         name='output', tag='im2col_conv_output')

    return output

def _schedule_im2col_conv2d(s, op):
    """schedule the Im2Col method for conv2d"""

    # get ops and tensors
    output = op.output(0)
    C = op.input_tensors[0]
    A, B = C.op.input_tensors
    kernel = A.op.input_tensors[0]
    data = B.op.input_tensors[0]

    # tuning parameter config
    tune_config = getattr(tvm.target.current_target(), "tune_config", None)
    if tune_config is None: # use rule
        bn = 4
        unroll_step = 16

        total_work = util.get_const_int(C.shape[0] * C.shape[1])
        reduce_work = util.get_const_int(A.shape[1])
        if total_work > 200000:
            last_work = util.get_const_int(C.shape[1])
            if last_work > 10000:
                num_thread = 16
            elif last_work > 3000:
                num_thread = 8
            elif reduce_work > 100:
                num_thread = 4
            else:
                num_thread = 2

            if reduce_work < 50 and last_work < 30000:
                num_thread = 4
        elif total_work > 150000:
            num_thread = 8
        elif total_work > 50000:
            num_thread = 4
        else:
            num_thread = 2

        if num_thread == 4:
            unroll_step = 2
    else:
        bn = tune_config["bn"]
        num_thread = tune_config["num_thread"]
        unroll_step = tune_config["unroll_step"]

    bna = bnb = bn
    num_thread1 = num_thread2 = num_thread
    if data.dtype == 'float16':
        bnb *= 2
        last_work = util.get_const_int(C.shape[1])
        if last_work % (bnb * num_thread2) != 0:
            num_thread1 = num_thread * 2
            num_thread2 = num_thread // 2

    # schedule dilation
    if isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag:
        s[kernel].compute_inline()

    # schedule padding
    if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
        data_pad = data
        s[data_pad].compute_inline()

    ##### SCHEDULE A #####
    if util.get_const_int(kernel.shape[2]) == 1 and util.get_const_int(kernel.shape[3]) == 1:
        s[A].compute_inline()
    else:
        y, x = s[A].op.axis
        yo, xo, yi, xi = s[A].tile(y, x, bna, util.get_const_int(kernel.shape[3]))
        s[A].vectorize(xi)
        fuse_and_bind(s, A, [yo, xo])

    # pack to vector form
    packedA = pack_tensor(s, A, bna, [C])

    # vectorize load
    y, x = s[packedA].op.axis[:2]
    tmp = s.cache_write(packedA, "local")
    x, xt = s[packedA].split(x, bna)
    _, _, _, xi = tile_and_bind(s, packedA, y, x, num_thread)
    s[tmp].compute_at(s[packedA], xi)
    s[tmp].vectorize(s[tmp].op.axis[1])
    s[tmp].unroll(s[tmp].op.axis[2])
    s[packedA].vectorize(s[packedA].op.axis[2])
    s[packedA].unroll(xt)

    ##### SCHEDULE B #####
    y, x = s[B].op.axis
    yo, xo, yi, xi = s[B].tile(y, x, 1, 1 * bnb)
    fuse_and_bind(s, B, [yo, xo])

    # transpose and pack to vector form
    B_transpose, B_tmp = transpose(s, B, [C])
    s[B_transpose].compute_inline()
    packedB = pack_tensor(s, B_transpose, bnb, [B_tmp])

    # vectorize load
    s[packedB].vectorize(s[packedB].op.axis[2])
    y, x = s[packedB].op.axis[:2]
    tile_and_bind(s, packedB, y, x, num_thread)

    ##### SCHEDULE C #####
    # vectorize and unroll dot
    y, x = s[C].op.axis
    y, x, yt, xt = s[C].tile(y, x, bna, bnb)

    k = s[C].op.reduce_axis[0]
    s[C].reorder(k, yt, xt)
    if unroll_step != 1:
        k, k_unroll = s[C].split(k, unroll_step)
        s[C].unroll(k_unroll)
    s[C].unroll(yt)
    s[C].vectorize(xt)

    tile_and_bind(s, C, y, x, num_thread1, num_thread2)

    ##### COPY TO OUTPUT #####
    if output.op in s.outputs:  # no bias
        output = output
    else:                       # has bias
        s[output].compute_inline()
        output = s.outputs[0]

    n, co, h, w = s[output].op.axis
    h, w, vh, vw = s[output].tile(h, w, 1, bnb)
    s[output].unroll(vh)
    if util.get_const_int(s[output].op.output(0).shape[3]) % bnb != 0:
        pass
    else:
        s[output].vectorize(vw)
    fuse_and_bind(s, output, [n, co, h, w])

def _decl_winograd(data, kernel, stride, padding, layout, out_dtype):
    """declare winograd fast convolution F(2x2, 3x3) for conv2d"""
    N, CI, H, W = [util.get_const_int(x) for x in data.shape]
    CO, CI, KH, KW = [util.get_const_int(x) for x in kernel.shape]
    HPAD, WPAD, _, _ = get_pad_tuple(padding, kernel)
    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride

    assert HSTR == 1 and WSTR == 1 and HPAD == 1 and WPAD == 1 and KH == 3 and KW == 3
    data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")

    B_data = np.array([
        [1, 0, 0, 0],
        [0, 1, -1, 1],
        [-1, 1, 1, 0],
        [0, 0, 0, -1]
    ], out_dtype)

    G_data = np.array([
        [1, 0, 0],
        [1.0/2, 1.0/2, 1.0/2],
        [1.0/2, -1.0/2, 1.0/2],
        [0, 0, 1],
    ], out_dtype)

    A_data = np.array([
        [1, 0],
        [1, 1],
        [1, -1],
        [0, -1],
    ], out_dtype)

    m = 2
    r = 3
    alpha = m + r - 1
    K = CO
    C = CI

    nH, nW = (H + m-1) // m, (W + m-1) // m
    P = N * nH * nW

    bna, bnb = 4, 4
    if data.dtype == 'float16':
        bnb *= 2
    P_round = (P + bnb - 1) // bnb * bnb
    assert K % bna == 0 and P_round % bnb == 0

    # pack input tile
    input_tile = tvm.compute((C, P_round // bnb, alpha, alpha, bnb),
                             lambda c, b, eps, nu, bb:
                             tvm.select(b * bnb + bb < P,\
                             data_pad[(b*bnb+bb) // (nH*nW)][c][(b*bnb+bb) // nW % nH * m + eps]\
                             [(b*bnb+bb) % nW * m + nu], tvm.const(0, data_pad.dtype)),
                             name='d')

    # transform kernel
    G = const_array(G_data, 'G')
    r_kh = tvm.reduce_axis((0, KH), 'r_kh')
    r_kw = tvm.reduce_axis((0, KW), 'r_kw')
    U = tvm.compute((alpha, alpha, K // bna, C, bna), lambda eps, nu, k, c, kk:
                    tvm.sum(kernel[k * bna + kk][c][r_kh][r_kw] * G[eps][r_kh] * G[nu][r_kw],
                            axis=[r_kh, r_kw]), name='U')

    # transform image
    B = const_array(B_data, 'B')
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    V = tvm.compute((alpha, alpha, P_round // bnb, C, bnb), lambda eps, nu, b, c, bb:
                    tvm.sum(input_tile[c][b][r_eps][r_nu][bb] * B[r_eps][eps] * B[r_nu][nu],
                            axis=[r_eps, r_nu]), name='V')

    # batch gemm
    c = tvm.reduce_axis((0, C), name='c')
    M = tvm.compute((alpha, alpha, K, P_round), lambda eps, nu, k, b:
                    tvm.sum(U[eps][nu][k // bna][c][k % bna] *
                            V[eps][nu][b // bnb][c][b % bnb], axis=c), name='M')

    # inverse transform
    A = const_array(A_data, 'A')
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    Y = tvm.compute((K, P, m, m), lambda k, b, vh, vw:
                    tvm.sum(M[r_eps][r_nu][k][b] * A[r_eps][vh] * A[r_nu][vw],
                            axis=[r_eps, r_nu]), name='Y')

    # unpack output
    output = tvm.compute((N, K, H, W), lambda n, k, h, w:
                         Y[k][n * nH * nW + (h//m) * nW + w//m][h % m][w % m]
                         # thw following term is used to make the padding effective,
                         # otherwise the padding will be eliminated by bound inference
                         + tvm.const(0, out_dtype) * M[alpha-1][alpha-1][K-1][P_round-1],
                         name='output', tag='winograd_conv_output')

    return output

def _schedule_winograd(s, op):
    """schedule winograd fast convolution F(2x2, 3x3) for conv2d"""

=======
    tile_and_bind3d(s, output, co, oh, ow, TC, TH, TW)

    return s

##### WINOGRAD TEMPLATE #####
def _pick_tile_size(data, kernel):
    N, CI, H, W = get_const_tuple(data.shape)

    if H % 4 == 0:
        return 4
    else:
        return 2

@conv2d_mali.register('winograd')
def decl_winograd(cfg, data, kernel, strides, padding, layout, out_dtype):
    tile_size = _pick_tile_size(data, kernel)
    return _decl_winograd(cfg, data, kernel, strides, padding, layout, out_dtype, tile_size)

def _decl_winograd(cfg, data, kernel, strides, padding, layout, out_dtype, tile_size):
    N, CI, IH, IW = get_const_tuple(data.shape)
    if len(kernel.shape) == 4:
        pre_computed = False
        CO, _, KH, KW = get_const_tuple(kernel.shape)
    else:
        pre_computed = True
        H_CAT, W_CAT, CO, CI, VC = get_const_tuple(kernel.shape)
        CO *= VC
        KH, KW = H_CAT - tile_size + 1, W_CAT - tile_size + 1
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    HPAD, WPAD, _, _ = get_pad_tuple(padding, kernel)

    assert layout == 'NCHW'
    assert KH == 3 and KW == 3 and HPAD == 1 and WPAD == 1 and HSTR == 1 and WSTR == 1
    data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")

    if tile_size == 4:
        G_data = np.array([
            [1 / 4.0, 0, 0],
            [-1 / 6.0, -1 / 6.0, -1 / 6.0],
            [-1 / 6.0, 1 / 6.0, -1 / 6.0],
            [1 / 24.0, 1 / 12.0, 1 / 6.0],
            [1 / 24.0, -1 / 12.0, 1 / 6.0],
            [0, 0, 1]], out_dtype)

        B_data = np.array([
            [4, 0, 0, 0, 0, 0],
            [0, -4, 4, -2, 2, 4],
            [-5, -4, -4, -1, -1, 0],
            [0, 1, -1, 2, -2, -5],
            [1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 1]], out_dtype)

        A_data = np.array([
            [1, 0, 0, 0],
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 2, 4, 8],
            [1, -2, 4, -8],
            [0, 0, 0, 1]], out_dtype)
    elif tile_size == 2:
        G_data = np.array([
            [1, 0, 0],
            [1.0/2, 1.0/2, 1.0/2],
            [1.0/2, -1.0/2, 1.0/2],
            [0, 0, 1]], out_dtype)

        B_data = np.array([
            [1, 0, 0, 0],
            [0, 1, -1, 1],
            [-1, 1, 1, 0],
            [0, 0, 0, -1]], out_dtype)

        A_data = np.array([
            [1, 0],
            [1, 1],
            [1, -1],
            [0, -1]], out_dtype)
    else:
        raise ValueError("Unsupported tile size for winograd: " + str(tile_size))

    m = A_data.shape[1]
    r = 3
    alpha = m + r - 1

    H = (IH + 2 * HPAD - 3) // HSTR + 1
    W = (IW + 2 * WPAD - 3) // WSTR + 1
    nH, nW = (H + m-1) // m, (W + m-1) // m
    P = N * nH * nW

    ##### space definition begin #####
    tile_bna_candidates = [1, 2, 4, 8, 16]
    factors = get_factors(CO)
    cfg.define_knob('tile_bna', [x for x in tile_bna_candidates if x in factors])
    cfg.define_knob('tile_bnb', [1, 2, 4, 8, 16])
    cfg.define_split('tile_t1', CI, num_outputs=2, max_factor=128)
    cfg.define_split('tile_t2', CO, num_outputs=2, max_factor=128)
    cfg.define_split('c_unroll', CI, num_outputs=2, max_factor=8)
    cfg.define_knob('yt', [1, 2, 4, 8, 16, 32])
    ##### space definition end #####

    if cfg.is_fallback:
        cfg['tile_bnb'].val = 4
        cfg['tile_bna'].val = 4
        while CO % cfg['tile_bna'].val != 0:
            cfg['tile_bna'].val //= 2
        cfg['yt'].val = 8
        cfg.fallback_split('tile_t1', [-1, 128])
        cfg.fallback_split('tile_t2', [-1, 128])
        cfg.fallback_split('c_unroll', [-1, 8])

    bna = cfg['tile_bna'].val
    bnb = cfg['tile_bnb'].val

    P_round = (P + bnb - 1) // bnb * bnb
    assert CO % bna == 0 and P_round % bnb == 0

    # pack input tile
    input_tile = tvm.compute((CI, P_round // bnb, alpha, alpha, bnb), lambda ci, b, eps, nu, bb: \
         tvm.select(b * bnb + bb < P,
                    data_pad[(b*bnb+bb) // (nH*nW)][ci][(b*bnb+bb) // nW % nH * m + eps]
                    [(b*bnb+bb) % nW * m + nu], tvm.const(0, data_pad.dtype)), name='d')

    # transform kernel
    if pre_computed:
        U = kernel
    else:
        G = const_matrix(G_data, 'G')
        r_kh = tvm.reduce_axis((0, KH), 'r_kh')
        r_kw = tvm.reduce_axis((0, KW), 'r_kw')
        U = tvm.compute((alpha, alpha, CO // bna, CI, bna), lambda eps, nu, co, ci, vco:
                        tvm.sum(kernel[co * bna + vco][ci][r_kh][r_kw] * G[eps][r_kh] * G[nu][r_kw],
                                axis=[r_kh, r_kw]), name='U')

    # transform image
    B = const_matrix(B_data, 'B')
    r_a = tvm.reduce_axis((0, alpha), 'r_a')
    r_b = tvm.reduce_axis((0, alpha), 'r_b')
    V = tvm.compute((alpha, alpha, P_round // bnb, CI, bnb), lambda eps, nu, p, ci, vp:
                    tvm.sum(input_tile[ci][p][r_a][r_b][vp] * B[r_a][eps] * B[r_b][nu],
                            axis=[r_a, r_b]), name='V')

    # batch gemm
    ci = tvm.reduce_axis((0, CI), name='c')
    M = tvm.compute((alpha, alpha, CO, P_round), lambda eps, nu, co, p:
                    tvm.sum(U[eps][nu][co // bna][ci][co % bna] *
                            V[eps][nu][p // bnb][ci][p % bnb], axis=ci), name='M')

    A = const_matrix(A_data, 'A')
    r_a = tvm.reduce_axis((0, alpha), 'r_a')
    r_b = tvm.reduce_axis((0, alpha), 'r_b')
    Y = tvm.compute((CO, P, m, m), lambda co, p, vh, vw:
                    tvm.sum(M[r_a][r_b][co][p] * A[r_a][vh] * A[r_b][vw],
                            axis=[r_a, r_b]), name='Y')

    # unpack output
    output = tvm.compute((N, CO, H, W), lambda n, co, h, w:
                         Y[co][n * nH * nW + (h//m) * nW + w//m][h % m][w % m]
                         # thw following term is used to make the padding effective,
                         # otherwise the padding will be eliminated by bound inference
                         + tvm.const(0, out_dtype) * M[alpha-1][alpha-1][CO-1][P_round-1],
                         name='output', tag='winograd_conv2d_output',
                         attrs={'workload': _winograd_conv_arg_to_workload(
                             data, kernel, strides, padding, layout, out_dtype, tile_size)})

    # we have to manually assign effective GFLOP for winograd
    cfg.add_flop(2 * N * CO * H * W * KH * KW * CI)
    return output

def _schedule_winograd(cfg, s, op):
    """schedule winograd fast convolution F(2x2, 3x3) for conv2d"""
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
    # get ops and tensors
    output = op.output(0)

    Y = op.input_tensors[0]
    M, A = s[Y].op.input_tensors
    U, V = s[M].op.input_tensors
<<<<<<< HEAD
    kernel, G = s[U].op.input_tensors
    d, B = s[V].op.input_tensors
    data_pad = s[d].op.input_tensors[0]
    data = s[data_pad].op.input_tensors[0]

    # dilation
    if isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag:
        s[kernel].compute_inline()
=======
    d, B = s[V].op.input_tensors
    data_pad = s[d].op.input_tensors[0]
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

    # padding
    s[data_pad].compute_inline()

<<<<<<< HEAD
    # pack input tiles
    c, b, eps, nu, bb = s[d].op.axis
    s[d].reorder(eps, nu, bb)
    aha = s[d].fuse(eps, nu)
    s[d].unroll(bb)
    tile_and_bind3d(s, d, c, b, aha, 4, 1, 1)

    # transform kernel
    s[G].compute_inline()
    eps, nu, k, c, kk, = s[U].op.axis
    r_kh, r_kw = s[U].op.reduce_axis
    s[U].reorder(k, c, kk, eps, nu, r_kh, r_kw)
    _ = [s[U].unroll(x) for x in [eps, nu, r_kh, r_kw]]
    s[U].vectorize(kk)
    tile_and_bind(s, U, k, c, 1, 256)

    # transform image
    s[B].compute_inline()
    eps, nu, b, c, bb = s[V].op.axis
    r_eps, r_nu = s[V].op.reduce_axis
    s[V].reorder(b, c, bb, eps, nu, r_nu, r_eps)
    _ = [s[V].unroll(x) for x in [eps, nu, r_eps, r_nu]]
    s[V].vectorize(bb)
    tile_and_bind(s, V, b, c, 2, 1)

    # batch gemm
    bna, bnb = 4, 4
    if data.dtype == 'float16':
        bnb *= 2

    eps, nu, k, b = s[M].op.axis
    c = s[M].op.reduce_axis[0]
    yo, xo, yi, xi = s[M].tile(k, b, bna, bnb)
    s[M].reorder(c, yi, xi)
    c, c_unroll = s[M].split(c, 2)
=======
    # transform kernel
    if isinstance(U.op, tvm.tensor.ComputeOp):
        kernel, G = s[U].op.input_tensors
        s[G].compute_inline()
        eps, nu, co, ci, vco, = s[U].op.axis
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # kernel transformation will be pre-computed during compilation, so we skip
            # this part to make tuning records correct
            s[U].pragma(eps, 'debug_skip_region')
        else:
            r_kh, r_kw = s[U].op.reduce_axis
            s[U].reorder(co, ci, eps, nu, r_kh, r_kw, vco)
            _ = [s[U].unroll(x) for x in [eps, nu, r_kh, r_kw]]
            s[U].vectorize(vco)
            tile_and_bind(s, U, co, ci, 1, 256)

        # dilation
        if isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag:
            s[kernel].compute_inline()

    # transform image
    s[B].compute_inline()
    VL = s.cache_write(V, 'local')

    eps, nu, p, ci, vp = s[V].op.axis
    s[V].reorder(p, ci, eps, nu, vp)
    for axis in [eps, nu]:
        s[V].unroll(axis)
    s[V].vectorize(vp)
    fused = s[V].fuse(p, ci)

    bb, tt = cfg['tile_t1'].apply(s, V, fused)
    s[V].bind(bb, tvm.thread_axis('blockIdx.x'))
    s[V].bind(tt, tvm.thread_axis('threadIdx.x'))

    eps, nu, p, ci, vp = s[VL].op.axis
    r_a, r_b = s[VL].op.reduce_axis
    for axis in [eps, nu, r_a, r_b]:
        s[VL].unroll(axis)
    s[VL].vectorize(vp)
    s[d].compute_at(s[V], tt)
    s[VL].compute_at(s[V], tt)

    # batch gemm
    bna = cfg['tile_bna'].val
    bnb = cfg['tile_bnb'].val

    eps, nu, k, b = s[M].op.axis
    alpha = eps.dom.extent
    c = s[M].op.reduce_axis[0]
    yo, xo, yi, xi = s[M].tile(k, b, bna, bnb)
    c, c_unroll = cfg['c_unroll'].apply(s, M, c)
    s[M].reorder(yo, xo, c, c_unroll, yi, xi)
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
    s[M].unroll(c_unroll)
    s[M].unroll(yi)
    s[M].vectorize(xi)
    z = s[M].fuse(eps, nu)
<<<<<<< HEAD
    tile_and_bind3d(s, M, z, yo, xo, 1, 8, 1)
=======
    tile_and_bind3d(s, M, z, yo, xo, 1, cfg['yt'].val, 1)
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

    # inverse transform
    s[A].compute_inline()
    k, b, vh, vw = s[Y].op.axis
<<<<<<< HEAD
    r_eps, r_nu = s[Y].op.reduce_axis
    _ = [s[Y].unroll(x) for x in [vh, vw, r_eps, r_nu]]
    tile_and_bind(s, Y, k, b, 4, 1)

    # schedule output
    if output.op in s.outputs:  # no bias
        output = output
    else:                       # has bias
        s[output].compute_inline()
        output = s.outputs[0]

    _, k, h, w = s[output].op.axis
    tile_and_bind3d(s, output, k, h, w, 1, 2, 2)
=======
    r_a, r_b = s[Y].op.reduce_axis
    for axis in [vh, vw, r_a, r_b]:
        s[Y].unroll(axis)

    # schedule output and fusion
    if output.op not in s.outputs:
        s[output].compute_inline()
        output = s.outputs[0]

    n, co, h, w = s[output].op.axis
    m = alpha - 3 + 1
    h, w, hi, wi = s[output].tile(h, w, m, m)
    s[output].unroll(hi)
    s[output].unroll(wi)
    fused = s[output].fuse(n, co, h, w)
    bb, tt = cfg['tile_t2'].apply(s, output, fused)
    s[output].bind(bb, tvm.thread_axis('blockIdx.x'))
    s[output].bind(tt, tvm.thread_axis('threadIdx.x'))

    s[Y].compute_at(s[output], tt)

@conv2d_alter_layout.register(["mali"])
def _alter_conv2d_layout(attrs, inputs, tinfos):
    try:
        return _alter_conv2d_layout_arm(attrs, inputs, tinfos)
    except KeyError:  # to filter out fallback opencl templates
        return None

##### REGISTER TOPI COMPUTE / SCHEDULE FOR WINOGRAD WITH WEIGHT TRANSFORM #####
@conv2d_winograd_without_weight_transform.register(['mali'])
@autotvm.task.dispatcher
def winograd_ww_config_dispatcher_(data, kernel, strides, padding, layout, out_dtype, tile_size):
    return _winograd_conv_arg_to_workload(data, kernel, strides, padding, layout, out_dtype,
                                          tile_size)


@winograd_ww_config_dispatcher_.register(['winograd'])
def decl_winograd_ww(cfg, data, kernel, strides, padding, layout, out_dtype, tile_size):
    return _decl_winograd(cfg, data, kernel, strides, padding, layout, out_dtype,
                          tile_size)


@autotvm.task.register_topi_schedule(schedule_conv2d_winograd_without_weight_transform,
                                     'mali', ['winograd'])
def schedule_conv2d_winograd_without_weight_transform_(cfg, outs):
    """TOPI schedule callback"""
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if 'winograd_conv2d_output' in op.tag:
            _schedule_winograd(cfg, s, op)

    traverse_inline(s, outs[0].op, _callback)
    return s


##### SCHECULE UTILITIES #####
def tile_and_bind(s, tensor, y, x, y_factor, x_factor=None):
    """ tile and bind to GPU threads """
    x_factor = x_factor or y_factor
    yo, xo, yi, xi = s[tensor].tile(y, x, y_factor, x_factor)
    s[tensor].bind(xo, tvm.thread_axis("blockIdx.x"))
    s[tensor].bind(xi, tvm.thread_axis("threadIdx.x"))
    s[tensor].bind(yo, tvm.thread_axis("blockIdx.y"))
    s[tensor].bind(yi, tvm.thread_axis("threadIdx.y"))
    return yo, xo, yi, xi


def tile_and_bind3d(s, tensor, z, y, x, z_factor=2, y_factor=None, x_factor=None):
    """ tile and bind 3d """
    y_factor = y_factor or z_factor
    x_factor = x_factor or y_factor
    zo, zi = s[tensor].split(z, z_factor)
    yo, yi = s[tensor].split(y, y_factor)
    xo, xi = s[tensor].split(x, x_factor)
    s[tensor].bind(zo, tvm.thread_axis("blockIdx.z"))
    s[tensor].bind(zi, tvm.thread_axis("threadIdx.z"))
    s[tensor].bind(yo, tvm.thread_axis("blockIdx.y"))
    s[tensor].bind(yi, tvm.thread_axis("threadIdx.y"))
    s[tensor].bind(xo, tvm.thread_axis("blockIdx.x"))
    s[tensor].bind(xi, tvm.thread_axis("threadIdx.x"))
    s[tensor].reorder(zo, yo, xo, zi, yi, xi)
    return zo, yo, xo, zi, yi, xi
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
