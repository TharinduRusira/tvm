# pylint: disable=invalid-name,unused-variable
"""dense schedule on ARM Mali GPU"""

from __future__ import absolute_import as _abs

import tvm
<<<<<<< HEAD

from .. import generic
from .. import util
from .. import tag

@generic.schedule_dense.register(["mali"])
def schedule_dense(outs):
=======
from tvm import autotvm

from .. import generic, nn
from ..util import traverse_inline

autotvm.register_topi_compute(nn.dense, 'mali', 'direct', nn.dense.fdefault)

@autotvm.register_topi_schedule(generic.schedule_dense, 'mali', 'direct')
def schedule_dense(cfg, outs):
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
    """Schedule for dense operator.

    Parameters
    ----------
<<<<<<< HEAD
=======
    cfg: ConfigEntity
        The config entity for this template
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
    outs: Array of Tensor
        The computation graph description of dense
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for dense.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
<<<<<<< HEAD
    def _schedule(dense):
        data = s[dense].op.input_tensors[0]
        weight = s[dense].op.input_tensors[1]

        hidden = util.get_const_int(weight.shape[1])
        out = util.get_const_int(weight.shape[0])

        # set tunable parameter
        tune_config = getattr(tvm.target.current_target(), "tune_config", None)
        if tune_config is None:
            if hidden > 8192:
                num_thread = 32
                unroll_step = 32
            else:
                if out <= 1024:
                    num_thread = 32
                    unroll_step = 16
                else:
                    num_thread = 256
                    unroll_step = 32

            if data.dtype == 'float16':
                if hidden > 8192:
                    num_thread = 2
                    unroll_step = 32
                else:
                    num_thread = 8
                    unroll_step = 256
        else:
            num_thread = tune_config['num_thread']
            unroll_step = tune_config['unroll_step']

        def fuse_and_bind(s, tensor, axis=None, num_thread=None):
            """ fuse all the axis and bind to GPU threads """
            axis = axis or s[tensor].op.axis
            fused = s[tensor].fuse(*axis)
            max_threads = tvm.target.current_target(allow_none=False).max_num_threads
            bx, tx = s[tensor].split(fused, num_thread or max_threads)
            s[tensor].bind(bx, tvm.thread_axis("blockIdx.x"))
            s[tensor].bind(tx, tvm.thread_axis("threadIdx.x"))
            return bx, tx

        output = outs[0]
        bx, tx = fuse_and_bind(s, output, num_thread=num_thread)

        k = s[dense].op.reduce_axis[0]
        k, k_unroll = s[dense].split(k, unroll_step)
        s[dense].unroll(k_unroll)

        if dense.op not in s.outputs:
            s[dense].compute_at(s[output], tx)

#        bias = s[outs[0]].op.input_tensors[1]
#        print(tvm.lower(s, [data, weight, bias, outs[0]], simple_mode=True))

    def traverse(OP):
        """Internal travserse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        # schedule dense
        elif OP.tag == 'dense':
            dense = OP.output(0)
            _schedule(dense)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

    traverse(outs[0].op)
    return s
=======

    def _callback(op):
        if op.tag == 'dense':
            vec_size = [1, 2, 4, 8, 16]
            max_unroll = 32

            dense = op.output(0)
            output = outs[0]

            y, x = s[output].op.axis
            c = s[dense].op.reduce_axis[0]

            ##### space definition begin #####
            cfg.define_split('tile_y', y, num_outputs=3)
            cfg.define_split('tile_x', x, num_outputs=3)
            cfg.define_split('c_unroll', c, num_outputs=2, max_factor=64)

            # fallback support
            if cfg.is_fallback:
                ref_log = autotvm.tophub.load_reference_log(
                    'mali', 'rk3399', 'dense', 'direct')
                cfg.fallback_with_reference_log(ref_log)
            ##### space definition end #####

            if dense.op in s.outputs:
                dense = s.cache_write(output, 'local')

            by, ty, yi = cfg['tile_y'].apply(s, output, y)
            bx, tx, xi = cfg['tile_x'].apply(s, output, x)

            s[output].bind(by, tvm.thread_axis('blockIdx.y'))
            s[output].bind(bx, tvm.thread_axis('blockIdx.x'))
            s[output].bind(ty, tvm.thread_axis('threadIdx.y'))
            s[output].bind(tx, tvm.thread_axis('threadIdx.x'))

            if cfg['tile_y'].size[-1] < max_unroll:
                s[output].unroll(yi)
            if cfg['tile_x'].size[-1] in vec_size:
                s[output].vectorize(xi)
            s[dense].compute_at(s[output], tx)

            k = s[dense].op.reduce_axis[0]
            y, x = s[dense].op.axis
            k, k_unroll = cfg['c_unroll'].apply(s, dense, k)
            s[dense].reorder(k, k_unroll, y, x)
            s[dense].unroll(k_unroll)
            if cfg['tile_y'].size[-1] < max_unroll:
                s[dense].unroll(y)
            if cfg['tile_x'].size[-1] in vec_size:
                s[dense].vectorize(x)

    traverse_inline(s, outs[0].op, _callback)
    return s

def fuse_and_bind(s, tensor, axis=None, num_thread=None):
    """ fuse all the axis and bind to GPU threads """
    axis = axis or s[tensor].op.axis
    fused = s[tensor].fuse(*axis)
    bx, tx = s[tensor].split(fused, num_thread)
    s[tensor].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[tensor].bind(tx, tvm.thread_axis("threadIdx.x"))
    return bx, tx
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
