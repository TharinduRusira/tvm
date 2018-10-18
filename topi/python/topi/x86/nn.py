# pylint: disable=invalid-name,too-many-locals
"""x86 nn operators"""
from __future__ import absolute_import as _abs
import tvm
<<<<<<< HEAD
from .. import generic
from .. import tag
=======

from .. import generic
from ..util import traverse_inline
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

@generic.schedule_softmax.register(["cpu"])
def schedule_softmax(outs):
    """Schedule for softmax

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of softmax
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    x = outs[0]
    s = tvm.create_schedule([x.op for x in outs])
    tvm.schedule.AutoInlineInjective(s)
    if len(s[x].op.axis) >= 5:
        fused = s[x].fuse(s[x].op.axis[0], s[x].op.axis[1], s[x].op.axis[2])
        s[x].parallel(fused)
    elif len(s[x].op.axis) >= 3:
        fused = s[x].fuse(s[x].op.axis[0], s[x].op.axis[1])
        s[x].parallel(fused)
    else:
        s[x].parallel(s[x].op.axis[0])
    return s


@generic.schedule_dense.register(["cpu"])
def schedule_dense(outs):
    """Schedule for dense

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of pool
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """

    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

<<<<<<< HEAD
    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)

        if 'dense' in op.tag:
            C = op.output(0)
            x, y = C.op.axis

            # Write cache for blocks
            CC = s.cache_write(C, 'global')
=======
    def _callback(op):
        if 'dense' in op.tag:
            output = outs[0]
            dense = op.output(0)

            # Write cache for blocks
            if dense.op in s.outputs:
                CC = s.cache_write(dense, 'local')
            else:
                CC = dense
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

            # Tile
            bnx = 1
            bny = 4
<<<<<<< HEAD
            _, yo, _, yi = s[C].tile(x, y, bnx, bny)
            s[CC].compute_at(s[C], yo)
=======
            x, y = output.op.axis
            xo, yo, xi, yi = s[output].tile(x, y, bnx, bny)

>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
            xc, yc = s[CC].op.axis
            k, = s[CC].op.reduce_axis
            ko, ki = s[CC].split(k, factor=4)
            s[CC].reorder(ko, xc, ki, yc)
<<<<<<< HEAD
            s[CC].unroll(ki)
            s[CC].vectorize(yc)

            # Vectorization
            s[C].vectorize(yi)

            # Parallelization
            s[C].parallel(yo)

    traverse(outs[0].op)
=======

            s[CC].unroll(ki)
            s[CC].vectorize(yc)

            s[output].unroll(xi)
            s[output].vectorize(yi)

            fused = s[output].fuse(xo, yo)
            s[output].parallel(fused)
            s[CC].compute_at(s[output], fused)

    traverse_inline(s, outs[0].op, _callback)
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
    return s
