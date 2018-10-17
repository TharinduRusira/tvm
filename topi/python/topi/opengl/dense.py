# pylint: disable=invalid-name, unused-variable
"""Schedule for dense operator"""
from __future__ import absolute_import as _abs
import tvm
from .. import tag
from .. import generic

@generic.schedule_dense.register(["opengl"])
def schedule_dense(outs):
    """Schedule for dense operator.

    Parameters
    ----------
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
=======
    scheduled_ops = []

>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
    def _schedule(Dense):
        if Dense.op in s.outputs:
            Out = Dense
        else:
            Out = outs[0].op.output(0)
            s[Dense].opengl()
        s[Out].opengl()

    def traverse(OP):
        """Internal travserse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
<<<<<<< HEAD
                if tensor.op.input_tensors:
=======
                if tensor.op.input_tensors and tensor.op not in scheduled_ops:
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
                    traverse(tensor.op)
        # schedule dense
        elif OP.tag == 'dense':
            Dense = OP.output(0)
            _schedule(Dense)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

<<<<<<< HEAD
=======
        scheduled_ops.append(OP)

>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
    traverse(outs[0].op)
    return s
