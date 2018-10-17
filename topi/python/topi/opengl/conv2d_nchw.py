#pylint: disable=invalid-name, no-member, too-many-locals, too-many-statements, too-many-arguments, too-many-branches, line-too-long
"""Schedule for conv2d_nchw with auto fusion"""
import tvm
from .. import tag
from .. import generic

@generic.schedule_conv2d_nchw.register(["opengl"])
def schedule_conv2d_nchw(outs):
    """Schedule for conv2d_nchw.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of conv2d_nchw
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d_nchw.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
<<<<<<< HEAD
=======
    scheduled_ops = []

>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
    def _schedule(conv2d, data):
        if conv2d.op in s.outputs:
            Out = conv2d
        else:
            Out = outs[0].op.output(0)
            s[conv2d].opengl()
        s[Out].opengl()
        s[data].opengl()

    def traverse(OP):
        """Internal travserse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].opengl()
            for tensor in OP.input_tensors:
<<<<<<< HEAD
                if tensor.op.input_tensors:
=======
                if tensor.op.input_tensors and tensor.op not in scheduled_ops:
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
                    traverse(tensor.op)
        # schedule conv2d_nchw
        elif OP.tag.startswith('conv2d_nchw'):
            conv2d = OP.output(0)
            data = OP.input_tensors[0]
            kernel = OP.input_tensors[1]
            if isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag:
                s[kernel].compute_inline()
            _schedule(conv2d, data)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

<<<<<<< HEAD
=======
        scheduled_ops.append(OP)

>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
    traverse(outs[0].op)
    return s
