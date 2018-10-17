<<<<<<< HEAD
# pylint: disable=invalid-name, unused-variable
=======
# pylint: disable=invalid-name, unused-variable, unused-argument
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
"""Schedule for pooling operators"""
import tvm
from .. import tag
from .. import generic

@generic.schedule_global_pool.register(["opengl"])
def schedule_global_pool(outs):
    """Schedule for global_pool.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of global_pool
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for global_pool.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
<<<<<<< HEAD
=======
    scheduled_ops = []

>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
    def _schedule(Pool):
        if Pool.op in s.outputs:
            Out = Pool
        else:
            Out = outs[0].op.output(0)
            s[Pool].opengl()
        s[Out].opengl()

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
        # schedule global_pool
        elif OP.tag.startswith('global_pool'):
            Pool = OP.output(0)
            _schedule(Pool)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

<<<<<<< HEAD
=======
        scheduled_ops.append(OP)

>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
    traverse(outs[0].op)
    return s


@generic.schedule_pool.register(["opengl"])
<<<<<<< HEAD
def schedule_pool(outs):
=======
def schedule_pool(outs, layout):
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
    """Schedule for pool.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of pool
        in the format of an array of tensors.

<<<<<<< HEAD
=======
    layout: str
        Data layout.

>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
    Returns
    -------
    s: Schedule
        The computation schedule for pool.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
<<<<<<< HEAD
=======
    scheduled_ops = []

>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
    def _schedule(PaddedInput, Pool):
        if isinstance(PaddedInput.op, tvm.tensor.ComputeOp):
            s[PaddedInput].opengl()
        if Pool.op in s.outputs:
            Out = Pool
        else:
            Out = outs[0].op.output(0)
            s[Pool].opengl()
        s[Out].opengl()

    def traverse(OP):
        """Internal travserse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
<<<<<<< HEAD
            for tensor in OP.input_tensors:
=======
            for tensor in OP.input_tensors and tensor.op not in scheduled_ops:
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        # schedule pool
        elif OP.tag.startswith('pool'):
            PaddedInput = OP.input_tensors[0]
            Pool = OP.output(0)
            _schedule(PaddedInput, Pool)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

<<<<<<< HEAD
=======
        scheduled_ops.append(OP)

>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
    traverse(outs[0].op)
    return s
