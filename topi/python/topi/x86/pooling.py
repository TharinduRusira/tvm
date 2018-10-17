# pylint: disable=invalid-name, unused-variable
"""Schedule for pooling operators"""
import tvm
from .. import generic
from .. import tag

<<<<<<< HEAD
def _parallel_sch(sch):
    if len(sch.op.axis) >= 5:
        fused = sch.fuse(sch.op.axis[0], sch.op.axis[1], sch.op.axis[2])
        sch.parallel(fused)
    elif len(sch.op.axis) >= 3:
        fused = sch.fuse(sch.op.axis[0], sch.op.axis[1])
        sch.parallel(fused)
    else:
        sch.parallel(sch.op.axis[0])


@generic.schedule_pool.register(["cpu"])
def schedule_pool(outs):
=======
def _parallel_sch(sch, oshape, do_vectorize=False):
    def vectorize(fused_axis, num_parallel_axis, vectorize_limit=64):
        """Internal vectorization utility function."""
        reorder_axis = [fused_axis]
        for i in range(num_parallel_axis, len(sch.op.axis) - 1):
            reorder_axis.append(sch.op.axis[i])
        kw, kh = sch.op.reduce_axis
        fuse_k = sch.fuse(kw, kh)
        c = sch.op.axis[len(sch.op.axis) - 1]
        reorder_axis += [fuse_k, c]
        sch.reorder(*reorder_axis)
        inner_length = oshape[len(oshape) - 1].value
        if inner_length <= vectorize_limit:
            sch.vectorize(c)
        else:
            split_factor = 1
            for i in range(vectorize_limit, 1, -1):
                if inner_length % i == 0:
                    split_factor = i
                    break
            if split_factor > 1:
                _, c_i = sch.split(c, split_factor)
                sch.vectorize(c_i)

    if len(sch.op.axis) >= 5:
        fused = sch.fuse(sch.op.axis[0], sch.op.axis[1], sch.op.axis[2])
        if do_vectorize:
            vectorize(fused, 3)

    elif len(sch.op.axis) >= 3:
        fused = sch.fuse(sch.op.axis[0], sch.op.axis[1])
        if do_vectorize:
            vectorize(fused, 2)
    else:
        sch.parallel(sch.op.axis[0])
        return
    sch.parallel(fused)


@generic.schedule_pool.register(["cpu"])
def schedule_pool(outs, layout):
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
    """Schedule for pool

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
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
<<<<<<< HEAD
=======
    scheduled_ops = []
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

    def _schedule(PaddedInput, Pool):
        if isinstance(PaddedInput.op, tvm.tensor.ComputeOp):
            s[PaddedInput].compute_inline()
<<<<<<< HEAD
        _parallel_sch(s[Pool])
=======
        do_vectorize = layout[-1] not in "HWhw"
        _parallel_sch(s[Pool], outs[0].shape, do_vectorize)
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

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


@generic.schedule_global_pool.register(["cpu"])
def schedule_global_pool(outs):
    """Schedule for global pool

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
=======
    scheduled_ops = []

>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
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
        # schedule pool
        elif OP.tag.startswith('global_pool'):
            Pool = OP.output(0)
<<<<<<< HEAD
            _parallel_sch(s[Pool])
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)
=======
            _parallel_sch(s[Pool], outs[0].shape)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

        scheduled_ops.append(OP)

>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
    traverse(outs[0].op)
    return s
