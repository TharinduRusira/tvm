# pylint: disable=invalid-name,consider-using-enumerate,unused-argument,len-as-condition
"""Elementwise operators"""
from __future__ import absolute_import as _abs
<<<<<<< HEAD
import tvm
from . import tag

@tvm.tag_scope(tag=tag.ELEMWISE)
def elemwise_sum(xs, num_args):
=======
from . import cpp

def elemwise_sum(xs):
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
    """Perform element-wise sum on inputs

    Parameters
    ----------
    xs : list of tvm.Tensor
        Input arguments.
<<<<<<< HEAD
    num_args : int
        Number of arguments
=======
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
<<<<<<< HEAD
    assert len(xs) > 0, "elemwise sum must have at least one input tensor."

    def _compute(*i):
        return sum([x(*i) for x in xs])

    return tvm.compute(xs[0].shape, _compute)


@tvm.tag_scope(tag=tag.ELEMWISE)
=======
    return cpp.elemwise_sum(xs)


>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
def full(shape, dtype, fill_value):
    """Fill tensor with fill_value

    Parameters
    ----------
    shape : tuple
        Input tensor shape.
    dtype : str
        Data type
    fill_value : float
        Value to be filled

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
<<<<<<< HEAD
    return tvm.compute(shape, lambda *i: tvm.const(fill_value, dtype))


@tvm.tag_scope(tag=tag.ELEMWISE)
=======
    return cpp.full(shape, dtype, fill_value)


>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
def full_like(x, fill_value):
    """Construct a tensor with same shape as input tensor,
       then fill tensor with fill_value.

    Parameters
    ----------
    x : tvm.Tensor
        Input argument.
    fill_value : float
        Value to be filled

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
<<<<<<< HEAD
    dtype = x.dtype
    return tvm.compute(x.shape, lambda *i: tvm.const(fill_value, dtype))


@tvm.tag_scope(tag=tag.ELEMWISE)
def greater(lhs, rhs, out_type=tvm.int8):
    """Compare two input tensors element-wise and return an mask tensor
       which contains 1 if lhs > rhs holds else 0

    Parameters
    ----------
    lhs : tvm.Tensor
        Left input argument.
    rhs : tvm.Tensor
        Right argument.
    out_type: str
        Output data type. Default is int8

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(lhs.shape,
                       lambda *i: tvm.select(lhs(*i) > rhs(*i),
                                             tvm.const(1, out_type),
                                             tvm.const(0, out_type)))

@tvm.tag_scope(tag=tag.ELEMWISE)
def less(lhs, rhs, out_type=tvm.int8):
    """Compare two input tensors element-wise and return an mask tensor
       which contains 1 if lhs < rhs holds else 0

    Parameters
    ----------
    lhs : tvm.Tensor
        Left input argument.
    rhs : tvm.Tensor
        Right argument.
    out_type: str
        Output data type. Default is int8

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    return tvm.compute(lhs.shape,
                       lambda *i: tvm.select(lhs(*i) < rhs(*i),
                                             tvm.const(1, out_type),
                                             tvm.const(0, out_type)))
=======
    return cpp.full_like(x, fill_value)
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
