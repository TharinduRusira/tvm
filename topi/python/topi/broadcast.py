"""Broadcast operators"""
from __future__ import absolute_import as _abs
<<<<<<< HEAD
import tvm
from .import tag
from .util import get_const_tuple, equal_const_int, get_const_int
=======
from .import cpp as _cpp
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

def broadcast_to(data, shape):
    """Broadcast the src to the target shape

    We follows the numpy broadcasting rule.
    See also https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    Parameters
    ----------
    data : tvm.Tensor
        The input data

    shape : list or tuple
        The target shape to be broadcasted.

    Returns
    -------
    ret : tvm.Tensor
    """
    return _cpp.broadcast_to(data, shape)


def add(lhs, rhs):
    """Addition with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.add(lhs, rhs)


def subtract(lhs, rhs):
    """Subtraction with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.subtract(lhs, rhs)


def multiply(lhs, rhs):
    """Multiplication with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.multiply(lhs, rhs)


def divide(lhs, rhs):
    """Division with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.divide(lhs, rhs)


def mod(lhs, rhs):
    """Modulus with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.mod(lhs, rhs)


def maximum(lhs, rhs):
    """Take element-wise maximum of two tensors with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.maximum(lhs, rhs)


def minimum(lhs, rhs):
    """Take element-wise maximum of two tensors with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
<<<<<<< HEAD
    def _bcast_to_arg_eval(data, bcast_info, *indices):
        indices_tuple = []
        for i, ind in enumerate(indices):
            if bcast_info[i] == 0:
                indices_tuple.append(ind)
            elif bcast_info[i] == 1:
                indices_tuple.append(0)
        return data[tuple(indices_tuple)]
    original_shape = data.shape
    shape = [get_const_int(i) for i in shape]
    bcast_info = _get_bcast_info(original_shape=original_shape, target_shape=shape)
    ret = tvm.compute(shape,
                      lambda *indices: _bcast_to_arg_eval(data,
                                                          bcast_info,
                                                          *indices), name=data.name + "_broadcast")
    return ret


@tvm.tag_scope(tag=tag.BROADCAST)
def broadcast_binary_op(lhs, rhs, func, name="bop"):
    """Binary operands that will automatically broadcast the inputs
=======
    return _cpp.minimum(lhs, rhs)
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199


def power(lhs, rhs):
    """Power with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.power(lhs, rhs)


def left_shift(lhs, rhs):
    """Left shift with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.left_shift(lhs, rhs)


def right_shift(lhs, rhs):
    """Right shift with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.right_shift(lhs, rhs)


def greater(lhs, rhs):
    """Compute (lhs>rhs) with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.greater(lhs, rhs)


def less(lhs, rhs):
    """Compute (lhs<rhs) with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.less(lhs, rhs)


def equal(lhs, rhs):
    """Compute (lhs==rhs) with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.equal(lhs, rhs)


def not_equal(lhs, rhs):
    """Compute (lhs!=rhs) with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.not_equal(lhs, rhs)


def greater_equal(lhs, rhs):
    """Compute (lhs>=rhs) with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.greater_equal(lhs, rhs)


def less_equal(lhs, rhs):
    """Compute (lhs<=rhs) with auto-broadcasting

    Parameters
    ----------
    lhs : tvm.Tensor or Expr
        The left operand
    rhs : tvm.Tensor or Expr
        The right operand

    Returns
    -------
    ret : tvm.Tensor or Expr
        Returns Expr if both operands are Expr.
        Otherwise returns Tensor.
    """
    return _cpp.less_equal(lhs, rhs)
