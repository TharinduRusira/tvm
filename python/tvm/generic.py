"""Generic opertors in TVM.
We follow the numpy naming convention for this interface
(e.g., tvm.generic.multitply ~ numpy.multiply).
The default implementation is used by tvm.ExprOp.
"""
# pylint: disable=unused-argument
from . import make as _make

#Operator precedence used when overloading.
__op_priority__ = 0

def add(lhs, rhs):
    """Generic add operator.

    Parameters
    ----------
    lhs : object
        The left operand.
    rhs : object
        The right operand.

    Returns
    -------
    op : tvm.Expr
        The result Expr of add operaton.
    """
<<<<<<< HEAD
    return _make.Add(lhs, rhs)
=======
    return _make._OpAdd(lhs, rhs)
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199


def subtract(lhs, rhs):
    """Generic subtract operator.

    Parameters
    ----------
    lhs : object
        The left operand.
    rhs : object
        The right operand.

    Returns
    -------
    op : tvm.Expr
        The result Expr of subtract operaton.
    """
<<<<<<< HEAD
    return _make.Sub(lhs, rhs)
=======
    return _make._OpSub(lhs, rhs)
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199


def multiply(lhs, rhs):
    """Generic multiply operator.

    Parameters
    ----------
    lhs : object
        The left operand.
    rhs : object
        The right operand.

    Returns
    -------
    op : tvm.Expr
        The result Expr of multiply operaton.
    """
<<<<<<< HEAD
    return _make.Mul(lhs, rhs)
=======
    return _make._OpMul(lhs, rhs)
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199


def divide(lhs, rhs):
    """Generic divide operator.

    Parameters
    ----------
    lhs : object
        The left operand.
    rhs : object
        The right operand.

    Returns
    -------
    op : tvm.Expr
        The result Expr of divide operaton.
    """
<<<<<<< HEAD
    return _make.Div(lhs, rhs)
=======
    return _make._OpDiv(lhs, rhs)


def cast(src, dtype):
    """Generic cast operator.

    Parameters
    ----------
    src : object
        The source operand.

    Returns
    -------
    op : tvm.Expr
        The result Expr of divide operaton.
    """
    return _make.static_cast(dtype, src)
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
