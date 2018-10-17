# pylint: disable=redefined-builtin,consider-using-enumerate,no-member
"""Reduce operators"""
from __future__ import absolute_import as _abs
from . import cpp

def _get_real_axis(ndim, axis):
    if axis is None:
        real_axis = list(range(ndim))
    else:
        if isinstance(axis, int):
            axis = [axis]
        else:
            assert isinstance(axis, (list, tuple))
        real_axis = []
        for ele in axis:
            if ele < 0:
                ele += ndim
            if ele >= ndim:
                raise ValueError(
                    "{} exceeds the maximum dimension {}. Received axis={}".format(ele, ndim, axis))
            real_axis.append(ele)
        real_axis.sort()
        real_axis = list(set(real_axis))  # Remove the duplicates
    return real_axis


def sum(data, axis=None, keepdims=False):
    """Sum of array elements over a given axis or a list of axes

    Parameters
    ----------
    data : tvm.Tensor
        The input tvm tensor

    axis : None or int or tuple of int
        Axis or axes along which a sum is performed.
        The default, axis=None, will sum all of the elements of the input array.
        If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    Returns
    -------
    ret : tvm.Tensor
    """
<<<<<<< HEAD
    ndim = len(data.shape)
    assert ndim != 0, "Reduce a dim-0 input is not supported!"
    real_axis = _get_real_axis(ndim, axis)
    reduce_axes = [tvm.reduce_axis((0, data.shape[i]), "k%d" %i) for i in real_axis]
    if keepdims:
        target_shape = [1 if i in real_axis else data.shape[i] for i in range(ndim)]
    else:
        target_shape = []
        for i in range(ndim):
            if i not in real_axis:
                target_shape.append(tvm.convert(data.shape[i]))
    def _compute(*indices):
        eval_range = []
        eval_indices = []
        if not keepdims:
            arg_counter = 0
        else:
            arg_counter = None
        red_counter = 0
        for i in range(len(data.shape)):
            if i in real_axis:
                eval_range.append(reduce_axes[red_counter])
                eval_indices.append(reduce_axes[red_counter].var)
                red_counter += 1
            else:
                if not keepdims:
                    eval_range.append(indices[arg_counter])
                    arg_counter += 1
                else:
                    eval_range.append(indices[i])
        if not is_idx_reduce:
            return func(data[tuple(eval_range)], axis=reduce_axes)
        idx = ravel_index(eval_indices, [data.shape[i] for i in real_axis])
        return func((idx, data[tuple(eval_range)]), axis=reduce_axes)
    if is_idx_reduce:
        temp_idx, temp_val = tvm.compute(target_shape, _compute, name=data.name + "_red_temp")
        out = tvm.compute(target_shape,
                          lambda *indices: _choose_idx(temp_idx, temp_val, *indices),
                          name=data.name + "_red")
    else:
        out = tvm.compute(target_shape, _compute, name=data.name + "_red")
    return out
=======
    return cpp.sum(data, axis, keepdims)
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199


def max(data, axis=None, keepdims=False):
    """Maximum of array elements over a given axis or a list of axes

    Parameters
    ----------
    data : tvm.Tensor
        The input tvm tensor

    axis : None or int or tuple of int
        Axis or axes along which the max operation is performed.
        The default, axis=None, will find the max element from all of the elements of the input
        array. If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    Returns
    -------
    ret : tvm.Tensor
    """
    return cpp.max(data, axis, keepdims)


def min(data, axis=None, keepdims=False):
    """Minimum of array elements over a given axis or a list of axes

    Parameters
    ----------
    data : tvm.Tensor
        The input tvm tensor

    axis : None or int or tuple of int
<<<<<<< HEAD
        Axis or axes along which the max operation is performed.
        The default, axis=None, will find the max element from all of the elements of the input
        array. If axis is negative it counts from the last to the first axis.
=======
        Axis or axes along which a minimum operation is performed.
        The default, axis=None, will find the minimum element from all of the elements of the
        input array. If axis is negative it counts from the last to the first axis.
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    Returns
    -------
    ret : tvm.Tensor
    """
    return cpp.min(data, axis, keepdims)


def argmax(data, axis=None, keepdims=False):
    """Returns the indices of the maximum values along an axis.

    Parameters
    ----------
    data : tvm.Tensor
        The input tvm tensor

    axis : None or int or tuple of int
<<<<<<< HEAD
        Axis or axes along which a minimum operation is performed.
        The default, axis=None, will find the minimum element from all of the elements of the
        input array. If axis is negative it counts from the last to the first axis.
=======
        Axis or axes along which a argmax operation is performed.
        The default, axis=None, will find the indices of the maximum element of the elements of
        the input array. If axis is negative it counts from the last to the first axis.
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    Returns
    -------
    ret : tvm.Tensor
    """
    return cpp.argmax(data, axis, keepdims)


def argmin(data, axis=None, keepdims=False):
    """Returns the indices of the minimum values along an axis.

    Parameters
    ----------
    data : tvm.Tensor
        The input tvm tensor

    axis : None or int or tuple of int
<<<<<<< HEAD
        Axis or axes along which a argmax operation is performed.
        The default, axis=None, will find the indices of the maximum element of the elements of
=======
        Axis or axes along which a argmin operation is performed.
        The default, axis=None, will find the indices of minimum element all of the elements of
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
        the input array. If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    Returns
    -------
    ret : tvm.Tensor
    """
    return cpp.argmin(data, axis, keepdims)


def prod(data, axis=None, keepdims=False):
    """Product of array elements over a given axis or a list of axes

    Parameters
    ----------
    data : tvm.Tensor
        The input tvm tensor

    axis : None or int or tuple of int
<<<<<<< HEAD
        Axis or axes along which a argmin operation is performed.
        The default, axis=None, will find the indices of minimum element all of the elements of
        the input array. If axis is negative it counts from the last to the first axis.
=======
        Axis or axes along which a prod operation is performed.
        The default, axis=None, will get the prod element over all of the elements of the
        input array. If axis is negative it counts from the last to the first axis.
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    Returns
    -------
    ret : tvm.Tensor
    """
    return cpp.prod(data, axis, keepdims)
