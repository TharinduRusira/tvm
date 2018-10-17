"""TVM operator upsampling compute."""
from __future__ import absolute_import
<<<<<<< HEAD
import tvm
from .. import util


def upsampling(data, scale, layout="NCHW"):
    """Perform nearest neighbor upsampling on the data.
       Bilinear upsampling is not supported.

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, channel, in_height, in_width]
        or  [batch, in_height, in_width, channel]

    scale: int
        upsampling scaling factor

    layout: string
        either "NCHW" or "NHWC"

=======
import topi
from ..util import simplify


def upsampling(data, scale, layout="NCHW", method='NEAREST_NEIGHBOR'):
    """Perform upsampling on the data.
       Nearest neighbor and bilinear upsampling are supported.

    Parameters
    ----------
    inputs : tvm.Tensor
        inputs is a 4-D tensor with shape
        [batch, channel, in_height, in_width]
        or  [batch, in_height, in_width, channel]

    scale : int
        Scaling factor

    layout : string, optional
        either "NCHW" or "NHWC"

    method : {"BILINEAR", "NEAREST_NEIGHBOR"}
        Method to be used for upsampling.

>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, channel, in_height*scale, in_width*scale]
        or [batch, in_height*scale, in_width*scale, channel]
    """

    if layout == "NCHW":
<<<<<<< HEAD
        return upsampling_nchw(data, scale)
    elif layout == "NHWC":
        return upsampling_nhwc(data, scale)
    else:
        raise ValueError("not support this layout {} yet".format(layout))


def upsampling_nchw(data, scale):
    """Perform nearest neighor upsampling on NCHW layout input.

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, channel, in_height, in_width]

    scale: int
        upsampling scaling factor

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, channel, in_height*scale, in_width*scale]
    """
    batch, channel, height, width = data.shape
    out_height = util.simplify(height * scale)
    out_width = util.simplify(width * scale)

    return tvm.compute((batch, channel, out_height, out_width), \
                        lambda n, c, h, w: data[n, c, h/scale, w/scale])


def upsampling_nhwc(data, scale):
    """Perform nearest neighor upsampling on NHWC layout input.

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, in_height, in_width, channel]

    scale: int
        upsampling scaling factor

    """

    batch, height, width, channel = data.shape
    out_height = util.simplify(height * scale)
    out_width = util.simplify(width * scale)

    return tvm.compute((batch, out_height, out_width, channel), \
                        lambda n, h, w, c: data[n, h/scale, w/scale, c])
=======
        out_shape = (simplify(data.shape[2] * scale), simplify(data.shape[3] * scale))
    elif layout == "NHWC":
        out_shape = (simplify(data.shape[1] * scale), simplify(data.shape[2] * scale))
    else:
        raise ValueError("not support this layout {} yet".format(layout))

    return topi.cpp.nn.upsampling(data, out_shape, layout, method)
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
