"""Tensor operator property registry

Provide information to lower and schedule tensor operators.
"""
from .attr_dict import AttrDict
from . import tensor
from . import nn
from . import transform
from . import reduction
from . import vision
<<<<<<< HEAD
=======
from . import image
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

from .registry import OpPattern
from .registry import register_compute, register_schedule, register_pattern
