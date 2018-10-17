TOPI
----
.. automodule:: topi

List of operators
~~~~~~~~~~~~~~~~~

.. autosummary::

   topi.identity
   topi.negative
<<<<<<< HEAD
=======
   topi.floor
   topi.ceil
   topi.trunc
   topi.round
   topi.abs
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
   topi.exp
   topi.tanh
   topi.log
   topi.sqrt
   topi.sigmoid
   topi.clip
   topi.cast
   topi.transpose
   topi.flip
   topi.strided_slice
   topi.expand_dims
   topi.reshape
   topi.squeeze
   topi.concatenate
   topi.split
   topi.take
   topi.full
   topi.full_like
<<<<<<< HEAD
   topi.greater
   topi.less
=======
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
   topi.nn.relu
   topi.nn.leaky_relu
   topi.nn.dilate
   topi.nn.pool
   topi.nn.global_pool
   topi.nn.upsampling
   topi.nn.softmax
   topi.nn.log_softmax
   topi.nn.conv2d_nchw
   topi.nn.conv2d_hwcn
   topi.nn.depthwise_conv2d_nchw
   topi.nn.depthwise_conv2d_nhwc
   topi.max
   topi.sum
   topi.min
   topi.argmax
   topi.argmin
   topi.prod
   topi.broadcast_to
<<<<<<< HEAD
   topi.broadcast_add
   topi.broadcast_sub
   topi.broadcast_mul
   topi.broadcast_div
   topi.broadcast_maximum
   topi.broadcast_minimum


=======
   topi.add
   topi.subtract
   topi.multiply
   topi.divide
   topi.mod
   topi.maximum
   topi.minimum
   topi.power
   topi.greater
   topi.less
   topi.equal
   topi.not_equal
   topi.greater_equal
   topi.less_equal
   topi.image.resize


>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
List of schedules
~~~~~~~~~~~~~~~~~
.. autosummary::

   topi.generic.schedule_conv2d_nchw
   topi.generic.schedule_depthwise_conv2d_nchw
   topi.generic.schedule_reduce
   topi.generic.schedule_broadcast
   topi.generic.schedule_injective

topi
~~~~
.. autofunction:: topi.negative
.. autofunction:: topi.identity
<<<<<<< HEAD
=======
.. autofunction:: topi.floor
.. autofunction:: topi.ceil
.. autofunction:: topi.trunc
.. autofunction:: topi.round
.. autofunction:: topi.abs
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
.. autofunction:: topi.exp
.. autofunction:: topi.tanh
.. autofunction:: topi.log
.. autofunction:: topi.sqrt
.. autofunction:: topi.sigmoid
.. autofunction:: topi.clip
.. autofunction:: topi.cast
.. autofunction:: topi.transpose
.. autofunction:: topi.flip
.. autofunction:: topi.strided_slice
.. autofunction:: topi.expand_dims
.. autofunction:: topi.reshape
.. autofunction:: topi.squeeze
.. autofunction:: topi.concatenate
.. autofunction:: topi.split
.. autofunction:: topi.take
.. autofunction:: topi.full
.. autofunction:: topi.full_like
<<<<<<< HEAD
.. autofunction:: topi.greater
.. autofunction:: topi.less
=======
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
.. autofunction:: topi.max
.. autofunction:: topi.sum
.. autofunction:: topi.min
.. autofunction:: topi.prod
.. autofunction:: topi.broadcast_to
<<<<<<< HEAD
.. autofunction:: topi.broadcast_add
.. autofunction:: topi.broadcast_sub
.. autofunction:: topi.broadcast_mul
.. autofunction:: topi.broadcast_div
.. autofunction:: topi.broadcast_maximum
.. autofunction:: topi.broadcast_minimum

=======
.. autofunction:: topi.add
.. autofunction:: topi.subtract
.. autofunction:: topi.multiply
.. autofunction:: topi.divide
.. autofunction:: topi.mod
.. autofunction:: topi.maximum
.. autofunction:: topi.minimum
.. autofunction:: topi.power
.. autofunction:: topi.greater
.. autofunction:: topi.less
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

topi.nn
~~~~~~~
.. autofunction:: topi.nn.relu
.. autofunction:: topi.nn.leaky_relu
.. autofunction:: topi.nn.dilate
.. autofunction:: topi.nn.pool
.. autofunction:: topi.nn.global_pool
.. autofunction:: topi.nn.upsampling
.. autofunction:: topi.nn.softmax
.. autofunction:: topi.nn.log_softmax
.. autofunction:: topi.nn.conv2d_nchw
.. autofunction:: topi.nn.conv2d_hwcn
.. autofunction:: topi.nn.depthwise_conv2d_nchw
.. autofunction:: topi.nn.depthwise_conv2d_nhwc

topi.image
~~~~~~~~~~
.. autofunction:: topi.image.resize


topi.generic
~~~~~~~~~~~~
.. automodule:: topi.generic

.. autofunction:: topi.generic.schedule_conv2d_nchw
.. autofunction:: topi.generic.schedule_depthwise_conv2d_nchw
.. autofunction:: topi.generic.schedule_reduce
.. autofunction:: topi.generic.schedule_broadcast
.. autofunction:: topi.generic.schedule_injective
