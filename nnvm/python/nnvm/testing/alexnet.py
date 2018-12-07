from .. import symbol as sym
from .utils import create_workload
""" Basic AlexNet workload

    adopted from https://github.com/IntelLabs/Latte.py/blob/master/benchmarks/alexnet.py

"""


def get_symbol(num_classes=1000, **kwargs):
    data = sym.Variable(name="data")

    conv1 = sym.conv2d(data=data, channels=64, kernel_size=(11,11), strides=(4,4), padding=(0,0), use_bias=True, name="conv1")
    relu1 = sym.relu(data=conv1, name="relu1")
    pool1 = sym.max_pool2d(data=relu1, pool_size=(3,3), strides=(2,2), padding=(0,0), name="pool1")

    conv2 = sym.conv2d(data=pool1, channels=192, kernel_size=(5,5), strides=(1,1), padding=(2,2), use_bias=True, name="conv2")
    relu2 = sym.relu(data=conv2, name="relu2")
    pool2 = sym.max_pool2d(data=relu2, pool_size=(3,3), strides=(2,2), padding=(0,0), name="pool2")

    conv3 = sym.conv2d(data=pool2, channels=384, kernel_size=(3,3), strides=(1,1), padding=(1,1), use_bias=True, name="conv3")
    relu3 = sym.relu(data=conv3, name="relu3")
    conv4 = sym.conv2d(data=relu3, channels=256, kernel_size=(3,3), strides=(1,1), padding=(1,1), use_bias=True, name="conv4")
    relu4 = sym.relu(data=conv4, name="relu4")
    conv5 = sym.conv2d(data=relu4, channels=256, kernel_size=(3,3), strides=(1,1), padding=(1,1), use_bias=True, name="conv5")
    relu5 = sym.relu(data=conv5, name="relu5")
    pool5 = sym.max_pool2d(data=relu4, pool_size=(3,3), strides=(2,2), padding=(0,0), name="pool5")

    flatten = sym.flatten(data=pool5, name="flatten")
    fc6bias = sym.dense(data=flatten, units=4096, name="fc6bias")
    fc7bias = sym.dense(data=fc6bias, units=4096, name="fc7bias")
    fc8bias = sym.dense(data=fc7bias, units=num_classes, name="fc8bias")
    softmax = sym.softmax(data=fc8bias, name="softmax")

    return softmax


def get_workload(batch_size=1, num_classes=1008,
                 image_shape=(3, 227, 227), dtype="float32", **kwargs):
    """Get benchmark workload for AlexNet 

    Parameters
    ----------
    batch_size : int
        The batch size used in the model

    num_classes : int, optional
        Number of classes

    image_shape : tuple, optional
        The input image shape

    dtype : str, optional
        The data type

    kwargs : dict
        Extra arguments

    Returns
    -------
    net : nnvm.Symbol
        The computational graph

    params : dict of str to NDArray
        The parameters.
    """
    net = get_symbol(num_classes=num_classes, **kwargs)
    return create_workload(net, batch_size, image_shape, dtype)
