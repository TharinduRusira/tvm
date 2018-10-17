"""
<<<<<<< HEAD
Deploy the Pretrained Model on ARM Mali GPU
=======================================================
**Author**: `Lianmin Zheng <https://lmzheng.net/>`_, `Ziheng Jiang <https://ziheng.org/>`_

This is an example of using NNVM to compile a ResNet model and
deploy it on Firefly-RK3399 with ARM Mali GPU.  We will use the
Mali-T860 MP4 GPU on this board to accelerate the inference.

This tutorial is based on the tutorial for deploying on Raspberry Pi by `Ziheng Jiang <https://ziheng.org/>`_.
Great thanks to the original author, I only do several lines of modification.

To begin with, we import nnvm (for compilation) and TVM (for deployment).
"""
import tvm
import nnvm.compiler
import nnvm.testing
from tvm.contrib import util, rpc
from tvm.contrib import graph_runtime as runtime

=======
.. _tutorial-deploy-model-on-mali-gpu:

Deploy the Pretrained Model on ARM Mali GPU
===========================================
**Author**: `Lianmin Zheng <https://lmzheng.net/>`_, `Ziheng Jiang <https://ziheng.org/>`_

This is an example of using NNVM to compile a ResNet model and
deploy it on Firefly-RK3399 with ARM Mali GPU. We will use the
Mali-T860 MP4 GPU on this board to accelerate the inference.
"""

import tvm
import nnvm.compiler
import nnvm.testing
from tvm import rpc
from tvm.contrib import util, graph_runtime as runtime
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

######################################################################
# Build TVM Runtime on Device
# ---------------------------
#
<<<<<<< HEAD
# There're some prerequisites: we need build tvm runtime and set up
# a RPC server on remote device.
#
# To get started, clone tvm repo from github. It is important to clone
# the submodules along, with --recursive option (Assuming you are in
# your home directory):
#
#   .. code-block:: bash
#
#     git clone --recursive https://github.com/dmlc/tvm
#
# .. note::
#
#   Usually device has limited resources and we only need to build
#   runtime. The idea is we will use TVM compiler on the local server
#   to compile and upload the compiled program to the device and run
#   the device function remotely.
#
#   .. code-block:: bash
#
#     make runtime
#
# After success of buildind runtime, we need set environment varibles
# in :code:`~/.bashrc` file of yourself account or :code:`/etc/profile`
# of system enviroment variables. Assuming your TVM directory is in
# :code:`~/tvm` and set environment variables below your account.
#
#   .. code-block:: bash
#
#    vi ~/.bashrc
#
# We need edit :code:`~/.bashrc` using :code:`vi ~/.bashrc` and add
# lines below (Assuming your TVM directory is in :code:`~/tvm`):
#
#   .. code-block:: bash
#
#    export TVM_HOME=~/tvm
#    export PATH=$PATH:$TVM_HOME/lib
#    export PYTHONPATH=$PYTHONPATH:$TVM_HOME/python
#
# To enable updated :code:`~/.bashrc`, execute :code:`source ~/.bashrc`.
=======
# The first step is to build tvm runtime on the remote device.
#
# .. note::
#
#   All instructions in both this section and next section should be
#   executed on the target device, e.g. Rk3399. And we assume it
#   has Linux running.
# 
# Since we do compilation on local machine, the remote device is only used
# for running the generated code. We only need to build tvm runtime on
# the remote device. Make sure you have opencl driver in your board.
# You can refer to `tutorial <https://gist.github.com/mli/585aed2cec0b5178b1a510f9f236afa2>`_
# to setup OS and opencl driver for rk3399.
#
# .. code-block:: bash
#
#   git clone --recursive https://github.com/dmlc/tvm
#   cd tvm
#   cp cmake/config.cmake .
#   sed -i "s/USE_OPENCL OFF/USE_OPENCL ON/" config.cmake 
#   make runtime -j4
#
# After building runtime successfully, we need to set environment varibles
# in :code:`~/.bashrc` file. We can edit :code:`~/.bashrc`
# using :code:`vi ~/.bashrc` and add the line below (Assuming your TVM 
# directory is in :code:`~/tvm`):
#
# .. code-block:: bash
#
#   export PYTHONPATH=$PYTHONPATH:~/tvm/python
#
# To update the environment variables, execute :code:`source ~/.bashrc`.
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

######################################################################
# Set Up RPC Server on Device
# ---------------------------
<<<<<<< HEAD
# To set up a TVM RPC server on the your ARM device (our remote device),
# we have prepared a one-line script so you only need to run this
# command after following the installation guide to install TVM on
# your device:
=======
# To start an RPC server, run the following command on your remote device
# (Which is RK3399 in our example).
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
#
#   .. code-block:: bash
#
#     python -m tvm.exec.rpc_server --host 0.0.0.0 --port=9090
#
<<<<<<< HEAD
# After executing command above, if you see these lines below, it's
# successful to start RPC server on your device.
#
#    .. code-block:: bash
#
#      Loading runtime library /home/YOURNAME/code/tvm/lib/libtvm_runtime.so... exec only
=======
# If you see the line below, it means the RPC server started
# successfully on your device.
#
#    .. code-block:: bash
#
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
#      INFO:root:RPCServer: bind to 0.0.0.0:9090
#

######################################################################
<<<<<<< HEAD
# For demonstration, we simply start an RPC server on the same machine,
# if :code:`use_mali` is False. If you have set up the remote
# environment, please change the three lines below: change the
# :code:`use_mali` to True, also change the :code:`host` and :code:`port`
# with your device's host address and port number.

use_mali = False
host = '10.42.0.96'
port = 9090

if not use_mali:
    # run server locally
    host = 'localhost'
    port = 9095
    server = rpc.Server(host=host, port=port, use_popen=True)

######################################################################
# Prepare the Pretrained Model
# ----------------------------
# Back to the host machine, firstly, we need to download a MXNet Gluon
# ResNet model from model zoo, which is pretrained on ImageNet. You
# can found more details about this part at `Compile MXNet Models`
=======
# Prepare the Pre-trained Model
# -----------------------------
# Back to the host machine, which should have a full TVM installed (with LLVM).
# 
# We will use pre-trained model from
# `MXNet Gluon model zoo <https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html>`_.
# You can found more details about this part at tutorial :ref:`tutorial-from-mxnet`.
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

from mxnet.gluon.model_zoo.vision import get_model
from mxnet.gluon.utils import download
from PIL import Image
import numpy as np

# only one line to get the model
block = get_model('resnet18_v1', pretrained=True)

######################################################################
# In order to test our model, here we download an image of cat and
# transform its format.
<<<<<<< HEAD
img_name = 'cat.jpg'
=======
img_name = 'cat.png'
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
download('https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true', img_name)
image = Image.open(img_name).resize((224, 224))

def transform_image(image):
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

x = transform_image(image)

<<<<<<< HEAD

=======
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
######################################################################
# synset is used to transform the label from number of ImageNet class to
# the word human can understand.
synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                      '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                      '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                      'imagenet1000_clsid_to_human.txt'])
<<<<<<< HEAD
=======

>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
synset_name = 'synset.txt'
download(synset_url, synset_name)
with open(synset_name) as f:
    synset = eval(f.read())

######################################################################
# Now we would like to port the Gluon model to a portable computational graph.
# It's as easy as several lines.

# We support MXNet static graph(symbol) and HybridBlock in mxnet.gluon
net, params = nnvm.frontend.from_mxnet(block)
# we want a probability so add a softmax operator
net = nnvm.sym.softmax(net)

######################################################################
# Here are some basic data workload configurations.
batch_size = 1
num_classes = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
<<<<<<< HEAD
out_shape = (batch_size, num_classes)
=======
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

######################################################################
# Compile The Graph
# -----------------
# To compile the graph, we call the :any:`nnvm.compiler.build` function
# with the graph configuration and parameters. As we use OpenCL for
# GPU computing, the tvm will generate both OpenCL kernel code and ARM
# CPU host code. The CPU host code is used for calling OpenCL kernels.
# In order to generate correct CPU code, we need to specify the target
# triplet for host ARM device by setting the parameter :code:`target_host`.

######################################################################
<<<<<<< HEAD
# If we run the example locally for demonstration, we can simply set
# it as :code:`llvm`. If to run it on the ARM device, you need to specify
# its instruction set. Here is the option I use for my Firefly-RK3399.

if use_mali:
    target_host = "llvm -target=aarch64-linux-gnu -mattr=+neon"
    target = tvm.target.mali()
else:
    target_host = "llvm"
    target = tvm.target.cuda()

# set target as  `tvm.target.mali` instead of 'opencl' to enable
# target-specified optimization
graph, lib, params = nnvm.compiler.build(net, target=target,
        shape={"data": data_shape}, params=params, target_host=target_host)
=======
# If we run the example on our x86 server for demonstration, we can simply
# set it as :code:`llvm`. If running it on the RK3399, we need to
# specify its instruction set. Set :code:`local_demo` to False if you
# want to run this tutorial with a real device.

local_demo = True

if local_demo:
    target_host = "llvm"
    target = "llvm"
else:
    # Here is the setting for my rk3399 board
    # If you don't use rk3399, you can query your target triple by 
    # execute `gcc -v` on your board.
    target_host = "llvm -target=aarch64-linux-gnu"

    # set target as  `tvm.target.mali` instead of 'opencl' to enable
    # optimization for mali
    target = tvm.target.mali()

with nnvm.compiler.build_config(opt_level=3):
    graph, lib, params = nnvm.compiler.build(net, target=target,
            shape={"data": data_shape}, params=params, target_host=target_host)
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

# After `nnvm.compiler.build`, you will get three return values: graph,
# library and the new parameter, since we do some optimization that will
# change the parameters but keep the result of model as the same.

# Save the library at local temporary directory.
tmp = util.tempdir()
lib_fname = tmp.relpath('net.tar')
lib.export_library(lib_fname)

######################################################################
# Deploy the Model Remotely by RPC
# --------------------------------
# With RPC, you can deploy the model remotely from your host machine
# to the remote device.

<<<<<<< HEAD
# connect the server
remote = rpc.connect(host, port)
=======
# obtain an RPC session from remote device.
if local_demo:
    remote = rpc.LocalSession()
else:
    # The following is my environment, change this to the IP address of your target device
    host = '10.77.1.145'
    port = 9090
    remote = rpc.connect(host, port)
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

# upload the library to remote device and load it
remote.upload(lib_fname)
rlib = remote.load_module('net.tar')

<<<<<<< HEAD
ctx = remote.cl(0) if use_mali else remote.gpu(0)
# upload the parameter
rparams = {k: tvm.nd.array(v, ctx) for k, v in params.items()}

# create the remote runtime module
module = runtime.create(graph, rlib, ctx)
# set parameter
module.set_input(**rparams)
=======
# create the remote runtime module
ctx = remote.cl(0) if not local_demo else remote.cpu(0)
module = runtime.create(graph, rlib, ctx)
# set parameter (upload params to the remote device. This may take a while)
module.set_input(**params)
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
# set input data
module.set_input('data', tvm.nd.array(x.astype('float32')))
# run
module.run()
# get output
<<<<<<< HEAD
out = module.get_output(0, tvm.nd.empty(out_shape, ctx=ctx))
# get top1 result
top1 = np.argmax(out.asnumpy())
print('TVM prediction top-1: {}'.format(synset[top1]))

if not use_mali:
    # terminate the local server
    server.terminate()
=======
out = module.get_output(0)
# get top1 result
top1 = np.argmax(out.asnumpy())
print('TVM prediction top-1: {}'.format(synset[top1]))
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
