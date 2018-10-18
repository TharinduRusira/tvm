#!/bin/bash

export PYTHONPATH=nnvm/python:python:topi/python
<<<<<<< HEAD
=======
# to avoid openblas threading error
export OMP_NUM_THREADS=1

# Rebuild cython
make cython || exit -1
make cython3 || exit -1
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

echo "Running unittest..."
python -m nose -v nnvm/tests/python/unittest || exit -1
python3 -m nose -v nnvm/tests/python/unittest || exit -1

echo "Running compiler test..."
python -m nose -v nnvm/tests/python/compiler || exit -1
python3 -m nose -v nnvm/tests/python/compiler || exit -1

echo "Running ONNX frontend test..."
<<<<<<< HEAD
python -m nose -v nnvm/tests/python/frontend/onnx || exit -1

echo "Running MXNet frontend test..."
python -m nose -v nnvm/tests/python/frontend/mxnet || exit -1

echo "Running Keras frontend test..."
python -m nose -v nnvm/tests/python/frontend/keras || exit -1
=======
python3 -m nose -v nnvm/tests/python/frontend/onnx || exit -1

echo "Running MXNet frontend test..."
python3 -m nose -v nnvm/tests/python/frontend/mxnet || exit -1

echo "Running Keras frontend test..."
python3 -m nose -v nnvm/tests/python/frontend/keras || exit -1

echo "Running Tensorflow frontend test..."
python3 -m nose -v nnvm/tests/python/frontend/tensorflow || exit -1
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
