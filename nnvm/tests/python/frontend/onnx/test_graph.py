"""Test graph equality of onnx models."""
import nnvm
import onnx
from nnvm.compiler import graph_util, graph_attr
from model_zoo import super_resolution, super_resolution_sym
<<<<<<< HEAD

def compare_graph(onnx_file, nnvm_sym, ishape):
    onnx_model = onnx.load(onnx_file)
=======
from model_zoo import squeezenet as squeezenet

def compare_graph(onnx_file, nnvm_sym, ishape):
    onnx_model = onnx.load_model(onnx_file)
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
    onnx_sym, params = nnvm.frontend.from_onnx(onnx_model)
    g1 = nnvm.graph.create(onnx_sym)
    g2 = nnvm.graph.create(nnvm_sym)
    input_name = onnx_model.graph.input[0].name
    ishapes = {input_name: ishape}
    graph_attr.set_shape_inputs(g1, ishapes)
    graph_attr.set_shape_inputs(g2, ishapes)
    g1 = g1.apply("InferShape").apply("SimplifyInference")
    g2 = g2.apply("InferShape").apply("SimplifyInference")
    graph_util.check_graph_equal(g1, g2)

def test_super_resolution_example():
<<<<<<< HEAD
    fname, symbol = super_resolution, super_resolution_sym
    compare_graph(fname, symbol, ishape=(1, 1, 224, 224))

if __name__ == '__main__':
    test_super_resolution_example()
=======
    fname, symbol = "super_resolution.onnx", super_resolution_sym
    compare_graph(fname, symbol, ishape=(1, 1, 224, 224))

def test_squeeze_net():
    # Only works for model downloaded from
    # https://github.com/onnx/models/tree/master/squeezenet
    fname = "squeezenet1_1.onnx"
    symbol, params = squeezenet.get_workload(version='1.1')
    compare_graph(fname, symbol, ishape=(1, 3, 224, 224))

if __name__ == '__main__':
    test_super_resolution_example()
    test_squeeze_net()
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
