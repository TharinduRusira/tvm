import numpy as np
import tvm
import topi
import time
from topi.util import get_const_tuple
import math
import topi.testing
import argparse
import tvm

import logging
import sys
from tvm import autotvm

parser = argparse.ArgumentParser()
parser.add_argument("-d", nargs=1, type=str, default=["alex4"])
args = parser.parse_args()
layer = args.d[0]
_3x3_layers ={
    'vgg2_1':[64,128,64,112,112,3,1,1],'vgg3_1':[64,256,128,56,56,3,1,1], 'vgg3_2':[64,256,256,56,56,3,1,1], 'vgg4_1':[64,512,256,28,28,3,1,1],
    'vgg4_2':[64,512,512,28,28,3,1,1],'vgg5_2':[64,512,512,14,14,3,1,1],'alex3':[128,384,192,13,13,3,1,1],
    'alex4':[128,256,384,13,13,3,1,1],'alex5':[128,256,256,13,13,3,1,1],
    'overfeat3':[64,512,256,12,12,3,1,1], 'overfeat4':[64,1024,512,12,12,3,1,1], 'overfeat5':[64,1024,1024,12,12,3,1,1], 'resnet1':[1,64,64,56,56,3,1,1]}

# The sizes of inputs and filters
batch = _3x3_layers[layer][0]
in_channel = _3x3_layers[layer][2]
out_channel = _3x3_layers[layer][1]
input_height = _3x3_layers[layer][3]
input_width = _3x3_layers[layer][4]
kernel_height = _3x3_layers[layer][5]
kernel_width = _3x3_layers[layer][5]
pad_height = _3x3_layers[layer][7]
pad_width = _3x3_layers[layer][7]
stride_height = _3x3_layers[layer][6]
stride_width = _3x3_layers[layer][6]
vlen = 16
assert(pad_height == pad_width)
assert(stride_height == stride_width)
assert(kernel_height == kernel_width)
assert(in_channel%vlen == 0)
assert(out_channel%vlen == 0)
padding = pad_height
output_width = ((input_width + 2 * pad_width - kernel_width) // stride_width) + 1
output_height = ((input_height + 2 * pad_height - kernel_height) // stride_height) + 1

def convolution():
  # Algorithm
  # Algorithm
  output_width = ((input_width + 2 * pad_width - kernel_width) // stride_width) + 1
  output_height = ((input_height + 2 * pad_height - kernel_height) // stride_height) + 1

  adims = (batch, in_channel, input_height + 2*pad_height, input_width+ 2*pad_width)
  wdims = (out_channel, in_channel, kernel_height,kernel_width)
  bdims = (batch,output_height, output_width, out_channel)

  A = tvm.placeholder(adims, name='A')
  W = tvm.placeholder(wdims, name='W')
  rco = tvm.reduce_axis((0, in_channel), name='rc')
  ry = tvm.reduce_axis((0, kernel_height), name='ry')
  rx = tvm.reduce_axis((0, kernel_width), name='rx')

  # Compute the convolution
  B = tvm.compute(bdims,
        lambda n, k, h, w: tvm.sum(
            A[n, rc, h + ry, w + rx] * W[k, rc, ry, rx],
      axis=[rco,ry, rx]),
  name='B')

  s = tvm.create_schedule(func.op)
  print(type(s))
  #print(tvm.lower(s, [A, W,func], simple_mode=True))
  n,ko,h,w,ki  = s[func].op.axis
  rco,ry,rx,rci = s[func].op.reduce_axis
  cfg = autotvm.get_config()

  #get h range
  h_list = []
  for i in range(2,input_height):
    if input_height%i == 0:
      h_list.append(i)
  #get w range
  w_list = []
  for j in range(2,input_width):
    if input_width%j == 0:
      w_list.append()


  tvm.define_knob("tile_h", h_list)
  tvm.define_knob("tile_w", w_list)
  tvm.define_knob("orders", [(nko, ho, wo, rco, hi, wi, rx, ry, rci, ki), (nko, wo, ho, rco, wi, hi, rx, ry, rci, ki)])

  nko = s[func].fuse(n,ko)

  ho, hi = s[func].split(h, cfg['tile_h'].val)
  wo, wi = s[func].split(w, cfg['tile_w'].val)
  s[func].reorder(cfg['orders'].val) 

  #fixed 
  s[func].unroll(rci)

  return [s]


def compile_and_run(s, A,W,B,A1,W1):

  with tvm.build_config(data_alignment=64):
        print(tvm.lower(s, [A, W, B], simple_mode=True))

        func = tvm.build(s, [A,W,B], target='llvm -mtriple=x86_64 -mcpu=skylake-avx512 -mattr=+skx,+fma,+fma4,+avx512ifma,+avx512f,+avx512cd,+avx512bw,+avx512vl,+avx512dq')
        func.save('code.S', 'asm')

        ctx = tvm.context('llvm', 0)
        a_np = np.random.uniform(size=(batch,in_channel//vlen,input_height + 2*pad_height,input_width+ 2*pad_width,vlen)).astype(A.dtype)
        w_np = np.random.uniform(size=(out_channel//vlen,in_channel//vlen,kernel_height,kernel_width,vlen,vlen)).astype(W.dtype)

        b = tvm.nd.array(np.zeros((batch, math.ceil(out_channel/vlen),output_height, output_width,vlen), dtype=B.dtype), ctx)
        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)

        func(a, w, b)

        evaluator = func.time_evaluator(func.entry_name, ctx, number=1000)
        t = evaluator(a, w, b).mean
        gflops = np.prod(get_const_tuple(B.shape))*in_channel*kernel_height*kernel_width*2
        gflops = gflops/1e9/t
        print("Time is : {0:.6f}".format(t))
        print("GFLOPS  : {0:.3f} ".format( gflops))
  return 0, gflops
def driver():
    #tuner
    task = autotvm.task.create(convolution,target='llvm')
    print(task.config_space)

    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

    measure_option = autotvm.measure_option(builder='local', 
      runner=autotvm.LocalRunner(number=5)
      )
    tuner = autotvm.tuner.RandomTuner(task)
    tuner.tune(n_trials = 20,
      measure_option=measure_option,
      callbacks=[autotvm.callback.log_to_file(convolution.log)])

if __name__ == "__main__":
    driver()

