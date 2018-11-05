import numpy as np
import tvm
import topi
import time
from topi.util import get_const_tuple
import math
import topi.testing
import argparse

import logging
import sys
from tvm import autotvm

vlen = 16

parser = argparse.ArgumentParser()
parser.add_argument("-d", nargs=1, type=str, default=["alex4"])
parser.add_argument("-debug", "--debug", dest='debug', default=False)
args = parser.parse_args()
layer = args.d[0]
_3x3_layers ={
    'vgg2_1':[64,128,64,112,112,3,1,1],'vgg3_1':[64,256,128,56,56,3,1,1], 'vgg3_2':[64,256,256,56,56,3,1,1], 'vgg4_1':[64,512,256,28,28,3,1,1],
    'vgg4_2':[64,512,512,28,28,3,1,1],'vgg5_2':[64,512,512,14,14,3,1,1],'alex3':[128,384,192,13,13,3,1,1],
    'alex4':[128,256,384,13,13,3,1,1],'alex5':[128,256,256,13,13,3,1,1],
    'overfeat3':[64,512,256,12,12,3,1,1], 'overfeat4':[64,1024,512,12,12,3,1,1], 'overfeat5':[64,1024,1024,12,12,3,1,1], 'resnet1':[1,64,64,56,56,3,1,1],  'test':[2,32,32,12,12,3,1,1]}

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

assert(in_channel%vlen == 0)
assert(out_channel%vlen == 0)
padding = pad_height

output_width = ((input_width + 2 * pad_width - kernel_width) // stride_width) + 1
output_height = ((input_height + 2 * pad_height - kernel_height) // stride_height) + 1


h_list = []
w_list = []
c_list = []

#verification with np
def forward(A, W, B):
    for n in range(0,batch):
        for ko in range(0,math.ceil(out_channel/vlen)):
            for p in range(0, output_height):
                for q in range(0,output_width):
                    for co in range(0, A.shape[1]):
                        for r in range(0,kernel_height):
                            for s in range(0,kernel_width):
                                for ci in range(0, A.shape[4]):
                                    for ki in range(0,vlen): 
                                        B[n][ko][p][q][ki] += A[n][co][p+r][q+s][ci]*W[ko][co][r][s][ci][ki]

    return B
def test():
  A = tvm.placeholder(adims, name='A')
  W = tvm.placeholder(wdims, name='W')

  rco = tvm.reduce_axis((0, math.ceil(in_channel/vlen)), name='rco')
  rci =  tvm.reduce_axis((0, vlen), name="rci")
  ry = tvm.reduce_axis((0, kernel_height), name='ry')
  rx = tvm.reduce_axis((0, kernel_width), name='rx')

  # Compute the convolution
  B = tvm.compute(bdims, lambda n, ko, h, w, ki: tvm.sum(A[n, rco, h + ry, w + rx, rci] * W[ko, rco, ry, rx, rci, ki], axis=[rco, rci ,ry, rx]), name='B')
  func = B
  s = tvm.create_schedule(func.op)
  #print(tvm.lower(s, [A, W,func], simple_mode=True))

  n, ko, h,w, ki  = s[func].op.axis
  rco, rci, ry,rx = s[func].op.reduce_axis

  ho, hi = s[func].split(h, 7)
  wo, wi = s[func].split(w, 14)

  nko = s[func].fuse(n,ko)
  s[func].reorder(nko, ho, wo, rco, hi, wi, rx, ry, rci, ki)

  #fixed 
  s[func].vectorize(ki)
  s[func].parallel(nko)
  s[func].unroll(rci)
  return s, A, W, B
def compile_and_run(s, A, W, B):

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

@autotvm.template
def convolution():

  cfg = autotvm.get_config()

  for i in range(2,in_channel):
    if in_channel%i == 0:
      c_list.append(i)

  # C tile factor
  cfg.define_knob("tile_c", c_list)
    
  #rco = tvm.reduce_axis((0, math.ceil(in_channel/vlen)), name='rco')
  #rci =  tvm.reduce_axis((0, vlen), name="rci")

  co_dim, ci_dim = math.ceil(in_channel/cfg['tile_c'].val), cfg['tile_c'].val 
  
  rco = tvm.reduce_axis((0,co_dim), name='rco')
  rci =  tvm.reduce_axis((0, ci_dim), name="rci")
 
  ry = tvm.reduce_axis((0, kernel_height), name='ry')
  rx = tvm.reduce_axis((0, kernel_width), name='rx')

  #adims = (batch, math.ceil(in_channel/vlen) , input_height + 2*pad_height, input_width+ 2*pad_width, vlen)
  #wdims = (math.ceil(out_channel/vlen), math.ceil(in_channel/vlen), kernel_height,kernel_width, vlen, vlen)
  #bdims = (batch, math.ceil(out_channel/vlen), output_height, output_width, vlen)

  adims = (batch, co_dim , input_height + 2*pad_height, input_width+ 2*pad_width, ci_dim)
  wdims = (math.ceil(out_channel/vlen), co_dim, kernel_height,kernel_width, ci_dim, vlen)
  bdims = (batch, math.ceil(out_channel/vlen), output_height, output_width, vlen)

  A = tvm.placeholder(adims, name='A', dtype='float32')
  W = tvm.placeholder(wdims, name='W', dtype='float32')


  B = tvm.compute(bdims, lambda n, ko, h, w, ki: tvm.sum(A[n, rco, h + ry, w + rx, rci] * W[ko, rco, ry, rx, rci, ki], axis=[rco, rci ,ry, rx]), name='B')
  func = B
  s = tvm.create_schedule(func.op)

  n, ko, h,w, ki  = s[func].op.axis
  rco, rci, ry,rx = s[func].op.reduce_axis
 
  for i in range(2,input_height):
    if input_height%i == 0:
      h_list.append(i)
  for j in range(2,input_width):
    if input_width%j == 0:
      w_list.append(j)

  #Tile H and W  
  cfg.define_knob("tile_h", h_list)
  cfg.define_knob("tile_w", w_list)

  ho, hi = s[func].split(h, cfg['tile_h'].val)
  wo, wi = s[func].split(w, cfg['tile_w'].val)
  
  cfg.define_knob('nk_fuse', ['fuse', 'no-fuse'])

  if cfg['nk_fuse'].val == 'fuse':
    nko = s[func].fuse(n,ko)
    s[func].reorder(nko, ho, wo, rco, hi, rx, ry, wi, rci, ki)
    s[func].parallel(nko)

  else:
    s[func].reorder(n, ko, ho, wo, rco, hi, rx, ry, wi, rci, ki)
    s[func].parallel(n)

  cfg.define_knob('ki_vectorize', ['vectorize', 'no-vectorize'])
  if cfg['ki_vectorize'].val == 'vectorize':
      s[func].vectorize(ki)
  
  cfg.define_knob('unroll_loop', ['rci', 'wi'])
  if cfg['unroll_loop'].val == 'rci':
    s[func].unroll(rci)
  else:
    s[func].unroll(wi)
  
  #TODO: tune loop orders

  return s,  [A, W, B]

def driver():
    task = autotvm.task.create(convolution, args=(), target='llvm')
    print(task.config_space)

    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

    measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(), runner=autotvm.LocalRunner(number=5, timeout=100))
    tuner = autotvm.tuner.RandomTuner(task)
    tuner.tune(n_trial=10, measure_option=measure_option, callbacks=[autotvm.callback.log_to_file("tune.log")])

def verify_history_best():
    with autotvm.apply_history_best('tune.log'):
        with tvm.target.create('llvm -mtriple=x86_64 -mcpu=skylake-avx512 -mattr=+skx,+fma,+fma4,+avx512ifma,+avx512f,+avx512cd,+avx512bw,+avx512vl,+avx512dq'):
            s, bufs = convolution()
            print(bufs)
            func = tvm.build(s, bufs[:3])

    adims, wdims, bdims = bufs[0].shape, bufs[1].shape, bufs[2].shape
    #print(adims, type(adims), type(adims[0]))
    Adims, Wdims, Bdims = [], [], []
    for i in adims:
        Adims.append(int(i))
    for j in wdims:
        Wdims.append(int(j))
    for k in bdims:
       Bdims.append(int(k))
 
    a1 = np.random.uniform(size=tuple(Adims)).astype(np.float32)
    w1 = np.random.uniform(size=tuple(Wdims)).astype(np.float32)
    b_ref = forward(a1, w1, np.zeros(tuple(Bdims), dtype=np.float32))

    b1 = tvm.nd.empty(b_ref.shape)
    func(tvm.nd.array(a1), tvm.nd.array(w1), b1)

    np.testing.assert_allclose(b1.asnumpy(), b_ref, rtol=1e-5)

if __name__ == "__main__":
    if args.debug:
        print("running debug mode...")
        s, A, W, B= test()
        compile_and_run(s, A, W, B)
    else:
        driver()
        verify_history_best()
