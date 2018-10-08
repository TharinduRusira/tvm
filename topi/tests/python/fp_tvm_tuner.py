import numpy as np
import tvm
import topi
import time
from topi.util import get_const_tuple
import math
import topi.testing
import wc as mdl 
import xlwt
import argparse

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

     
'''
Workload('float32', 'float32', 56, 56, 64, 64, 1, 1, 0, 0, 1, 1),
Workload('float32', 'float32', 56, 56, 64, 128, 3, 3, 1, 1, 2, 2),
Workload('float32', 'float32', 56, 56, 64, 128, 1, 1, 0, 0, 2, 2),
Workload('float32', 'float32', 28, 28, 128, 128, 3, 3, 1, 1, 1, 1),
Workload('float32', 'float32', 28, 28, 128, 256, 3, 3, 1, 1, 2, 2),
Workload('float32', 'float32', 28, 28, 128, 256, 1, 1, 0, 0, 2, 2),
Workload('float32', 'float32', 14, 14, 256, 256, 3, 3, 1, 1, 1, 1),
Workload('float32', 'float32', 14, 14, 256, 512, 3, 3, 1, 1, 2, 2),
Workload('float32', 'float32', 14, 14, 256, 512, 1, 1, 0, 0, 2, 2),
Workload('float32', 'float32', 7, 7, 512, 512, 3, 3, 1, 1, 1, 1),
# workloads of resnet34_v1 on imagenet, no extra workload required
# workloads of resnet50_v1 on imagenet
Workload('float32', 'float32', 56, 56, 64, 256, 1, 1, 0, 0, 1, 1),
Workload('float32', 'float32', 56, 56, 256, 64, 1, 1, 0, 0, 1, 1),
Workload('float32', 'float32', 56, 56, 256, 128, 1, 1, 0, 0, 2, 2),
Workload('float32', 'float32', 28, 28, 128, 512, 1, 1, 0, 0, 1, 1),
Workload('float32', 'float32', 56, 56, 256, 512, 1, 1, 0, 0, 2, 2),
Workload('float32', 'float32', 28, 28, 512, 128, 1, 1, 0, 0, 1, 1),
Workload('float32', 'float32', 28, 28, 512, 256, 1, 1, 0, 0, 2, 2),
Workload('float32', 'float32', 14, 14, 256, 1024, 1, 1, 0, 0, 1, 1),
Workload('float32', 'float32', 28, 28, 512, 1024, 1, 1, 0, 0, 2, 2),
Workload('float32', 'float32', 14, 14, 1024, 256, 1, 1, 0, 0, 1, 1),
Workload('float32', 'float32', 14, 14, 1024, 512, 1, 1, 0, 0, 2, 2),
Workload('float32', 'float32', 7, 7, 512, 2048, 1, 1, 0, 0, 1, 1),
Workload('float32', 'float32', 14, 14, 1024, 2048, 1, 1, 0, 0, 2, 2),
Workload('float32', 'float32', 7, 7, 2048, 512, 1, 1, 0, 0, 1, 1),
'''


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

def generate_variants(func, tf):
  s = tvm.create_schedule(func.op)
  print(type(s))
  #print(tvm.lower(s, [A, W,func], simple_mode=True))
  n,ko,h,w,ki  = s[func].op.axis
  rco,ry,rx,rci = s[func].op.reduce_axis


  wo = None
  wi = None
  wi_o = None
  wi_i = None
  hi_o = None
  hi_i = None
  ho = None
  hi = None
  rco_o = None
  rco_i = None
  w_tile = None
  h_tile = None


  if(tf['W'] < output_width):
      wo, wi= s[func].split(w, factor=tf['W'])
  else:
      wo = None
      wi = w
  if(tf['H'] < output_height):  
      ho, hi= s[func].split(h, factor=tf['H'])     
  else:
      ho = None
      hi = h
  if tf['CC']//vlen < in_channel//vlen:
      rco_o, rco_i= s[func].split(rco, factor=tf['CC']//vlen) # might cause problem
  else:
      rco_o = None
      rco_i = rco

  #rci_o, rci_i = s[func].split(rci, factor=2)
  rci_o = None
  rci_i = rci   

  

  
  w_threshold = min(tf['W'],14)
 
     

 
  while tf['W'] % w_threshold != 0:
     w_threshold = w_threshold-1


  h_threshold  = min(tf['H'], math.floor(28/w_threshold))
 
  while tf['H'] % h_threshold != 0:
     h_threshold = h_threshold-1



  
  if(tf['W'] < output_width):
    if w_threshold < tf['W'] and w_threshold > 1:
        wi_o, wi_i = s[func].split(wi, factor=w_threshold)
        w_unroll = wi_i
        #w_tile = wi_o
    elif w_threshold == 1:
        w_unroll  = None
        
        wi_o = wi
        wi_i = None
        #w = None
    elif w_threshold == tf['W']:
        wi_i = wi
        w_unroll = wi_i
        wi_o = None
        #wi = None
    
  elif w_threshold > 1 and w_threshold < tf['W']:
    wi_o, wi_i = s[func].split(w, factor=w_threshold)  
    w_unroll = wi_i
    #w_tile = wo

  elif w_threshold == 1:
    w_unroll  = None
    wi_o = w
    wi_i = None
    w_o = None
  elif tf['W'] == w_threshold and tf['W'] == output_width:
    
    w_o = None
    wi_o = None
    wi_i = w        
    w_unroll = wi_i


  if(tf['H'] < output_height):
    if h_threshold < tf['H'] and h_threshold > 1:
        hi_o, hi_i = s[func].split(hi, factor=h_threshold)
        h_unroll = hi_i
        #h_tile = hi_o
    elif h_threshold == 1:
        h_unroll  = None
        hi_o = hi
        hi_i = None
        #w = None
    elif h_threshold == tf['H']:
        hi_i = hi
        h_unroll = hi_i
        hi_o = None
        #wi = None
 
  elif h_threshold > 1 and h_threshold < tf['H']:
    hi_o, hi_i = s[func].split(h, factor=h_threshold)
    h_unroll = hi_i
    #w_tile = wo
 
  elif h_threshold == 1:
    h_unroll  = None
    hi_o = hi
    hi_i = None
    h_o = None
  elif tf['H'] == h_threshold and tf['H'] == output_height:
     
    h_o = None
    hi_o = None
    hi_i = h
    h_unroll = hi_i

  
  order = [n,ko]
  for i in [rco_o,ho,wo,rco_i,hi_o,wi_o,ry,rx,rci_o,rci_i,hi_i,wi_i,ki]:
     if i != None:
         order.append(i) 
      
  s[func].reorder(*order)   
  s[func].vectorize(ki)
  par = s[func].fuse(n, ko)
  s[func].parallel(par)
  if w_unroll != None:
      s[func].unroll(w_unroll)
  if h_unroll != None:
      s[func].unroll(h_unroll)
  #s[func].unroll(rci_i)
  

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
    # Algorithm
    #output_width = ((input_width + 2 * pad_width - kernel_width) // stride_width) + 1
    #output_height = ((input_height + 2 * pad_height - kernel_height) // stride_height) + 1
    
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

    
    indir = '/homes/tharindu/tvm/topi/tests/python'
    book = xlwt.Workbook(encoding="utf-8")
    sheet1 = book.add_sheet("Sheet 1")
    row1=0
    row2=0
    sheet1.write(0,0,"Layer")
    sheet1.write(0,1,"Rank")
    sheet1.write(0,2,"FLOPS")
    sheet1.write(0,3,"TVM_FLOPS")
    sheet1.write(0,4,"Cost")
    sheet1.write(0,5,"Factors")
    sheet1.write(0,6,"order")
    sheet1.write(0,7,"total")
    sheet1.write(0,8,"input")
    sheet1.write(0,9,"output")
    sheet1.write(0,10,"weight")
    sheet1.write(0,11,"n")
    sheet1.write(0,12,"k")
    sheet1.write(0,13,"h")
    sheet1.write(0,14,"w")
    sheet1.write(0,15,"c")
    row1= row1 + 1
    
    v = [batch,out_channel,in_channel,int(output_height),int(output_width), kernel_height, stride_height, pad_height]
    lb = mdl.init(v)
    conv_configs = mdl.tile_and_footprint_analysis(lb, search_harder=False, output_volume_multiplier=1)
    
    which_loads = {'output':2, 'input':4, 'weight':3, 'total':5}
    idx = which_loads['total']
    loads_ = sorted(conv_configs, key=lambda x: x[idx])
    total_fp_ = loads_[0][5]
    

    no_dups = [0]
    for i in range(len(loads_)):
      if loads_[i][5] == total_fp_:
          continue
      else:
          no_dups.append(i)
          total_fp_ = loads_[i][5]
 
    no_dups.append(len(loads_))
 
    counter=1
 
    for i in range(min(1,len(no_dups)-1)):
        best_flops = -1
        best_flops_tvm = -1
        for config in range(no_dups[i+1] - no_dups[i]):
            variants =  generate_variants(B, loads_[no_dups[i] + config][1] )
            skip = False
            for net in variants:
 
                fflops_tvm, fflops = compile_and_run(net,A,W,B,A1,W1)
                
                if fflops_tvm > best_flops_tvm:
                  best_flops_tvm = fflops_tvm
                if fflops > best_flops:
                  best_flops = fflops
                  best_cost = loads_[no_dups[i] + config][which_loads['total']]
                  input_cost = loads_[no_dups[i] + config][which_loads['input']]
                  output_cost = loads_[no_dups[i] + config][which_loads['output']]
                  weight_cost = loads_[no_dups[i] + config][which_loads['weight']]
                  t = loads_[no_dups[i] + config][6]
                  n1 = loads_[no_dups[i] + config][7]
                  k = loads_[no_dups[i] + config][8]
                  h = loads_[no_dups[i] + config][9]
                  w = loads_[no_dups[i] + config][10]
                  c = loads_[no_dups[i] + config][11]
                  order = str(loads_[no_dups[i] + config][0])
                  params = str(loads_[no_dups[i] + config][1])
                  break

            sheet1.write(row1,0,layer)
            sheet1.write(row1,1,counter)
            sheet1.write(row1,2,best_flops)
            sheet1.write(row1,3,best_flops_tvm)
            sheet1.write(row1,4,best_cost)
            sheet1.write(row1,5,params)
 
            sheet1.write(row1,6,order)
            sheet1.write(row1,7,t)
            sheet1.write(row1,8,input_cost)
            sheet1.write(row1,9,output_cost)
            sheet1.write(row1,10,weight_cost)
            sheet1.write(row1,11, n1)
 
            sheet1.write(row1,12,k)
            sheet1.write(row1,13,h)
            sheet1.write(row1,14,w)
            sheet1.write(row1,15,str(c))
            row1 = row1 + 1
            counter = counter + 1
            book.save( layer +".xls")
 
if __name__ == "__main__":
    driver()

