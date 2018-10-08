from __future__ import absolute_import, print_function

import tvm
import numpy as np
from tvm.contrib import cblas

def bp(A,W,B):
    for n in range(0,batch):
        for k in range(0, out_channel) :
            for h in range(0, input_height):
                for w in range(0,input_width):
                    for c in range(0, in_channel):
                        for r in range(0,kernel_height):
                            for s in range(0,kernel_width):
     
                                #B[n][h][w][r][s][c] += A[n][h][w][k]*W[k][c][r][s]
                                B[n][h+r][w+s][c] += A[n][h][w][k]*W[k][r][s][c]

    return B


layer = 'l'

_3x3_layers = {'l':[2,8,16,4,4,3,1,1]}

batch = _3x3_layers[layer][0]           #N
in_channel = _3x3_layers[layer][2]      #C
out_channel = _3x3_layers[layer][1]     #K
input_height = _3x3_layers[layer][3]    #H
input_width = _3x3_layers[layer][4]     #W
kernel_height = _3x3_layers[layer][5]   #R
kernel_width = _3x3_layers[layer][5]    #S
pad_height = _3x3_layers[layer][7]      
pad_width = _3x3_layers[layer][7]       
stride_height = _3x3_layers[layer][6]   #u
stride_width = _3x3_layers[layer][6]    #v
padding = pad_height
output_width = ((input_width + 2 * pad_width - kernel_width) // stride_width) + 1       #Q
output_height = ((input_height + 2 * pad_height - kernel_height) // stride_height) + 1  #P

adims = (batch, input_height+kernel_height-1, input_width+kernel_width-1, out_channel)
wdims = (out_channel, kernel_height, kernel_width, in_channel)
bdims1 = (batch, input_height, input_width, kernel_height, kernel_width, in_channel)
bdims = (batch, input_height, input_width , in_channel)

rk = tvm.reduce_axis((0,out_channel), name='rk')

A = tvm.placeholder(adims,name='A')
W = tvm.placeholder(wdims,name='W')

B = tvm.compute(bdims, lambda n, h, w, c: tvm.sum(A[n][h][w][rk]*W[rk][r][s][c], axis=[rk,r,s]), name='B')

s = tvm.create_schedule(B.op)
print(tvm.lower(s, [A,W,B], simple_mode=True))

##############################################

func = tvm.build(s, [A,W,B], target='llvm -mtriple=x86_64 -mcpu=skylake-avx512 -mattr=+skx,+fma,+fma4,+avx512ifma,+avx512f,+avx512cd,+avx512bw,+avx512vl,+avx512dq')
ctx = tvm.context('llvm',0)

grad = tvm.nd.array(np.random.uniform(size=adims).astype(A.dtype), ctx)
weight = tvm.nd.array(np.random.uniform(size=wdims).astype(W.dtype), ctx)
grad_input = tvm.nd.array(np.zeros(bdims, dtype=B.dtype), ctx)

func(grad, weight, grad_input)
bp_ref = bp(grad.asnumpy(), weight.asnumpy(), np.zeros(bdims, dtype=grad_input.dtype))

print(grad_input.shape)
print(bp_ref.shape)

np.testing.assert_allclose(grad_input.asnumpy(), bp_ref, rtol=1e-5)


