import tvm
import numpy as np

#for verification
def wu1(A, W, B):
    for n in range(0,batch):
        for k in range(0,out_channel):
            for h in range(0, output_height):
                for w in range(0,output_width):
                    for c in range(0, in_channel):
                        for r in range(0,kernel_height):
                            for s in range(0,kernel_width):
                                B[k][c][r][s] += A[n][k][h][w]*W[n][c][h+r][w+s]

    return B
 

parser = argparse.ArgumentParser()
parser.add_argument("-d", nargs=1, type=str, default=["alex4"])
args = parser.parse_args()
layer = args.d[0]
_3x3_layers ={
    'vgg2_1':[64,128,64,112,112,3,1,1],'vgg3_1':[64,256,128,56,56,3,1,1], 'vgg3_2':[64,256,256,56,56,3,1,1], 'vgg4_1':[64,512,256,28,28,3,1,1],
    'vgg4_2':[64,512,512,28,28,3,1,1],'vgg5_2':[64,512,512,14,14,3,1,1],'alex3':[128,384,192,13,13,3,1,1],
    'alex4':[128,256,384,13,13,3,1,1],'alex5':[128,256,256,13,13,3,1,1],
    'overfeat3':[64,512,256,12,12,3,1,1], 'overfeat4':[64,1024,512,12,12,3,1,1], 'overfeat5':[64,1024,1024,12,12,3,1,1], 'resnet1':[1,64,64,56,56,3,1,1], 'test':[2, 32,32,12,12,3,1,1]}


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

#original dimensions
input_dims1 = (batch, in_channel, input_height + 2*pad_height , input_width + 2*pad_width)    #FP input dims
grad_dims1 = (batch, out_channel, output_height, output_width) #FP output dims
grad_weight_dims1 = (out_channel, in_channel, kernel_height, kernel_width)    #FP weight dims

grad1 = tvm.placeholder(grad_dims1, name='grad1')
inputs1 = tvm.placeholder(input_dims1 , name='inputs1')

#reduction axes
rn = tvm.reduce_axis((0,batch), name='n')
rh = tvm.reduce_axis((0, input_height), name='rh')
rw = tvm.reduce_axis((0, input_width), name='rw')

#compute weight update

grad_weight1 = tvm.compute(grad_weight_dims1, 
                        lambda k, c, rr, ss: tvm.sum(
                           grad1[rn, k, rh, rw] * inputs1[rn, c, rh+rr, rw+ss], 
                            axis=[rn,rh, rw]), 
                        name = 'grad_weight1')


s = tvm.create_schedule(grad_weight1.op)

func = tvm.build(s, [grad1, inputs1, grad_weight1], target = 'llvm -mtriple=x86_64 -mcpu=skylake-avx512 -mattr=+skx,+fma,+fma4,+avx512ifma,+avx512f,+avx512cd,+avx512bw,+avx512vl,+avx512dq')

ctx = tvm.context('llvm',0)
   
a1 = tvm.nd.array(np.random.uniform(size=grad_dims1).astype(grad1.dtype), ctx)
w1 = tvm.nd.array(np.random.uniform(size=input_dims1).astype(inputs1.dtype), ctx)
b1 = tvm.nd.array(np.zeros(grad_weight_dims1, dtype=grad_weight1.dtype), ctx)

func(a1, w1, b1)

#call python function to get the reference for comparison
b_ref = wu1(a1.asnumpy(), w1.asnumpy(), np.zeros(grad_weight_dims1, dtype=grad_weight1.dtype))

np.testing.assert_allclose(b1.asnumpy(), b_ref, rtol=1e-5)

