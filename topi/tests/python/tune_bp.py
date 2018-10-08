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

parser = argparse.ArgumentParser()
parser.add_argument("-d", nargs=1, type=str, default=["alex4"])
args = parser.parse_args()
layer = args.d[0]
_3x3_layers ={
    'vgg2_1':[64,128,64,112,112,3,1,1],'vgg3_1':[64,256,128,56,56,3,1,1], 'vgg3_2':[64,256,256,56,56,3,1,1], 'vgg4_1':[64,512,256,28,28,3,1,1],
    'vgg4_2':[64,512,512,28,28,3,1,1],'vgg5_2':[64,512,512,14,14,3,1,1],'alex3':[128,384,192,13,13,3,1,1],
    'alex4':[128,256,384,13,13,3,1,1],'alex5':[128,256,256,13,13,3,1,1],
    'overfeat3':[64,512,256,12,12,3,1,1], 'overfeat4':[64,1024,512,12,12,3,1,1], 'overfeat5':[64,1024,1024,12,12,3,1,1], 'resnet1':[1,64,64,56,56,3,1,1], 'test':[2, 32,32,12,12,3,1,1]}

# The sizes of inputs and filters
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
vlen = 16                               #SW
assert(pad_height == pad_width)
assert(stride_height == stride_width)
assert(kernel_height == kernel_width)
assert(in_channel%vlen == 0)
assert(out_channel%vlen == 0)
padding = pad_height
output_width = ((input_width + 2 * pad_width - kernel_width) // stride_width) + 1       #Q
output_height = ((input_height + 2 * pad_height - kernel_height) // stride_height) + 1  #P

'''
grad_input_dims = (batch,  math.ceil(in_channel/vlen), input_height + 2*pad_height , input_width + 2*pad_width, vlen)    #FP input dims
grad_dims = (batch, math.ceil(out_channel/vlen), output_height, output_width, vlen) #FP output dims
weight_dims = (math.ceil(out_channel/vlen), math.ceil(in_channel/vlen), kernel_height, kernel_width, vlen, vlen)    #FP weight dims
'''

#original dims
grad_input_dims = (batch,  input_height + 2*pad_height , input_width + 2*pad_width, in_channel)    #FP input dims
grad_dims = (batch, output_height, output_width, out_channel) #FP output dims
weight_dims = (out_channel, in_channel, kernel_height, kernel_width)    #FP weight dims



def generate_variants(func, tf):
    s = tvm.create_schedule(func.op)

    #a list of transformations go here. Transformation parameters are coming from the model and passed in 'tf'
       
    return [s]

def default_transform(func):

    s = tvm.create_schedule(func.op)
    '''
    n, co, h, w, ci = s[func].op.axis
    rko, ry, rx, rki = s[func].op.reduce_axis
    '''

    return [s]

def compile_and_run(s, A, B, W):

    with tvm.build_config(data_alignment=64):
        func = tvm.build(s, [A, W, B], target = 'llvm -mtriple=x86_64 -mcpu=skylake-avx512 -mattr=+skx,+fma,+fma4,+avx512ifma,+avx512f,+avx512cd,+avx512bw,+avx512vl,+avx512dq')
        func.save('backward.S', 'asm')
        ctx = tvm.context('llvm',0)
        
        grad_np = np.random.uniform(size=grad_dims).astype(A.dtype)
        weight_np = np.random.uniform(size=weight_dims).astype(W.dtype)

        grad_input = tvm.nd.array(np.zeros(grad_input_dims, dtype=B.dtype), ctx)

        grad = tvm.nd.array(grad_np, ctx)
        weight = tvm.nd.array(weight_np, ctx)

        func(grad, weight, grad_input)


        evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
        t = evaluator(grad, weight, grad_input).mean
        gflops = np.prod(get_const_tuple(B.shape))*in_channel*kernel_height*kernel_width*2/(t*1e9)
        
        print(tvm.lower(s, [A,W,B], simple_mode=True))       
        print("====Backward====\n")
        print("Time is : {0:.6f}".format(t))
        print("GFLOPS  : {0:.3f} ".format( gflops))
    #return 0,0
    return 0, gflops

def driver():
    A = tvm.placeholder(grad_dims, name='A')
    W = tvm.placeholder(weight_dims , name='W')
    #B = tvm.placeholder(grad_input_dims, name='B')
    #B = tvm.compute(grad_input_dims, lambda n,co,hr,ws,ci:0)
    B = tvm.nd.array(np.zeros(grad_input_dims))
    #print(B)
    #BP
    '''
    def bp(*args):
        for n in range(0,batch):
            for ko in range(0,math.ceil(out_channel/vlen)) :
                for h in range(0, input_height):
                    for w in range(0,input_width):
                        for co in range(0, math.ceil(in_channel/vlen)):
                            for r in range(0,kernel_height):
                                for s in range(0,kernel_width):
                                    for ci in range(0,vlen):
                                        for ki in range(0,vlen): 
                                            B[n][co][h+r][w+s][ci] += A[n][ko][h][w][ki]*W[ko][co][r][s][ki][ci]
                                             
     
        return B
    '''

    '''
    #BP with original dims
    def bp(*args):
        for n in range(0,batch):
            for k in range(0, out_channel) :
                for h in range(0, input_height):
                    for w in range(0,input_width):
                        for c in range(0, in_channel):
                            for r in range(0,kernel_height):
                                for s in range(0,kernel_width):
         
                                    B[n][h+r][w+s][c] += A[n][h][w][k]*W[k][c][r][s]
                                    #B[(((((n*batch)+c)*in_channel+h+r)*input_height+w+s)*input_width)] += A[((((n*batch)+h)*input_height+w)*input_width+k)*out_channel]*W[((((k*out_channel)+c)*in_channel+r)*kernel_height+s)*kernel_width]
        return B
    '''

    #iterable axes
    #rco = tvm.reduce_axis((0,math.ceil(in_channel/vlen)), name='rco')
    #rci = tvm.reduce_axis((0,vlen), name='rci')
    #ry = tvm.reduce_axis((0, kernel_height), name='ry')
    #rx = tvm.reduce_axis((0,kernel_width), name='rx')
    #rh = tvm.reduce_axis((0, input_height), name='rh')
    #rw = tvm.reduce_axis((0, input_width), name='rw')

    #rko = tvm.reduce_axis((0, math.ceil(out_channel/vlen)), name='rko')
    #rki = tvm.reduce_axis((0,vlen), name='rki')

    #compute backward 
    rk = tvm.reduce_axis((0,out_channel), name='rk')

    grad = tvm.placeholder((batch, input_height, input_width, out_channel),name='grad')
    weights = tvm.placeholder((out_channel, in_channel, kernel_height, kernel_width),name='weights')

    Btmp = tvm.compute((batch, input_height, input_width, kernel_height, kernel_width, in_channel), lambda n, h, w, r,s,c: tvm.sum(grad[n][h][w][c]*weights[rk][c][r][s], axis=[rk]), name='tmp')
    rr = tvm.reduce_axis((0, kernel_height), name='rr')
    ss = tvm.reduce_axis((0, kernel_width), name='ss')

    grad_input = tvm.compute((batch, input_height+kernel_height-1, input_width+kernel_width-1 ,in_channel), lambda n, h, w, c: tvm.sum(Btmp[n][h][w][rr][ss][c], axis=[rr,ss]), name='grad_input')


    #TEST
    x = generate_variants(grad_input, [])
    _ , flops = compile_and_run(x[0], grad, grad_input, weights)
    
    '''
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

    v = [batch, out_channel, in_channel, int(output_height), int(output_width), kernel_height, stride_height, pad_height]
    lb = mdl.init(v)
    conv_configs = mdl.tile_and_footprint_analysis(lb, search_harder=False, output_volume_multiplier=1)
    
    which_loads = {'grad_input':2, 'grad':4, 'weight':3, 'total':5}
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
            variants =  generate_variants(grad_input, loads_[no_dups[i] + config][1] )
            skip = False
            for net in variants:
 
                fflops_tvm, fflops = compile_and_run(net,A,W,B)
                
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
    '''
 
if __name__ == "__main__":
    driver()
