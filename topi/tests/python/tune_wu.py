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
    'overfeat3':[64,512,256,12,12,3,1,1], 'overfeat4':[64,1024,512,12,12,3,1,1], 'overfeat5':[64,1024,1024,12,12,3,1,1], 'resnet1':[32,64,64,56,56,3,1,1], 'test':[2,32,32,12,12,3,1,1]}

# The sizes of inputs and filters
batch = _3x3_layers[layer][0]           #N
in_channel = _3x3_layers[layer][2]      #C
out_channel = _3x3_layers[layer][1]     #K
input_height = _3x3_layers[layer][3]    #H
input_width = _3x3_layers[layer][4]     #WU
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

input_dims = (batch, math.ceil(in_channel/vlen), input_height + 2*pad_height , input_width + 2*pad_width,  vlen)    #FP input dims
grad_dims = (batch, math.ceil(out_channel/vlen), output_height, output_width, vlen) #FP output dims
grad_weight_dims = (math.ceil(out_channel/vlen), math.ceil(in_channel/vlen), kernel_height, kernel_width, vlen, vlen)    #FP weight dims

#for verification
def wu(A, W, B):
    for n in range(0,batch):
        for ko in range(0,math.ceil(out_channel/vlen)):
            for h in range(0, output_height):
                for w in range(0,output_width):
                    for co in range(0, math.ceil(in_channel/vlen)):
                        for r in range(0,kernel_height):
                            for s in range(0,kernel_width):
                                for ci in range(0,vlen):
                                    for ki in range(0,vlen): 
                                        B[ko][co][r][s][ki][ci] += A[n][ko][h][w][ki]*W[n][co][h+r][w+s][ci]

    return B

def wu1(A, W, B):
    for n in range(0,batch):
        for k in range(0,out_channel):
            for h in range(0, output_height):
                for w in range(0,output_width):
                    for c in range(0,in_channel):
                        for r in range(0,kernel_height):
                            for s in range(0,kernel_width): 
                                B[k][c][r][s] += A[n][k][h][w]*W[n][c][h+r][w+s]

    return B
                               

 
def generate_variants(func, tf):
    s = tvm.create_schedule(func.op)

    #a list of transformations go here. Transformation parameters are coming from the model and passed in 'tf'
       
    return [s]

def resnet_transform(func):

    #'resnet1':[1,64,64,56,56,3,1,1]
    s = tvm.create_schedule(func.op)

    #iter_var(name, Range(min=0, extent=bound))
    ko, co,  rr, ss, ki, ci  = s[func].op.axis
    n, rh, rw = s[func].op.reduce_axis
    
    tile = input_height-1
    while input_height%tile != 0:
        tile = tile - 1
    ho, wo, hi, wi = s[func].tile(rh, rw,14 , 14)
   
    order =  [ko, co, n, ho, wo, rr, ss, hi, wi, ki, ci]

    s[func].reorder(*order)
    
    #can't parallelize n, dependency
    koco_fused = s[func].fuse(ko, co)
    s[func].parallel(koco_fused)
    
    s[func].vectorize(ci)
    s[func].unroll(ki)
    s[func].unroll(wi)
    #s[func].unroll(hi)
    s[func].unroll(ss)
    #s[func].unroll(rr)
   
    return [s]


def default_transform(func):
    #an experimental schedule to figure out what's 'generally' good for WU
    s = tvm.create_schedule(func.op)
    
    #iter_var(name, Range(min=0, extent=bound))
    ko, co,  rr, ss, ki, ci  = s[func].op.axis
    n, rh, rw = s[func].op.reduce_axis
    
    tile = input_height-1
    while input_height%tile != 0:
        tile = tile - 1
    ho, wo, hi, wi = s[func].tile(rh, rw, tile, tile)
   
    inner_tile = tile -1
    
    if inner_tile != 0:
        while tile%inner_tile !=0:
            inner_tile = inner_tile-1

        hi_outer, hi_inner = s[func].split(hi, factor=inner_tile)
        order = [ko, co, n, ho, wo, hi_outer, rr, ss,wi, hi_inner, ki, ci]
        
    else:
        order =  [ko, co, n, ho, wo, hi, rr, ss, wi, ki, ci]
    
    s[func].reorder(*order)
    
    #can't parallelize n, dependency
    koco_fused = s[func].fuse( ko, co)
    s[func].parallel(koco_fused)
    
    s[func].vectorize(ci)
    s[func].unroll(ki)
    if inner_tile!=0:
        s[func].unroll(hi_inner)   

    else:
        s[func].unroll(wi)
    #s[func].unroll(ss)
    #s[func].unroll(rr)
    return [s]

def compile_and_run(s, A, B, W):

    with tvm.build_config(data_alignment=64):
        func = tvm.build(s, [A, W, B], target = 'llvm -mtriple=x86_64 -mcpu=skylake-avx512 -mattr=+skx,+fma,+fma4,+avx512ifma,+avx512f,+avx512cd,+avx512bw,+avx512vl,+avx512dq')
        func.save('weight_update.S', 'asm')
        ctx = tvm.context('llvm',0)
        
        grad_np = np.random.uniform(size=grad_dims).astype(A.dtype)
        input_np = np.random.uniform(size=input_dims).astype(W.dtype)

        b = tvm.nd.array(np.zeros(grad_weight_dims, dtype=B.dtype), ctx)

        a = tvm.nd.array(grad_np, ctx)
        w = tvm.nd.array(input_np, ctx)
      
        func(a, w, b)
        print(tvm.lower(s, [A,W,B], simple_mode=True))       

        #verify correctness
       
        #b_ref = wu(a.asnumpy(), w.asnumpy(), np.zeros(grad_weight_dims, dtype=B.dtype))
        #np.testing.assert_allclose(b.asnumpy(), b_ref, rtol=1e-5)
 
        #print("Verified...")
        evaluator = func.time_evaluator(func.entry_name, ctx, number=1, repeat=1000)
        t = evaluator(a, w, b).mean
        gflops = batch*in_channel*out_channel*input_height*input_width*kernel_height*kernel_width*2/1e9
        perf = gflops/t
        
        print("====WU====\n")
        print("Time is : {0:.6f}".format(t))
        print("GFLOPS  : {0:.3f} ".format( gflops))
        print("GFLOPS/s  : {0:.6f} ".format( perf))

    return 0, perf
    #return 0,0

def driver():
    grad = tvm.placeholder(grad_dims, name='grad')
    inputs = tvm.placeholder(input_dims , name='inputs')
 
    #reduction axes
    rn = tvm.reduce_axis((0,batch), name='n')
    rh = tvm.reduce_axis((0, input_height), name='rh')
    rw = tvm.reduce_axis((0, input_width), name='rw')

    #compute weight update
    
    grad_weight = tvm.compute(grad_weight_dims, 
                            lambda ko, co, rr, ss, ki, ci: tvm.sum(
                                grad[rn, ko, rh, rw, ki] * inputs[rn, co, rh+rr, rw+ss, ci], 
                                axis=[rn,rh, rw]), 
                            name = 'grad_weight')
    '''
    grad_weight = tvm.extern(grad_weight_dims, [grad, inputs], lambda ins, outs: tvm.call_packed("tvm.contrib.wu", ins[0], ins[1], outs[0]), name="grad_weights")

    '''

    #TEST
    #x = generate_variants(grad_weight, [])
    if layer == 'resnet1' or layer == 'test':
        x = resnet_transform(grad_weight)
    else:
        x = default_transform(grad_weight)
    _ , flops = compile_and_run(x[0], grad, grad_weight, inputs)
    
    #f = open('wu_results.txt','a')
    #f.write(layer+' '+ str(flops)+'\n')
    #f.close()
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
