#!/bin/bash -l
#PBS -N tvm-tune
#PBS -l walltime=3:00:00,nodes=1:ppn=112
#PBS -o tune.out
#PBS -m abe 
#PBS -M tharindurusira@gmail.com
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
export KMP_BLOCKTIME=1
cd /homes/tharindu/tvm/topi/tests/python

#    'vgg2_1':[64,128,64,112,112,3,1,1],'vgg3_1':[64,256,128,56,56,3,1,1], 'vgg3_2':[64,256,256,56,56,3,1,1], 'vgg4_1':[64,512,256,28,28,3,1,1],
#    'vgg4_2':[64,512,512,28,28,3,1,1],'vgg5_2':[64,512,512,14,14,3,1,1],'alex3':[128,384,192,13,13,3,1,1],
#    'alex4':[128,256,384,13,13,3,1,1],'alex5':[128,256,256,13,13,3,1,1],
#    'overfeat3':[64,512,256,12,12,3,1,1], 'overfeat4':[64,1024,512,12,12,3,1,1], 'overfeat5':[64,1024,1024,12,12,3,1,1], 'resnet1':[1,64,64,56,56,3,1,1], 'test':[2,32,32,12,12,3,1,1]}

python test.py -d test
#python tune_wu.py -d vgg2_1
#python tune_wu.py -d vgg3_1
#python tune_wu.py -d vgg3_2
#python tune_wu.py -d vgg4_1
#python tune_wu.py -d vgg4_2
#python tune_wu.py -d vgg5_2
#python tune_wu.py -d alex3
#python tune_wu.py -d alex4
#python tune_wu.py -d alex5
#python tune_wu.py -d overfeat3
#python tune_wu.py -d overfeat4
#python tune_wu.py -d overfeat5
#python tune_wu.py -d resnet1
#python tune_wu.py -d test

