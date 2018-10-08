#model for backward pass

from itertools import *
import math
import numpy

l1_size = 32768

L1_latency = 3
L2_latency = 15
DRAM_latency = 100

L2_cache_lines = 16384
L1_cache_lines = 512
L1_cache_lines = 32768/32
VLEN = 16


def tile_and_footprint_analysis(loop_bounds, search_harder=False):    
    if loop_bounds['C'] >= VLEN:
          order = [['CCC','HH','WW','CC', 'H', 'W','K', 'R','S', 'C']]
    else:
          order = [['HH','WW','CC', 'H', 'W','K', 'R', 'S', 'C']]

    all_permutations = []
 

def init(l):
    loop_bounds = {}
    loop_bounds['N'] = l[0]
    loop_bounds['C'] = l[2]
    loop_bounds['K'] = l[1]
    loop_bounds['H'] = l[3]
    loop_bounds['W'] = l[4]
    loop_bounds['R'] = l[5]
    loop_bounds['S'] = l[5]

    return loop_bounds
