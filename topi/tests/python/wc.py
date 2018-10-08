from itertools import *
#import latte.core
import math
import numpy
#from latte import *


l1_size = 32768
#cache_lines = l1_size/64
L1_latency = 3
L2_latency = 15
DRAM_latency = 100

#if "AVX-512" in latte.config.vec_config:
L2_cache_lines = 16384
L1_cache_lines = 512
#else: 
#    L2_cache_lines = (256*1024)/32
#    L1_cache_lines = 32768/32
VLEN = 16



#def pruning_heuristics():
'''
     1. Fix the inner unroll factors, if unrolled output elements in innermost tile < 14 ignore 

     2. Compute footprint of 56 threads*chunk*fp/thread  ensure < 38.5 MB(L3) if not prune

     3. Compute footprint of 2 Threads*2(innermost tile footprint)*chunk <  1 MB , if not prune 

     4. Compute innermost tile footprint < 1/2*(32 KB) if not prune

     5. Outline of core calculation

          NNN
           KKK
            HHH
             WWW
               NN
                KK 
                 HH
                  WW
                    -PREFETCH OUTPUT NEXT NN+tn+1,KK+tk+1,HH+th+1,WW+tw+1   to L1
                    -PREFETCH OUTPUT NEXT NN+tn+1,KK+tk+1,HH+th+1,WW+2*tw+1 to L2 
                    
                   CC
                    -PREFETCH input NEXT NN+tn+1,CC+tc+1,HH + th + r +1,WW+ts + s +1   to L1
                    -PREFETCH input next NN+tn+1,CC+2*tc+1,HH + th + r+1,WW+ts + s + 1 to L2 
                    -PREFETCH weight NEXT KK + tk + 1,CC + tc + 1, r,s to L1
                    -PREFETCH weight NEXT KK + tk + 1,CC + 2*tc + 1, r,s to L2
 

                                    
                     -- unroll boundary
                     -- N    -disappear
                     --  K   -disappear
                     --   H  -disappear
                     --    W -disappear
                           -PREFETCH NEXT N,K,H,W, WITHIN C,R,S LOOPS 
                           VECTOR LOAD   
                     --     C 
                     --       R  
                     --        S
                               fma 
                            
      6. cost with prefetching = NN*KK*HH*WW*CC + //assume perfect prefetching
         prefetch_cost = get_reused_load_cost   
''' 

def calculate_input_fp(tiled_loop_order, tiled_loop_bounds,start_loop,target_loop, actual_loop_bounds):
    input_fp = 1
    output_fp =1
    weight_fp = 1
      

    ''' 
    for i in tiled_loop_order[tiled_loop_order.index(start_loop):]:
        
              if i == 'CCC':
                input_fp *= actual_loop_bounds['C']/tiled_loop_bounds['CC']
              elif i == 'CC':
                input_fp *= tiled_loop_bounds[i]
              elif i == 'H':
                input_fp *= (tiled_loop_bounds['H']+ tiled_loop_bounds['R'] - 1)
              elif i == 'W':
                input_fp *= (tiled_loop_bounds['W']+ tiled_loop_bounds['S'] - 1)
              elif  i == 'HH':
                input_fp *= tiled_loop_bounds[i]

    return input_fp/16

    '''


    if target_loop == "WW":
          for i in tiled_loop_order[tiled_loop_order.index(start_loop):]:
 
              if i == 'CCC':
                input_fp *= actual_loop_bounds['C']/tiled_loop_bounds['CC']
              elif i == 'CC':
                input_fp *= tiled_loop_bounds[i]
              elif i == 'H':
                input_fp *= (tiled_loop_bounds['H']+ tiled_loop_bounds['R'] - 1)
              #elif i == 'W':
                
              #  input_fp *= (tiled_loop_bounds['W']+ tiled_loop_bounds['S'] - 1)
              #elif  i == 'HH':
              #  input_fp *= tiled_loop_bounds[i]

              #return input_fp/16
        
          #return input_fp
    elif target_loop == "HH":
          for i in tiled_loop_order[tiled_loop_order.index(start_loop):]:
 
              if i == 'CCC' :
                  input_fp *= actual_loop_bounds['C']/tiled_loop_bounds['CC']
              elif i == 'CC':
                  input_fp *= tiled_loop_bounds[i]
              #elif i == 'H':
              #  input_fp *= tiled_loop_bounds['R'] - 1
              #elif i == 'W':
              #    input_fp *= (tiled_loop_bounds['W']+ tiled_loop_bounds['S'] - 1)
 


              #return input_fp/16
              #input_fp *= 1
 
              #elif  i == 'WW':
              #    input_fp *= tiled_loop_bounds[i]
 
    elif target_loop == "H":
          for i in tiled_loop_order[tiled_loop_order.index(start_loop):]:
 
              if i == 'CCC' :
                  input_fp *= actual_loop_bounds['C']/tiled_loop_bounds['CC']
              elif i == 'CC':
                  input_fp *= tiled_loop_bounds[i]
              #elif i == 'H':
              #  input_fp *= tiled_loop_bounds['R'] - 1
              #elif i == 'W':
              #    input_fp *= (tiled_loop_bounds['W']+ tiled_loop_bounds['S'] - 1)
 
 
 
              #return input_fp/16
              #input_fp *= 1
 
              #elif  i == 'WW':
              #    input_fp *= tiled_loop_bounds[i]
 
 
 
 
    return input_fp/VLEN






def calculate_fp(tiled_loop_order, tiled_loop_bounds,start_loop, data_struct, actual_loop_bounds):
    input_fp = 1
    output_fp =1
    weight_fp = 1
    add = 1
    if data_struct == "input":
        for i in tiled_loop_order[tiled_loop_order.index(start_loop):]:
          if i in ['WW','HH','CCC']:
            if i == 'CCC':
               input_fp = (input_fp + add)*actual_loop_bounds['C']/tiled_loop_bounds['CC']
            else:
              if i == 'WW':
                #add = (tiled_loop_bounds['CC']/tiled_loop_bounds['C'])*(tiled_loop_bounds['R'] - 1)*(actual_loop_bounds['W'] + tiled_loop_bounds['S'] -  1)
                add =  (tiled_loop_bounds['CC']/tiled_loop_bounds['C'])*(tiled_loop_bounds['S'] - 1)*(tiled_loop_bounds['H'] + tiled_loop_bounds['R'] - 1)
                input_fp /= (tiled_loop_bounds['W'])
 
                #add += input_fp*(tiled_loop_bounds['S'] -1)  
                input_fp *= (actual_loop_bounds['W'])
 
              elif i == 'HH':
                add = (tiled_loop_bounds['CC']/tiled_loop_bounds['C'])*(tiled_loop_bounds['R'] - 1)*(actual_loop_bounds['W'] + tiled_loop_bounds['S'] -  1)
                input_fp /= (tiled_loop_bounds['H'])
                input_fp *= (actual_loop_bounds['H'])
 
 
          if i == 'CC':
              input_fp *= (tiled_loop_bounds[i]/VLEN)
              add *= (tiled_loop_bounds[i]/VLEN)
          if i == 'H':
              input_fp *= (tiled_loop_bounds['H'])
              add =  (tiled_loop_bounds['R'] - 1)*(tiled_loop_bounds['W'] + tiled_loop_bounds['S'] -1)
          if i == 'W':
              input_fp *= (tiled_loop_bounds['W'])
              add = (tiled_loop_bounds['R'] - 1)*input_fp


          '''      
          if i in ['WW','HH','CCC']:
            if i == 'CCC':
               input_fp *= actual_loop_bounds['C']/tiled_loop_bounds['CC']
            else:
              if i == 'WW':
                input_fp  =(tiled_loop_bounds['W'] + tiled_loop_bounds['S'] - 1)*(tiled_loop_bounds['H'] + tiled_loop_bounds['R'] - 1)*(actual_loop_bounds['C']/tiled_loop_bounds['CC']
              elif i == 'HH':
                add = 0
                input_fp  = (tiled_loop_bounds['R'] - 1)*(actual_loop_bounds['W'] + tiled_loop_bounds['S'] - 1)*(actual_loop_bounds['C']/tiled_loop_bounds['CC']
                #input_fp /= (tiled_loop_bounds['H'] + tiled_loop_bounds['R'] - 1)
                #input_fp *= (actual_loop_bounds['H'] + tiled_loop_bounds['R'] -1)
 
          if i == 'CC':
            input_fp *= (tiled_loop_bounds[i]/VLEN)
            add *= (tiled_loop_bounds[i]/VLEN)
          if i == 'H':
            input_fp *= (tiled_loop_bounds['H']+ tiled_loop_bounds['R'] - 1)
            add =  (tiled_loop_bounds['R'] - 1)*(tiled_loop_bounds['W'] + tiled_loop_bounds['S'] -1)
          if i == 'W':
            input_fp *= tiled_loop_bounds['R']*(tiled_loop_bounds['W']+ tiled_loop_bounds['S'] - 1)
            add = (tiled_loop_bounds['R'] - 1)*(tiled_loop_bounds['W']+ tiled_loop_bounds['S'] - 1)
          '''

  
        return input_fp,add

    if data_struct == "output":
        for i in tiled_loop_order[tiled_loop_order.index(start_loop):]: 
          if i in ['WW','HH', 'H','W','KK', 'KKK']:
            output_fp *= tiled_loop_bounds[i]

        return output_fp

    if data_struct == "weight":
        for i in tiled_loop_order[tiled_loop_order.index(start_loop):]:
 
          if i in ['KKK', 'KK','CC','C', 'CCC', 'R', 'S']:
             if i == 'CCC':
                weight_fp *= actual_loop_bounds['C']/tiled_loop_bounds['CC']
             elif i == 'CC':
                weight_fp *= tiled_loop_bounds['CC']/tiled_loop_bounds['C']
             else:
                weight_fp *= tiled_loop_bounds[i];
        return weight_fp

    return -1



def get_reuse_cost(key, last_loop_level,last_bound,tiled_loop_order, parallel_loop_bounds, tiled_loop_bounds, actual_loop_bounds):
      to_return = 1  
      
      if key in tiled_loop_order and tiled_loop_order.index(key) > tiled_loop_order.index(last_loop_level):
               to_return = 1
      elif tiled_loop_order.index(key) == tiled_loop_order.index(last_loop_level): 
          to_return = actual_loop_bounds[key[0]]/(last_bound*tiled_loop_bounds[key[0]])
      elif tiled_loop_order.index(key) < tiled_loop_order.index(last_loop_level) and  tiled_loop_order.index(key[0]) >  tiled_loop_order.index(last_loop_level):
          to_return = actual_loop_bounds[key[0]]/ tiled_loop_bounds[key[0]]
      elif tiled_loop_order.index(key) < tiled_loop_order.index(last_loop_level) and  tiled_loop_order.index(key[0]) ==  tiled_loop_order.index(last_loop_level):
          to_return = actual_loop_bounds[key[0]]/ last_bound
      else: 
          to_return = actual_loop_bounds[key[0]]
      return to_return


def calculate_outer_iteration_count(tiled_loop_order,tiled_loop_bounds,k):

    to_return = 1
    for i in tiled_loop_order[:tiled_loop_order.index(k)]:
          to_return *= tiled_loop_bounds[i]

         

    return to_return


def compute_cost_for_LLC(actual_loop_bounds, tiled_loop_order, tiled_loop_bounds, parallel_loops, parallel_loop_bounds, num_cores, output_volume_multiplier=1, llc='L1'):


    cost_per_thread=1/num_cores
    l2_tile = 2
 
 
    input_fp = 1
    weight_fp = 1
    output_fp = 1
    last_loop_level = -1
 
    input_reloads = 1
    output_reloads = 1
    weight_reloads = 1
    free_reloads = 0
    input_initial_misses =(actual_loop_bounds['H'] - 1)*tiled_loop_bounds['S']+(actual_loop_bounds['W'] - 1)\
                            *tiled_loop_bounds['R']+tiled_loop_bounds['R']*tiled_loop_bounds['S'] 
    L1_input_reloads = (0)
    L1_output_reloads = (0)
    L1_weight_reloads = (0)
    input_displaced = False
    output_displaced = False
    weight_displaced = False
    weight_of = False
    add = 1
    is_displaced ={"output":False, "input":False, "weight":False}
    reuse = {"output":1, "input": 1, "weight": 1}
    if llc == 'L1':
       cache_lines = L1_cache_lines
    elif llc == 'L2':
       cache_lines = L2_cache_lines
    else:
       assert(False)

    threshold_reached = False
    overflow = False
    flag_h  = False
    flag_w = False
    for i in reversed(tiled_loop_order):
      last_input_fp = input_fp
      last_output_fp = output_fp
      last_weight_fp = weight_fp
      

      if not overflow:

        if i in ['WW','HH','CCC']:
            if i == 'CCC':
               input_fp = (input_fp + add)*actual_loop_bounds['C']/tiled_loop_bounds['CC']
            else:
              if i == 'WW':
                add = (tiled_loop_bounds['CC']/tiled_loop_bounds['C'])*(tiled_loop_bounds['R'] - 1)*(actual_loop_bounds['W'] + tiled_loop_bounds['S'] -  1)

                input_fp /= (tiled_loop_bounds['W'] + tiled_loop_bounds['S'] -  1)
                
                #add += input_fp*(tiled_loop_bounds['S'] -1)  
                input_fp *= (actual_loop_bounds['W'] + tiled_loop_bounds['S'] -1)
                  
              elif i == 'HH':
                add = (tiled_loop_bounds['CC']/tiled_loop_bounds['C'])*(tiled_loop_bounds['R'] - 1)*(actual_loop_bounds['W'] + tiled_loop_bounds['S'] -  1)
                input_fp /= (tiled_loop_bounds['H'])
                input_fp *= (actual_loop_bounds['H'])

       
        if i == 'CC':
            input_fp *= (tiled_loop_bounds[i]/VLEN)
            add *= (tiled_loop_bounds[i]/VLEN)  
        if i == 'H':
            input_fp *= (tiled_loop_bounds['H'])
            add =  (tiled_loop_bounds['R'] - 1)*(tiled_loop_bounds['W'] + tiled_loop_bounds['S'] -1)  
        if i == 'W':
            input_fp *= (tiled_loop_bounds['W']+ tiled_loop_bounds['S'] - 1)
            add = (tiled_loop_bounds['R'] - 1)*input_fp  
        if i in ['WW','HH', 'H','W','KK', 'KKK']:
            output_fp *= tiled_loop_bounds[i];
 
        if i in ['KKK', 'KK', 'CCC','C', 'R', 'S']:
           if i == 'CCC':
              weight_fp *= actual_loop_bounds['C']/tiled_loop_bounds['CC']
           else:
              weight_fp *= tiled_loop_bounds[i];
        if i == 'CC':
            weight_fp *= tiled_loop_bounds['CC']/VLEN

        if llc == 'L1':
 
          if input_fp + weight_fp + output_fp + add <= cache_lines:
            threshold_reached = False            
            if i not in ['CCC','CC','R','S','C','K']:
                output_reloads *= tiled_loop_bounds[i]
            
            if i not in ['R','S','C','K']: 
                if i == 'CC':
                    input_reloads *= tiled_loop_bounds[i]/VLEN
                    #weight_reloads *= tiled_loop_bounds[i]
                elif i == 'CCC':
                    input_reloads *= (actual_loop_bounds['C']/tiled_loop_bounds['CC'])
                    #weight_reloads *= (actual_loop_bounds['C']/tiled_loop_bounds['CC'])                                
                elif i in ['H']:
                    input_reloads *= (tiled_loop_bounds[i])
                    input_reloads += ((tiled_loop_bounds['W'] + tiled_loop_bounds['S'] - 1 )*(tiled_loop_bounds['R'] - 1))  
                elif i in['W']:
                    input_reloads *= (tiled_loop_bounds[i] + tiled_loop_bounds['S'] - 1 )
                elif i == 'WW':
                    #halo for one H and whole W
                    input_reloads -= ((tiled_loop_bounds['CC']/tiled_loop_bounds['C'])*\
                                            (tiled_loop_bounds['W'] + tiled_loop_bounds['S'] - 1 )*(tiled_loop_bounds['R'] - 1))
                    # divide by W tile  
                    input_reloads /= ((tiled_loop_bounds['CC']/tiled_loop_bounds['C'])*(tiled_loop_bounds['W'] + tiled_loop_bounds['S'] - 1))
                    # multiply by whole W
                    input_reloads *= ((tiled_loop_bounds['CC']/tiled_loop_bounds['C'])*(actual_loop_bounds['W'] +tiled_loop_bounds['S'] -1)) 
                    # add halo
                    input_reloads += ((tiled_loop_bounds['CC']/tiled_loop_bounds['C'])*(actual_loop_bounds['W'] + tiled_loop_bounds['S'] - 1 )*(tiled_loop_bounds['R'] - 1))

                    
                elif i == 'HH':
                    input_reloads -= ((tiled_loop_bounds['CC']/tiled_loop_bounds['C'])*(actual_loop_bounds['W'] + tiled_loop_bounds['S'] - 1 )*(tiled_loop_bounds['R'] - 1))
                    input_reloads *= tiled_loop_bounds[i]
                    input_reloads += ((tiled_loop_bounds['CC']/tiled_loop_bounds['C'])*(actual_loop_bounds['W'] + tiled_loop_bounds['S'] - 1 )*(tiled_loop_bounds['R'] - 1))     

                    #input_reloads /= (tiled_loop_bounds['H'] + tiled_loop_bounds['R']  - 1)
                    #input_reloads *= (actual_loop_bounds['H'] +tiled_loop_bounds['R']  - 1)
                    #input_reloads = (tiled_loop_bounds['CC']/tiled_loop_bounds['C'])*(actual_loop_bounds['W'] + tiled_loop_bounds['S'] - 1 )*\
                    #                   (actual_loop_bounds['H'] + tiled_loop_bounds['R'] - 1 )*(tiled_loop_bounds['CC']/tiled_loop_bounds['C'])
                    # 
                    '''  
                    if tiled_loop_order.index('HH') < tiled_loop_order.index('WW'):
                            free_reloads += actual_loop_bounds['W']*(tiled_loop_bounds['R'] - 1)*tiled_loop_bounds['HH'] 
                    elif tiled_loop_order.index('HH') < tiled_loop_order.index('W'): 
                            free_reloads += (tiled_loop_bounds['H']*(tiled_loop_bounds['R'] - 1)*tiled_loop_bounds['HH']) 
                    '''    
  

            if i not in ['K', 'HH', 'H', 'WW', 'W']:

                if i == 'CC':
                    weight_reloads *= tiled_loop_bounds[i]/VLEN
                elif i == 'CCC':
                    weight_reloads *= (actual_loop_bounds['C']/tiled_loop_bounds['CC'])
                else:       
                    weight_reloads *= tiled_loop_bounds[i]
          else:
              threshold_reached = True
              last_loop_level = i
        else:  
          if input_fp + add +  2*(weight_fp + output_fp)  <= cache_lines:
 
            threshold_reached = False
            if i not in ['CCC','CC','R','S','C','K']:
                output_reloads *= tiled_loop_bounds[i]
           

 
            if i not in ['R','S','C','K']: 
                if i == 'CC':
                    input_reloads *= tiled_loop_bounds[i]/VLEN
                    #weight_reloads *= tiled_loop_bounds[i]/VLEN
                elif i == 'CCC':
                    input_reloads *= (actual_loop_bounds['C']/tiled_loop_bounds['CC'])
                    #weight_reloads *= (actual_loop_bounds['C']/tiled_loop_bounds['CC'])                                
                elif i in ['H']:
                    input_reloads *= (tiled_loop_bounds[i])
                    input_reloads += ((tiled_loop_bounds['W'] + tiled_loop_bounds['S'] - 1 )*(tiled_loop_bounds['R'] - 1))
                elif i in['W']:
                    input_reloads *= (tiled_loop_bounds[i] + tiled_loop_bounds['S'] - 1 )
                elif i == 'WW':
                    #halo for one H and whole W
                    input_reloads -= ((tiled_loop_bounds['CC']/tiled_loop_bounds['C'])*\
                                            (tiled_loop_bounds['W'] + tiled_loop_bounds['S'] - 1 )*(tiled_loop_bounds['R'] - 1))
                    # divide by W tile  
                    input_reloads /= ((tiled_loop_bounds['CC']/tiled_loop_bounds['C'])*(tiled_loop_bounds['W'] + tiled_loop_bounds['S'] - 1))
                    # multiply by whole W
                    input_reloads *= ((tiled_loop_bounds['CC']/tiled_loop_bounds['C'])*(actual_loop_bounds['W'] +tiled_loop_bounds['S'] -1))
                    # add halo
                    input_reloads += ((tiled_loop_bounds['CC']/tiled_loop_bounds['C'])*(actual_loop_bounds['W'] + tiled_loop_bounds['S'] - 1 )*(tiled_loop_bounds['R'] - 1))
 
 
                elif i == 'HH':
                    input_reloads -= ((tiled_loop_bounds['CC']/tiled_loop_bounds['C'])*(actual_loop_bounds['W'] + tiled_loop_bounds['S'] - 1 )*(tiled_loop_bounds['R'] - 1))
                    input_reloads *= tiled_loop_bounds[i]
                    input_reloads += ((tiled_loop_bounds['CC']/tiled_loop_bounds['C'])*(actual_loop_bounds['W'] + tiled_loop_bounds['S'] - 1 )*(tiled_loop_bounds['R'] - 1))
 

 
            if i not in ['K', 'HH', 'H', 'WW', 'W']:
 
                if i == 'CC':
                    weight_reloads *= tiled_loop_bounds[i]/VLEN
                elif i == 'CCC':
                    weight_reloads *= (actual_loop_bounds['C']/tiled_loop_bounds['CC'])
                else:       
                    weight_reloads *= tiled_loop_bounds[i]

          else:
            threshold_reached = True                
            last_loop_level = i
      if (threshold_reached): 
              input_reloads = max(1, input_reloads - free_reloads)    
              last_bound = 0  
              add = 0            
              for j in range(tiled_loop_bounds[i]):
                        
                 prev_input_fp = last_input_fp
                 prev_output_fp = last_output_fp
                 prev_weight_fp = last_weight_fp  
 
                 if i == 'WW':
                     prev_input_fp /= (tiled_loop_bounds['W'] + tiled_loop_bounds['S'] - 1)
                     prev_input_fp *= ((j+1)*tiled_loop_bounds['W'] + tiled_loop_bounds['S'] - 1)  
                     add = (tiled_loop_bounds['CC']/tiled_loop_bounds['C'])*(tiled_loop_bounds['R'] - 1)*((j+1)*tiled_loop_bounds['W'] + tiled_loop_bounds['S'] -  1)
 
                 elif i == 'HH':
                     prev_input_fp /= (tiled_loop_bounds['H'])
                     prev_input_fp *= ((j+1)*tiled_loop_bounds['H'])
                     add = (tiled_loop_bounds['CC']/tiled_loop_bounds['C'])*(tiled_loop_bounds['R'] - 1)*(actual_loop_bounds['W'] + tiled_loop_bounds['S'] -  1) 
  
                 if i == 'CC':
                    prev_input_fp *= math.ceil(float(j+1)/VLEN )
                 if i == 'H':
                    prev_input_fp *= (j+1)
                    add = (tiled_loop_bounds['R'] - 1)*(tiled_loop_bounds['W'] + tiled_loop_bounds['S'] -1)
                 if i == 'W':
                    prev_input_fp *= (j+1 +  tiled_loop_bounds['S'] - 1)*(tiled_loop_bounds['R'])
                 if i == 'CCC':
                    prev_input_fp *= (j+1) #actual_loop_bounds['C']/tiled_loop_bounds['CC'] 
 
 
                 if i in ['WW','HH', 'H','W','KK', 'KKK']:
                    prev_output_fp *= (j+1);
 
 
                 if i in ['KKK', 'KK', 'C', 'R', 'S']:
                    prev_weight_fp *=  (j+1);
                 if i == 'CC':
                    prev_weight_fp *= math.ceil((j+1)/VLEN)
                 if i == 'CCC':
                    prev_weight_fp *= (j+1)#actual_loop_bounds['C']/tiled_loop_bounds['CC'] 
 
                 if llc == 'L1':  

                    if prev_input_fp + prev_weight_fp + prev_output_fp + add <= cache_lines:
                        last_bound = j
                        #last_input_fp = prev_input_fp
                        #last_output_fp = prev_output_fp
                        #last_weight_fp = prev_weight_fp
                    else:
                        break
                 else:
                    if prev_input_fp + 2*(prev_weight_fp + prev_output_fp) + add <= cache_lines:
                        last_bound = j
                        #last_input_fp = prev_input_fp
                        #last_output_fp = prev_output_fp
                        #last_weight_fp = prev_weight_fp
                    else:
                        break

              overflow = True
        
      if overflow: 
          start_loop = tiled_loop_order[tiled_loop_order.index(i) + 1]
          input_fp_displaced, additional =  (calculate_fp(tiled_loop_order, tiled_loop_bounds,start_loop, "input", actual_loop_bounds))
          output_fp_displaced = (calculate_fp(tiled_loop_order, tiled_loop_bounds,start_loop, "output", actual_loop_bounds))
          weight_footprint = (calculate_fp(tiled_loop_order, tiled_loop_bounds,start_loop, "weight", actual_loop_bounds))
          if llc == 'L1':
              total_fp = input_fp_displaced + output_fp_displaced + weight_footprint
              temp = output_fp_displaced
              temp2 = weight_footprint  
          else:
              total_fp = input_fp_displaced + 2*(output_fp_displaced + weight_footprint)
              temp = output_fp_displaced*2
              temp2 = weight_footprint*2
          if not threshold_reached:
              last_bound2 = 0
          else:
              last_bound2 = last_bound
          if i in ['WW','HH','CCC','CC','H','W','R','S']:
                if i == 'CCC':
                    #last_bound = 0  
                    input_reloads *= actual_loop_bounds['C']/tiled_loop_bounds['CC']
                    weight_reloads *= actual_loop_bounds['C']/tiled_loop_bounds['CC']    

                    if not threshold_reached:
                        if temp  > cache_lines and \
                            (temp-cache_lines )*(actual_loop_bounds['C']/tiled_loop_bounds['CC']  - 1) > 0:
                            output_reloads += ((temp-cache_lines)*(actual_loop_bounds['C']/tiled_loop_bounds['CC']  - 1))

                    else:
                        if  temp  > cache_lines and \
                         (temp-cache_lines)*(actual_loop_bounds['C']/tiled_loop_bounds['CC'] -last_bound - 1) > 0 and \
                                (actual_loop_bounds['C']/tiled_loop_bounds['CC'] -last_bound - 1) > 0:
                                output_reloads += \
                                    ((temp-cache_lines)*(actual_loop_bounds['C']/tiled_loop_bounds['CC'] -last_bound - 1))
    
                elif i == 'CC':
                    input_reloads *= (tiled_loop_bounds[i]/VLEN)
                    weight_reloads *= (tiled_loop_bounds[i]/VLEN)


                    #if 2*(temp + input_fp_displaced) + tiled_loop_bounds['C']*tiled_loop_bounds['R']*tiled_loop_bounds['S'] > cache_lines:
                    #    dp = 2*(temp + input_fp_displaced) + temp2 - cache_lines
                    #    weight_reloads = (weight_reloads+ dp)*(tiled_loop_bounds[i]/VLEN)                      

                    if temp > cache_lines and\
                        (temp-cache_lines)*(tiled_loop_bounds[i]/VLEN -last_bound2 - 1) > 0 and \
                          (tiled_loop_bounds[i]/VLEN -last_bound2 - 1) > 0:  
                        output_reloads += ((temp-cache_lines)*(tiled_loop_bounds[i]/VLEN -last_bound2 - 1)) 
                else:
                   
     

                   #last_bound2 = 0 
                   if i in ['WW'] :
                        if temp2 + temp > cache_lines:
                            weight_reloads *= tiled_loop_bounds[i]
                            weight_of = True      
                   if i in ['HH'] and weight_of:        
                          weight_reloads *= tiled_loop_bounds[i]

                   #if i in ['HH', 'WW']:    
                   #   input_fp_displaced_temp = input_fp_displaced/(tiled_loop_bounds['CC']/VLEN) 
                   #   temp3 = temp2/(tiled_loop_bounds['CC']/VLEN) 
                   #L:qelse:
                   #   input_fp_displaced_temp = input_fp_displaced
                   #   temp3 = temp2
                   

                   # additional is halo region
                   if i == 'H':
                      additional = (tiled_loop_bounds['R'] - 1)*(tiled_loop_bounds['W'] + tiled_loop_bounds['S'] -1) 
                      input_threshold = output_fp_displaced + additional + input_fp_displaced + temp2
                   elif i == 'W':
                      additional = tiled_loop_bounds['S'] - 1    
                      input_threshold = additional + temp2      
                   elif i == 'WW':
                      additional = (tiled_loop_bounds['S'] -1 )*tiled_loop_bounds['H']
                      input_threshold = output_fp_displaced + input_fp_displaced + additional + temp2
                   else:              
                      additional = (tiled_loop_bounds['R'] - 1)*(actual_loop_bounds['W'] + tiled_loop_bounds['S'] -1)
                      input_threshold = (additional + actual_loop_bounds['W']) + temp2     

                   if  input_threshold > cache_lines: 
                      #last_bound2 = 0
                      if i in ['H', 'W']:
                         if i == 'W':
                              last_w = last_bound2 
                              flag_w = True  
                              t = 0
                              if last_bound2 > 0:  
                                  t = input_reloads*(last_bound2 + tiled_loop_bounds['S'] - 1 )

                              t += (input_reloads*(tiled_loop_bounds['W'] - last_bound2)*tiled_loop_bounds['S'])  
                              input_reloads = t  
                              #input_reloads += tiled_loop_bounds['S']*tiled_loop_bounds['R'] + max(0, (last_bound2 - 1))*tiled_loop_bounds['R']\
                               #                     +  max(0, (tiled_loop_bounds[i] - last_bound2)*tiled_loop_bounds['R']*tiled_loop_bounds['S'])
                         elif i == 'H':
                              # .....
                              # .....
                              # .....
                              # ..... 
                              flag_h = True        
                              cache = input_reloads  
                              t = 0 
                              if last_bound2 > 0:
                                  t = cache*(last_bound2 + tiled_loop_bounds['R'] - 1 )
                                
                              t += (cache*(tiled_loop_bounds['H'] - last_bound2)*tiled_loop_bounds['R'])


                              input_reloads = t    

                              #input_reloads +=  tiled_loop_bounds['S'] + max(0, last_bound2 - 1)*tiled_loop_bounds['W']\
                              #                  + max(0, (tiled_loop_bounds[i] - last_bound2)*tiled_loop_bounds['R']*tiled_loop_bounds['S'])*\
                              #                    tiled_loop_bounds['W']
                              
                              #if flag_w:
                              #    input_reloads += tiled_loop_bounds['S']*(tiled_loop_bounds['R'] - 1)      

                      elif i in ['HH', 'WW']:
                         input_reloads *= (tiled_loop_bounds[i])
                   else:
                     if i in ['H']:
                        input_reloads *= (tiled_loop_bounds[i])
                        input_reloads += ((tiled_loop_bounds['W'] + tiled_loop_bounds['S'] - 1 )*(tiled_loop_bounds['R'] - 1))
                     elif i in['W']:
                        input_reloads *= (tiled_loop_bounds[i] + tiled_loop_bounds['S'] - 1 )
                     elif i == 'WW':
                        input_reloads -= ((tiled_loop_bounds['CC']/tiled_loop_bounds['C'])*(tiled_loop_bounds['W'] + tiled_loop_bounds['S'] - 1 )*(tiled_loop_bounds['R'] - 1))
 
                        input_reloads /= (tiled_loop_bounds['W'] + tiled_loop_bounds['S'] - 1)
                        input_reloads *= (actual_loop_bounds['W'] +tiled_loop_bounds['S'] -1)
                          
                        input_reloads += ((tiled_loop_bounds['CC']/tiled_loop_bounds['C'])*(actual_loop_bounds['W'] + tiled_loop_bounds['S'] - 1 )*(tiled_loop_bounds['R'] - 1)) 

                          #input_reloads = (tiled_loop_bounds['H'] + tiled_loop_bounds['R'] - 1 )*(tiled_loop_bounds['CC']/tiled_loop_bounds['C'])*(actual_loop_bounds['W'] + tiled_loop_bounds['S'] - 1 )*(tiled_loop_bounds['R'] - 1)#+= ((tiled_loop_bounds['CC']/tiled_loop_bounds['C'])*(actual_loop_bounds['W'] + tiled_loop_bounds['S'] - 1 )*(tiled_loop_bounds['R'] - 1))
 
 
                     elif i == 'HH':
                        input_reloads -= ((tiled_loop_bounds['CC']/tiled_loop_bounds['C'])*(actual_loop_bounds['W'] + tiled_loop_bounds['S'] - 1 )*(tiled_loop_bounds['R'] - 1))
                        input_reloads *= tiled_loop_bounds[i]
                        input_reloads += ((tiled_loop_bounds['CC']/tiled_loop_bounds['C'])*(actual_loop_bounds['W'] + tiled_loop_bounds['S'] - 1 )*(tiled_loop_bounds['R'] - 1))

                     #   input_reloads =(tiled_loop_bounds['CC']/tiled_loop_bounds['C'])*(actual_loop_bounds['W'] + tiled_loop_bounds['S'] - 1 )*\
                     #                 (actual_loop_bounds['H'] + tiled_loop_bounds['R'] - 1 )*(tiled_loop_bounds['CC']/tiled_loop_bounds['C'])
   
                   if i in ['H','HH', 'W', 'WW'] :
                      output_reloads *= (tiled_loop_bounds[i])
          threshold_reached = False



    


   
    return 1, input_reloads + input_initial_misses , 1


def compute_cost(actual_loop_bounds, tiled_loop_order, tiled_loop_bounds, parallel_loops, parallel_loop_bounds, num_cores, output_volume_multiplier=1):

  l1_output_reloads,l1_input_reloads,l1_weight_reloads =    compute_cost_for_LLC(actual_loop_bounds, tiled_loop_order, tiled_loop_bounds, parallel_loops, parallel_loop_bounds, num_cores, output_volume_multiplier, llc='L1')

  l2_output_reloads,l2_input_reloads,l2_weight_reloads =    compute_cost_for_LLC(actual_loop_bounds, tiled_loop_order, tiled_loop_bounds, parallel_loops, parallel_loop_bounds, num_cores, output_volume_multiplier, llc='L2')

  last_bound = 0
  last_loop_level=0
  cost_per_thread=1/num_cores
  l2_tile = 2

  input_loads_stores = l2_input_reloads*DRAM_latency + l1_input_reloads*L2_latency
  output_loads_stores =(output_volume_multiplier)*(l2_output_reloads*DRAM_latency + l1_output_reloads*L2_latency)
  weight_loads_stores = l2_weight_reloads*DRAM_latency + l1_weight_reloads*L2_latency
  total_loads_stores =  input_loads_stores + weight_loads_stores + output_loads_stores
  
  '''
  print("total cost")
  print(l1_output_reloads)

  print(l1_weight_reloads)

  print(l1_input_reloads)
  print(l2_output_reloads)
 
  print(l2_weight_reloads)
 
  print(l2_input_reloads)
  '''
  #print(total_loads_stores) 
                        
                        
  #if(input_loads_stores  < 0):
  #    print(input_loads_stores)
    
  return output_loads_stores,weight_loads_stores,input_loads_stores, L1_latency*total_loads_stores*cost_per_thread,(total_loads_stores),l1_output_reloads +  l2_output_reloads,l1_weight_reloads +  l2_weight_reloads,l1_input_reloads +  l2_input_reloads,last_bound,last_loop_level








def tile_and_footprint_analysis(loop_bounds, search_harder=False, output_volume_multiplier=1):

    if loop_bounds['C'] >= VLEN:
          order = [['CCC','HH','WW','CC', 'H', 'W','K', 'R','S', 'C']]
          # ['HH','WW','CCC', 'H','CC','W','K', 'R','S', 'C']]
          #order = [['CCC','HH','WW','CC', 'H', 'W','K', 'R','S', 'C']]
    else:
          order = [['HH','WW','CC', 'H', 'W','K', 'R', 'S', 'C']]
   

    all_permutations =[]
    for j in order:
        temp = list(j)

        if temp.index('HH') > temp.index('H'):
            continue
        
        if 'CCC' in temp:  
            if temp.index('CCC') > temp.index('CC'):
                  continue
        if temp.index('CC') > temp.index('C'):
            continue
        if temp.index('WW') > temp.index('W'):
            continue



        for h in range(loop_bounds['H'], loop_bounds['R'] , -1):
            for w in range(loop_bounds['W'], loop_bounds['S'] , -1):
                #if "AVX-512" in latte.config.vec_config:      
                if w < min(14, loop_bounds['W']):
                        continue
                #if w < 14 and  loop_bounds['W'] >= 14:
                #       continue
                if h*w <=  20 and loop_bounds['H']*loop_bounds['W'] > 20:
                        continue
                if w > 14:
                        w_threshold = min(w,28)
                        h_threshold = 1
                else:
                        w_threshold = w
                        h_threshold  = min(h, math.floor(28/w_threshold))
 
                while h % h_threshold != 0:
                        h_threshold = h_threshold-1
 
                if h_threshold * w_threshold <= 20 and w_threshold != loop_bounds['W']:
                        continue

                #if loop_bounds['W']<=28 and w != loop_bounds['W']:
                #    continue
                #print("h is ")
                #print(h)
                #print("w is")
                #print(w)

                if loop_bounds['W']% w  == 0 and loop_bounds['H']% h == 0 :
                     for c in range(max(VLEN,loop_bounds['C']), VLEN - 1, -VLEN):
                      if(loop_bounds['C']//VLEN) %(c//VLEN) == 0:
                        if loop_bounds['C'] < VLEN:
                            bounds = {'N':loop_bounds['N'], 'HH':loop_bounds['H']//h,'H':h, 'WW': loop_bounds['W'] // w , 'KK':  loop_bounds['K'] // VLEN , 'W': w, 'CC': min(loop_bounds['C'],c), 'K':VLEN , 'C':loop_bounds['C'], 'R':loop_bounds['R'], 'S': loop_bounds['S'] }

                        else:
                            bounds = {'N':loop_bounds['N'], 'HH':loop_bounds['H']//h,'H':h, 'WW': loop_bounds['W'] // w , 'KK':  loop_bounds['K'] // VLEN , 'CCC' : loop_bounds['C']//c, 'W': w, 'CC': min(loop_bounds['C'],c), 'K':VLEN , 'C':VLEN , 'R':loop_bounds['R'], 'S': loop_bounds['S'] }




                        parallel_loop_bounds = {'N': loop_bounds['N'], 'K': loop_bounds['K']/VLEN , 'H': 1, 'W':1}
                        parallel_loops = ['N', 'K']
                        num_cores = 56
                        actual_loop_bounds = loop_bounds
                        tiled_loop_order = temp
                        tiled_loop_bounds = bounds

                        output_loads_stores, weight_loads_stores, input_loads_stores, total_loads_stores,t,n1,k1,h1,w1,c1 = \
                                                                                  compute_cost(actual_loop_bounds, tiled_loop_order, tiled_loop_bounds, parallel_loops, parallel_loop_bounds, num_cores, output_volume_multiplier)
                        #print(total_loads_stores)
                        #print(bounds)
                        '''

                        if  not(search_harder) and  (((h+loop_bounds['R']-1)*(w+loop_bounds['S']-1)*c)/16 + w*h  + c*loop_bounds['S']*loop_bounds['R']  > cache_lines):
                               continue
                        total_tiles = loop_bounds['N']*(loop_bounds['K']/16)/64

                        input_loads_stores = loop_bounds['W']*loop_bounds['C']*loop_bounds['H']
                        output_loads_stores = (loop_bounds['C']/c)*loop_bounds['W']*loop_bounds['H']*16
                        weight_loads_stores = (loop_bounds['H']/h)*(loop_bounds['W']/w)*16*loop_bounds['C']*loop_bounds['R']*loop_bounds['S']

                        total_loads_stores = total_tiles*(input_loads_stores + output_loads_stores + weight_loads_stores)
                        final_order  = ['N' ,'KK'] +  j
                        all_permutations.append([final_order, bounds, output_loads_stores, weight_loads_stores, input_loads_stores, total_loads_stores])
                        '''


                        final_order  = ['N' ,'KK'] +  temp
                        
                        if(total_loads_stores >= 0 and input_loads_stores >= 0 and weight_loads_stores >= 0  and  output_loads_stores >= 0):
                            all_permutations.append([final_order, bounds, output_loads_stores, weight_loads_stores, input_loads_stores, (total_loads_stores),t,n1,k1,h1,w1,c1])


                        #print("Enetered\n")

    return all_permutations


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
