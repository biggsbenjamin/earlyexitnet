import torch

import ctypes as ct
from fxpmath import Fxp
# import numpy as np
import math

def quick_exp_float(val : float) -> float:
  # convert to 16 bit fixed point
  x = Fxp(val, signed=True, n_word=16, n_frac=8)
  
  # construct IEEE-754 32 floating point
  # use the integer part of the number as the exponent
  exp = (x.val >> 8) & 0xFF # extract integer val
  exp += 127 # add bias
  exp &= 0xFF # make sure it's only 8 bits
  res = exp << 23 # move exp to be in the correct position
  
  # print(f"{res:032b}")
  
  res_float = ct.cast(ct.pointer(ct.c_uint32(res)), ct.POINTER(ct.c_float)).contents.value
  return res_float

# take the final layer and a threshold and find out if branching can happen
def base2_softmax(final_layer: torch.Tensor) -> list[float]: 
  # import pdb;pdb.set_trace()
  zs = final_layer.squeeze().tolist()
  
  exp_zs = []
  exp_sum = 0
  for z in zs:
    e_z = quick_exp_float(z)
    exp_sum += e_z
    exp_zs.append(e_z)
    
  exp_zs = [z / exp_sum for z in exp_zs]
  
  return exp_zs

def baseE_subMax_softmax_float(final_layer: torch.Tensor) -> list[float]: 
  zs = final_layer.squeeze().tolist()
  
  exp_zs = []
  exp_sum = 0
  max_z = max(zs)
  for z in zs:
    e_z = math.exp(z - max_z)
    print(e_z)
    exp_sum += e_z
    exp_zs.append(e_z)
    
  exp_zs = [z / exp_sum for z in exp_zs]
  
  return exp_zs

def base2_subMax_softmax_fixed(final_layer: torch.Tensor) -> list[float]: 
  LAYER = Fxp(None, signed=True, n_word=16, n_frac=8)
  EXP   = Fxp(None, signed=False, n_word=16, n_frac=14)
  
  zs = final_layer.squeeze().tolist()
  # convert to fixed point
  fxd_zs = Fxp(zs).like(LAYER)
  
  max_z = Fxp().like(LAYER)
  max_z = max(fxd_zs)  
  
  exp_zs = Fxp([]).like(EXP)
  exp_sum = Fxp(0).like(EXP)
  
  for i, z in enumerate(fxd_zs):
    e_z = Fxp().like(EXP)
    e_z.equal(2 ** z) # compute two to the power z
    exp_sum += e_z
    exp_zs.append(e_z)
    
  exp_zs = [z / exp_sum for z in exp_zs]
  
  return exp_zs

def main():
  
  # print(quick_exp(-4))
  
  test = torch.tensor([ -9.4018, -22.9105,  -3.9112,   1.6748,  -6.1232,  -4.9361, -18.7000, 2.2737,   1.3246,   4.0967])
  
  print(base2_softmax(test))
  print(torch.softmax(test,dim=-1))
  print(baseE_subMax_softmax_float(test))
  # y = 5.75
  
  # x = Fxp(y, signed=True, n_word=16, n_frac=8)
  
  # print(x.n_frac, x.n_int, x.upper, x.lower)
  # print(x.get_val(), type(x.get_val()), x.bin(), type(x.bin()))
  # print(x.info(verbose=3))
  # bits1 = ct.cast(ct.pointer(ct.c_float(x.get_val())), ct.POINTER(ct.c_uint32)).contents.value
  # bits2 = ct.cast(ct.pointer(ct.c_float(y)), ct.POINTER(ct.c_uint32)).contents.value

  # print(f"{bits1:032b}")
  # print(f"{bits2:032b}")
  # print(x.val, type(x.val), f"{x.val:032b}")
  
  # print(np.binary_repr(int(5.75), width=16), type(np.binary_repr(int(5.75), width=16)))

  # tensor([[ -9.4018, -22.9105,  -3.9112,   1.6748,  -6.1232,  -4.9361, -18.7000,
  #          2.2737,   1.3246,   4.0967]], device='cuda:0'), tensor([[-20.4055, -46.4818, -14.8476,  22.9752,  -5.0832,  -2.9427, -39.8003,
  #          5.7668,  26.4024,  32.4166]], device='cuda:0')]

if __name__ == "__main__":
  main()