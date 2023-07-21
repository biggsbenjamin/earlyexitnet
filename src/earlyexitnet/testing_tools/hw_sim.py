import torch
import functorch

import ctypes as ct
from fxpmath import Fxp
# import numpy as np
import math
import numpy as np


def quick_exp_float(val : float) -> float:
  # convert to 16 bit fixed point
  x = Fxp(val, signed=True, n_word=16, n_frac=8)
  
  # construct IEEE-754 32 floating point
  # use the integer part of the number as the exponent
  exp = (x.val >> 8) + (1 if x < 0 else 0) # extract integer val and account for 2s complement
  exp += 127 # add bias
  exp &= 0xFF # make sure it's only 8 bits
  res = exp << 23 # move exp to be in the correct position
  
  # print(f"{res:032b}")
  
  res_float = ct.cast(ct.pointer(ct.c_uint32(res)), ct.POINTER(ct.c_float)).contents.value
  return res_float

def quick_exp_float_vec(val : np.array) -> np.array:
  # same thing as above without all the complication
  return np.power(2,np.trunc(val))

def base2_softmax_slow(final_layer: torch.Tensor) -> torch.Tensor: 
  # import pdb;pdb.set_trace()
  zs = final_layer.squeeze().tolist()
  
  exp_zs = []
  exp_sum = 0
  for z in zs:
    e_z = quick_exp_float(z)
    exp_sum += e_z
    exp_zs.append(e_z)
  # print(exp_zs)
  exp_zs = [z / exp_sum for z in exp_zs]
  
  return torch.Tensor(exp_zs)

# take the final layer and a threshold and find out if branching can happen
def base2_softmax_torch(final_layer: torch.Tensor) -> torch.Tensor: 
  # import pdb;pdb.set_trace()
  dev = final_layer.device
  exp_zs = torch.pow(2,torch.trunc(final_layer).to(dev)).to(dev)
  # print(exp_zs)
  return exp_zs.divide(torch.sum(exp_zs, dim=-1).unsqueeze(1)).to(dev)

def nonTrunc_base2_softmax_torch(final_layer: torch.Tensor) -> torch.Tensor: 
  # import pdb;pdb.set_trace()
  dev = final_layer.device
  exp_zs = torch.pow(2,final_layer).to(dev)
  # print(exp_zs)
  return exp_zs.divide(torch.sum(exp_zs, dim=-1).unsqueeze(1)).to(dev)


def base2_sub_softmax_torch(final_layer: torch.Tensor) -> torch.Tensor:
  dev = final_layer.device
  
  max_val = torch.trunc(torch.max(final_layer))
  final_layer = torch.trunc(final_layer) - max_val
  
  exp_zs = torch.pow(2,final_layer)
  return exp_zs.divide(torch.sum(exp_zs)).to(dev)
  
  

def baseE_subMax_softmax_float(final_layer: torch.Tensor) -> list[float]: 
  zs = final_layer.squeeze().tolist()
  
  exp_zs = []
  exp_sum = 0
  max_z = max(zs)
  for z in zs:
    e_z = math.exp(z - max_z)
    # print(e_z)
    exp_sum += e_z
    exp_zs.append(e_z)
    
  exp_zs = [z / exp_sum for z in exp_zs]
  
  return exp_zs

def base2_subMax_softmax_fixed(final_layer: torch.Tensor) -> tuple[np.array, np.array]: 
  NUM_EXP_BITS = 16
  
  LAYER = Fxp(None, signed=True, n_word=16, n_frac=8)
  EXP   = Fxp(None, signed=False, n_word=NUM_EXP_BITS, n_frac=NUM_EXP_BITS-1) # sacrificing many bits 
  
  
  
  zs = final_layer.cpu().numpy()
  
  # convert to fixed point
  fxd_zs = Fxp(zs).like(LAYER)
  
  # max_z = Fxp().like(LAYER)
  num_batches = final_layer.size(dim=0)
  max_z = np.trunc(np.max(fxd_zs,-1))
  max_z = np.reshape(max_z, (num_batches,1,))
  
  exp_zs = Fxp(np.ones(fxd_zs.shape)).like(EXP)
  exp_zs.rounding= 'around'
  # exp_sum = Fxp(0).like(EXP)
  fxd_zs = np.trunc(fxd_zs) # use only integer part
  fxd_zs -= max_z  # now all elements are 0 or negative
  exponents = abs(fxd_zs.get_val())  # make all values positive
  
  # shift the underlying representation of the numbers
  # in this way the exponentiation is computer
  exp_zs.val >>= exponents.astype(np.uint64)   
  
  exp_sum = Fxp(0, signed=False, n_word=36, n_frac=31) 
  exp_sum.equal(np.sum(exp_zs,-1))
  exp_sum = np.reshape(exp_sum, (num_batches,1,))
  
  return exp_zs, exp_sum

def main():
  
  # print(quick_exp(-4))
  
  test = torch.tensor([[ -9.4018, -22.9105,  -3.9112,   1.6748,  -6.1232,  -4.9361, -18.7000, 2.2737,   1.3246,   4.0967]])
  
  # test = np.random.default_rng().uniform(low=-128, high=128, size=10)
  
  # test = torch.randn(10)
  
  print(test)
  torch_vec = base2_softmax_torch(test)
  print(torch_vec)
  slow =base2_softmax_slow(test) 
  print(slow)
  
  sub = base2_sub_softmax_torch(test)
  print(sub)
  
  fixed_sub = base2_subMax_softmax_fixed(test)
  print(fixed_sub)
  print("difference:", torch.sub(torch_vec,slow))
  print(torch.softmax(test,dim=-1))
  diff1 = torch.sub(torch_vec,torch.softmax(test,dim=-1))
  diff2 = torch.sub(slow,torch.softmax(test,dim=-1))
  diff3 = torch.sub(sub,torch.softmax(test,dim=-1))
  diff4 = torch.sub(fixed_sub, torch.softmax(test,dim=-1))
  print("difference quick:", diff1, torch.norm(diff1))
  print("difference slow:", diff2, torch.norm(diff2))  
  print("difference sub:", diff3, torch.norm(diff3))  
  print("difference fixed sub:", diff4, torch.norm(diff4))  
  
  
  print(baseE_subMax_softmax_float(test))
  
  # for i in range(100):
  #   test = np.random.default_rng().uniform(low=-128, high=128, size=10)
  #   vec1 = quick_exp_float_vec(test)
  #   vec2 = np.array([quick_exp_float(x) for x in test])
  #   if (np.array_equal(vec1, vec2)):
  #     print("Different!", i)
  #     print(vec1)
  #     print(vec2)
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