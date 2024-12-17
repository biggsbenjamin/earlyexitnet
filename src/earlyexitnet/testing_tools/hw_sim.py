import torch
import functorch

import ctypes as ct
from fxpmath import Fxp

# import numpy as np
import math
import numpy as np


def quick_exp_float(val: float) -> float:
    # convert to 16 bit fixed point
    x = Fxp(val, signed=True, n_word=16, n_frac=8)

    # construct IEEE-754 32 floating point
    # use the integer part of the number as the exponent
    exp = (x.val >> 8) + (
        1 if x < 0 else 0
    )  # extract integer val and account for 2s complement
    exp += 127  # add bias
    exp &= 0xFF  # make sure it's only 8 bits
    res = exp << 23  # move exp to be in the correct position

    # print(f"{res:032b}")

    res_float = ct.cast(
        ct.pointer(ct.c_uint32(res)), ct.POINTER(ct.c_float)
    ).contents.value
    return res_float


def quick_exp_float_vec(val: np.array) -> np.array:
    # same thing as above without all the complication
    return np.power(2, np.trunc(val))


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
    dev = final_layer.device
    # simulate constructing floating point by truncating the final layer values
    # compute 2 to the power of the truncated values to obtain the approximated exponential
    # this operational is mathematically identical to extracting the integer bits
    # from the final layer and placing them in the exponent field of a floating point
    exp_zs = torch.pow(2, torch.trunc(final_layer).to(dev)).to(dev)
    return exp_zs.divide(torch.sum(exp_zs, dim=-1).unsqueeze(1)).to(dev)


def nonTrunc_base2_softmax_torch(final_layer: torch.Tensor) -> torch.Tensor:
    dev = final_layer.device
    # perform the same sequence of steps as the normal base e softmax, using base 2 instead
    exp_zs = torch.pow(2, final_layer).to(dev)
    return exp_zs.divide(torch.sum(exp_zs, dim=-1).unsqueeze(1)).to(dev)


def base2_sub_softmax_torch(final_layer: torch.Tensor) -> torch.Tensor:
    dev = final_layer.device

    max_val = torch.trunc(torch.max(final_layer))
    final_layer = torch.trunc(final_layer) - max_val
    # similar to base2_softmax_torch the trunc operation is used to simulate
    # placing the integer bits into the exponent field of a floating point
    # here the exponents (final_layer) are normalised to all be negative,
    # this way when the base is raised to the exponent the result is in the range 0-1
    exp_zs = torch.pow(2, final_layer)
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


def base2_subMax_softmax_fixed(
    final_layer: torch.Tensor,
) -> tuple[np.array, np.array]:
    # define the fixed point precision of the incoming final activation layer
    LAYER = Fxp(None, signed=True, n_word=16, n_frac=8)

    # define the fixed point precision of the exponential computation to be performed
    # the operation is returns a value in the range 0-1 so 1 bit is for the integer part
    # and the remaining bits are for the fractional part
    NUM_EXP_BITS = 28 #this would be the 28, output for exp vals
    EXP = Fxp(
        None, signed=False, n_word=NUM_EXP_BITS, n_frac=NUM_EXP_BITS - 1
    )  # sacrificing many bits

    # transfer logit back to cpu and numpy array (to work with fxp)
    zs = final_layer.cpu().numpy()

    # convert to fixed point
    fxd_zs = Fxp(zs).like(LAYER)

    num_batches = final_layer.size(dim=0)
    # find the maximum value and keep integer part
    max_z = np.trunc(np.max(fxd_zs, -1))
    # resize from [B, 1, 1] to [B, 1]
    max_z = np.reshape(
        max_z,
        (
            num_batches,
            1,
        ),
    )

    # create the array of the resulting exponential computation
    exp_zs = Fxp(np.ones(fxd_zs.shape)).like(EXP)
    # set the rounding mode to around, so rounding to nearest frac val
    #exp_zs.rounding = "around"
    # setting rounding and overflow behaviour to vitis like:
    exp_zs.overflow = 'wrap'
    exp_zs.rounding = 'around' #'floor'

    # max is already truncated so other other vals need to be too
    # extract integer part of the fixed point value
    fxd_zs = np.trunc(fxd_zs)
    # normalize so all value are negative
    fxd_zs -= max_z
    # make all values positive
    #exponents = abs(fxd_zs.get_val()) # NOTE maybe different to hw negation
    exponents = ~(fxd_zs)
    exponents = exponents + 1

    # compute the exponentiation operation by
    # shifting the ones defined earlier right by the amount defined in exponents
    # this is done by shifting the underlying representation as fxp doesn't support this operation
    #exp_zs.val >>= exponents.astype(np.uint64)
    exp_zs.val >>= exponents.get_val().astype(np.uint64)

    # FIXME bitwise ops can be done, just needs a work around
    #exp_zs = exp_zs >> exponents.astype(np.uint64)
    #shiftmask = exponents.lt(NUM_EXP_BITS)
    #exp_zs[shiftmask] = exp_zs[] >> exponents.astype(np.uint64)


    # define the variable where the sum will be accumulated
    exp_sum = Fxp(0, signed=False, n_word=36, n_frac=31)
    # compute sum
    exp_sum.equal(np.sum(exp_zs, -1))
    # reshape from [B, 1, 1] to [B, 1]
    exp_sum = np.reshape(
        exp_sum,
        (
            num_batches,
            1,
        ),
    )
    return exp_zs, exp_sum

def main():
    # print(quick_exp(-4))

    test = torch.tensor(
        [
            [
                -9.4018,
                -22.9105,
                -3.9112,
                1.6748,
                -6.1232,
                -4.9361,
                -18.7000,
                2.2737,
                1.3246,
                4.0967,
            ]
        ]
    )

    # test = np.random.default_rng().uniform(low=-128, high=128, size=10)

    # test = torch.randn(10)

    print('test logit:', test)
    torch_vec = base2_softmax_torch(test)
    print('base2 softmax using torch:', torch_vec)
    slow = base2_softmax_slow(test)
    print('slow base2 softmax', slow)
    print("difference between torch vec and slow:", torch.sub(torch_vec, slow))

    sub = base2_sub_softmax_torch(test)
    print('base 2 sub', sub)

    fixed_sub_zs_arr, fixed_sub_sum_arr = base2_subMax_softmax_fixed(test)
    ## result of base2_subMax_softmax_fixed is tuple of fxp objects
    ## requires Tensor to be compatible torch.sub
    fixed_sub_zs = torch.from_numpy(fixed_sub_zs_arr.get_val())
    fixed_sub_sum = torch.from_numpy(fixed_sub_sum_arr.get_val())
    print('fixed sub zs:', fixed_sub_zs)
    print('fixed sub sum:', fixed_sub_sum)

    print('OG torch sftmx:', torch.softmax(test, dim=-1))
    diff1 = torch.sub(torch_vec, torch.softmax(test, dim=-1))
    diff2 = torch.sub(slow, torch.softmax(test, dim=-1))
    diff3 = torch.sub(sub, torch.softmax(test, dim=-1))
    print("difference quick:", diff1, torch.norm(diff1))
    print("difference slow:", diff2, torch.norm(diff2))
    print("difference sub:", diff3, torch.norm(diff3))
    print('converting values to fxp, calculating exp and sum, returning to float, performing division')
    print('So bit accurate up to the division (which would be a multiplication for the hw)')
    diff4 = torch.sub(torch.div(fixed_sub_zs, fixed_sub_sum), torch.softmax(test, dim=-1))
    print("difference fixed sub:", diff4, torch.norm(diff4))

    #print(baseE_subMax_softmax_float(test))

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
