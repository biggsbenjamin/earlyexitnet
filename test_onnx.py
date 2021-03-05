'''
Testing the onnx lib with pytorch branchynet early exit model.

- Saving a model to onnx
- Running a loaded onnx model against same pytorch model


- Check what the differences are with diff batch sizes
- Loading a model from pytorch and changing to onnx
'''

from models.Branchynet import Branchynet, ConvPoolAc
from tools import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
import torchvision.transforms as transforms

import os
import numpy as np
from datetime import datetime as dt

import io
import torch.onnx

def to_onnx(model, input_size, batch_size=1, path='outputs/onnx'):
    #convert the model to onnx format - trial with onnx lib

    sv_pnt = os.path.join(path, 'brn-not_speedy_inf.onnx')
    if not os.path.exists(path):
        os.makedirs(path)

    x = torch.randn(batch_size, *input_size)

    model.eval()

    torch.onnx.export(
        model,              # model being run
        x,                  # model input (or a tuple for multiple inputs)
        sv_pnt,             # where to save the model (can be a file or file-like object)
        export_params=True, # store the trained parameter weights inside the model file
        opset_version=10,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['input'],   # the model's input names
        output_names = ['output'], # the model's output names
        dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                      'output' : {0 : 'batch_size'}})

def main():
    #set up model
    model = Branchynet()
    print("Model done")

    #save to onnx
    to_onnx(model, [1,28,28])

    #load from onnx
    #feed same input to both

if __name__ == "__main__":
    main()
