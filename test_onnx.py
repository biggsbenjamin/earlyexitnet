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

def to_onnx(model, input_size, batch_size=1,
        path='outputs/onnx', name='brn.onnx', speedy=False):
    #convert the model to onnx format - trial with onnx lib
    if speedy:
        fname = 'speedy-'+name
        model.set_fast_inf_mode()
    else:
        fname = 'slow-'+name
        model.eval()

    sv_pnt = os.path.join(path, fname)
    if not os.path.exists(path):
        os.makedirs(path)

    x = torch.randn(batch_size, *input_size)

    torch.onnx.export(
        model,          # model being run
        x,              # model input (or a tuple for multiple inputs)
        sv_pnt,         # where to save the model (can be a file or file-like object)
        export_params=True, # store the trained parameter weights inside the model file
        opset_version=10,          # the ONNX version to export the model to
        do_constant_folding=True,  # t/f execute constant folding for optimization
        input_names = ['input'],   # the model's input names
        output_names = ['ee1', 'eeF'], # the model's output names
        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                      'ee1' : {0 : 'exit_size'},
                      'eeF' : {0 : 'exit_size'}
                      })
    return sv_pnt

def main():
    #set up model
    model = Branchynet(exit_threshold=0.1)
    print("Model done")
    bs = 1
    shape = [1,28,28]

    #save to onnx
    save_path = to_onnx(model, shape, batch_size=bs)

    #load from onnx
    import onnx
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)

    #feed same input to both
    test_x = torch.randn(bs, *shape)

    #fast inf pytorch
    model.set_fast_inf_mode()

    output = model(test_x)
    print("RAW OUT:", output)

    for i in output:
        print("out:", i)



    #onnx runtime model
    import onnxruntime
    ort_session = onnxruntime.InferenceSession(save_path)
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else \
            tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(test_x)}
    ort_outs = ort_session.run(None, ort_inputs)

    print("ort_outs", ort_outs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose([to_numpy(out) for out in output],
        ort_outs, rtol=1e-03, atol=1e-05)

if __name__ == "__main__":
    main()
