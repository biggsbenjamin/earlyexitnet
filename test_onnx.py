'''
Testing the onnx lib with pytorch branchynet early exit model.

- Saving a model to onnx
- Running a loaded onnx model against same pytorch model


- Check what the differences are with diff batch sizes
- Loading a model from pytorch and changing to onnx
'''

from models.Branchynet import Branchynet, ConvPoolAc
from models.Lenet import Lenet
from models.Testnet import Testnet

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

import onnx

def to_onnx(model, input_size, batch_size=1,
        path='outputs/onnx', name='brn.onnx', speedy=False, test_in=None):
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

    if test_in is None:
        x = torch.randn(batch_size, *input_size)
    else:
        x=test_in

    #trying the scripty thing
    scr_model = torch.jit.script(model)
    print(scr_model.graph)
    ex_out = scr_model(x)

    torch.onnx.export(
        scr_model,      # model being run
        x,              # model input (or a tuple for multiple inputs)
        sv_pnt,         # where to save the model (can be a file or file-like object)
        export_params=True, # store the trained parameter weights inside the model file
        opset_version=12,          # the ONNX version to export the model to
        do_constant_folding=True,  # t/f execute constant folding for optimization
        example_outputs=ex_out,
        input_names = ['input'],   # the model's input names
        output_names = ['exit'],#, 'eeF'], # the model's output names
        #dynamic_axes={#'input' : {0 : 'batch_size'},    # variable length axes
                      #'exit' : {0 : 'exit_size'}#,
                      #'eeF' : {0 : 'exit_size'}
                      #}
    )
    return sv_pnt

def brn_main():
    bs = 1
    shape = [1,28,28]
    #set up model
    model = Branchynet(fast_inf_batch_size=bs, exit_threshold=0.1)

    md_pth = '/home/benubu/phd/pytorch_play/earlyexitnet/outputs/\
pre_Trn_bb_2021-03-03_133905/pretrn-joint-2021-03-03_140528.pth'
    checkpoint = torch.load(md_pth)
    #model.load_state_dict(checkpoint['model_state_dict'])


    #fast inf pytorch
    model.set_fast_inf_mode()
    print("Model done")

    #feed same input to both
    test_x = torch.ones(1, *shape)#torch.randn(1, *shape)

    import torchvision
    tfs = transforms.Compose([
        transforms.ToTensor()
        ])
    mnist_dl = DataLoader( torchvision.datasets.MNIST('../data/mnist',
                                    download=True, train=False, transform=tfs),
                batch_size=1, drop_last=True, shuffle=False)

    mnistiter = iter(mnist_dl)
    xb, yb = mnistiter.next()

    combi = torch.cat((test_x, test_x, xb), 0)
    #print(combi)


    print("STARTING RUN")
    output = model(xb)
    output2 = model(test_x)
    print("PT OUT:", output)
    #print("SPACING")
    #for i in output:
    #    print(i)

    print("PT OUT2:", output2)


    #'''
    #save to onnx
    print("SAVING")
    save_path = to_onnx(model, shape, batch_size=bs, speedy=True, name='brn-top1ee-bsf-trnInc-sftmx.onnx', test_in=xb)
    print("SAVED")

    #load from onnx
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    print("IMPORTED")

    #onnx runtime model
    import onnxruntime
    ort_session = onnxruntime.InferenceSession(save_path)
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else \
            tensor.cpu().numpy()

    print("RUNNING ONNX")
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(xb)}
    ort_outs = ort_session.run(None, ort_inputs)

    print("ONNX_OUT", ort_outs)
    #print("SPACING")
    #for i in ort_outs:
    #    print(i)
    ort_inputs2 = {ort_session.get_inputs()[0].name: to_numpy(test_x)}
    ort_outs2 = ort_session.run(None, ort_inputs2)

    print("ONNX_OUT2", ort_outs2)


    # compare ONNX Runtime and PyTorch results
    #outlist=[]
    #for out in output:
        #olist=[]
        #for o in out:
        #    if isinstance(o, torch.Tensor):
        #        olist.append(to_numpy(o))
        #outlist.append(olist)

    np.testing.assert_allclose(to_numpy(output),
        ort_outs[0], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(to_numpy(output2),
        ort_outs2[0], rtol=1e-03, atol=1e-05)
    #'''

def lenet_main():
    bs = 64
    shape = [1,28,28]
    #set up model
    #model = Lenet()
    model = Testnet()

    model.eval()
    print("Model done")

    #feed same input to both
    test_x = torch.ones(1, *shape)#torch.randn(1, *shape)

    import torchvision
    tfs = transforms.Compose([
        transforms.ToTensor()
        ])
    mnist_dl = DataLoader( torchvision.datasets.MNIST('../data/mnist',
                                    download=True, train=False, transform=tfs),
                batch_size=1, drop_last=True, shuffle=False)

    mnistiter = iter(mnist_dl)
    xb, yb = mnistiter.next()

    combi = torch.cat((test_x, test_x, xb), 0)


    print("STARTING RUN")
    output = model(xb)
    output2 = model(test_x)
    print("PT OUT:", output)
    #print("SPACING")
    #for i in output:
    #    print(i)

    print("PT OUT2:", output2)


    #'''
    #save to onnx
    print("SAVING")

    path='outputs/onnx'
    fname='pt_fulltest.onnx'
    sv_pnt = os.path.join(path, fname)
    if not os.path.exists(path):
        os.makedirs(path)

    x = torch.randn(bs, *shape)

    #trying the scripty thing
    #scr_model = torch.jit.script(model)
    #print(scr_model.graph)
    #ex_out = scr_model(x)

    torch.onnx.export(
        #scr_model,      # model being run
        model,
        x,              # model input (or a tuple for multiple inputs)
        sv_pnt,         # where to save the model (can be a file or file-like object)
        export_params=True, # store the trained parameter weights inside the model file
        opset_version=12,          # the ONNX version to export the model to
        do_constant_folding=True,  # t/f execute constant folding for optimization
        #example_outputs=ex_out,
        input_names = ['lenet_in'],   # the model's input names
        output_names = ['lenet_out'],#, 'eeF'], # the model's output names
        dynamic_axes={'lenet_in' : {0 : 'batch_size'},    # variable length axes
                      'lenet_out' : {0 : 'exit_size'}
                      })

    print("SAVED")

    #load from onnx
    import onnx
    #onnx_model_ni = onnx.load(sv_pnt)
    onnx.checker.check_model(onnx_model_ni)
    print("IMPORTED")

    #work around since no automated inference in export stage
    #add_value_info_for_constants(onnx_model_ni)
    #inferred_model = onnx.shape_inference.infer_shapes(onnx_model_ni)
    #fnm, ext = os.path.splitext(sv_pnt)
    #infer_sv_pnt = fnm+'_infer'+ext
    #onnx.save_model(inferred_model, infer_sv_pnt)
    #print("RE-SAVED")

    #onnx_model = onnx.load(infer_sv_pnt)
    #onnx.checker.check_model(onnx_model)
    #print("IMPORTED AGAIN")
    onnx_model = onnx_model_ni


    #onnx runtime model
    import onnxruntime
    ort_session = onnxruntime.InferenceSession(sv_pnt)
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else \
            tensor.cpu().numpy()

    print("RUNNING ONNX")
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(xb)}
    ort_outs = ort_session.run(None, ort_inputs)

    print("ONNX_OUT", ort_outs)
    #print("SPACING")
    #for i in ort_outs:
    #    print(i)
    ort_inputs2 = {ort_session.get_inputs()[0].name: to_numpy(test_x)}
    ort_outs2 = ort_session.run(None, ort_inputs2)

    print("ONNX_OUT2", ort_outs2)


    # compare ONNX Runtime and PyTorch results
    #outlist=[]
    #for out in output:
        #olist=[]
        #for o in out:
        #    if isinstance(o, torch.Tensor):
        #        olist.append(to_numpy(o))
        #outlist.append(olist)

    np.testing.assert_allclose(to_numpy(output),
        ort_outs[0], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(to_numpy(output2),
        ort_outs2[0], rtol=1e-03, atol=1e-05)
    #'''


def main():
    brn_main()
    #lenet_main()

if __name__ == "__main__":
    main()
