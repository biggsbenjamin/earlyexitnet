"""
Methods to do things with onnx models into pytorch land
"""

import onnx
import onnxruntime
from onnx import numpy_helper
from onnx import version_converter, helper

import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

from earlyexitnet.tools import path_check,save_model,CIFAR10DataColl
from earlyexitnet.models.ResNet8 import ResNet8

"""
Method to export pytorch models to onnx.

NOTE - This method goes via pytorch script to avoid
constant propagation removing dynamic inference.
"""
def to_onnx(model,
            input_size, # shape of input e.g. [1,28,28]
            batch_size=1, # size of batch input
            path='outputs/onnx', # output path to save to
            fname='brn.onnx', # filename of onnx op
            test_in=None): # pytorch tensor input to put thru exporter
    #convert the model to onnx format - trial with onnx lib
    # making sure in evaluation mode
    model.eval()

    sv_pnt = os.path.join(path, fname)
    if not os.path.exists(path):
        os.makedirs(path)

    if test_in is None:
        x = torch.randn(batch_size, *input_size)
    else:
        x=test_in

    # Just In Time compilation of pytorch model
    # to script Intermediate Representation
    scr_model = torch.jit.script(model)
    print("PRINTING PYTORCH MODEL SCRIPT")
    print(scr_model.graph, "\n")
    ex_out = scr_model(x) # get output of script model

    torch.onnx.export(
        scr_model,      # model being run
        x,              # model input (or a tuple for multiple inputs)
        sv_pnt,         # where to save the model (can be a file or file-like object)
        export_params=True, # store the trained parameter weights inside the model file
        opset_version=12,          # the ONNX version to export the model to
        do_constant_folding=True,  # t/f execute constant folding for optimization
        input_names = ['input'],   # the model's input names
        output_names = ['exit'],#, 'eeF'], # the model's output names
        #dynamic_axes={#'input' : {0 : 'batch_size'}, # NOTE not used, variable length axes
                      #'exit' : {0 : 'exit_size'}#,
                      #'eeF' : {0 : 'exit_size'}
                      #}
    )
    return sv_pnt

def run_onnx(onnx_path, input_tensor):
    # transform tensor to np array
    np_arr = input_tensor.detach().cpu().numpy() \
        if input_tensor.requires_grad else \
        input_tensor.cpu().numpy()
    # set up onnx runtime session
    ort_session = onnxruntime.InferenceSession(onnx_path)
    # load inputs
    ort_inputs = {ort_session.get_inputs()[0].name: np_arr}
    # run inference on inputs
    ort_outs = ort_session.run(None, ort_inputs)
    print("ONNX_OUT", ort_outs)
    # return output for comparison
    return ort_outs

def extract_params(node, init_dict):
    inits = [init_dict[input_name] for input_name in node.input if input_name in init_dict]
    init_len = len(inits)
    if init_len == 1:
        w = torch.from_numpy(inits[0])
        b = None
    elif init_len == 2:
        w = torch.from_numpy(inits[0])
        b = torch.from_numpy(inits[1])
    else:
        raise ValueError("Number of parameters exceeds weights and biases")
    if node.op_type == 'Gemm':
        w = w.t()
    return w,b

# load onnx model into pytorch version of model
# not trivial to automate this...
def onnx_param_import_resnet8(args):
    onnx_model = onnx.load(args.onnx_model_path)
    onnx_model_name = os.path.splitext(os.path.basename(args.onnx_model_path))[0]
    initializers = onnx_model.graph.initializer
    nodes = onnx_model.graph.node

    # dict for initializers from onnx, linked to name
    onnx_wbs = {}
    for init in initializers:
        w = numpy_helper.to_array(init)
        # creating writeable copy
        w_copy = np.copy(w)
        onnx_wbs[init.name] = w_copy

    for n in nodes:
        inits = [onnx_wbs[input_name] for input_name in n.input if input_name in onnx_wbs]
        print(f"node name:{n.name} w&b:{len(inits)} type:{n.op_type}")

    # set parameterised layers
    wb_layer_types=['Conv','Gemm','MatMul']
    #
    onnx_wb_layers = [n for n in nodes if n.op_type in wb_layer_types]

    pt_model=ResNet8() # model of same structure
    # get real test input
    test_dl=CIFAR10DataColl(batch_size_test=1,shuffle=True).get_test_dl()
    dl_iter = iter(test_dl)
    x,y_gnd= next(dl_iter)

    # Input values for ml perf resnet8 are NOT scaled down from 0,255
    x=x*255

    print("what numbers?:",x[0][0])

    f,ax = plt.subplots(2,2)
    for m in range(2):
        for n in range(2):
            #
            a,b = next(dl_iter)
            ax[m,n].imshow(a[0].permute(1,2,0))
            ax[m,n].get_xaxis().set_visible(False)
            ax[m,n].get_yaxis().set_visible(False)
    f.subplots_adjust(hspace=0.1)
    f.subplots_adjust(wspace=0)
    plt.show()

    #y = pt_model(x)
    #print(f"pre-load result: {y}")

    if len(onnx_wb_layers) != len(pt_model.wb_list):
        raise ValueError("Mismatch in parameterised layers between onnx and pt")

    count=0
    for pt_l, onnx_l in zip(pt_model.wb_list, onnx_wb_layers):
        # layers that have weights and biases
        weights, bias = extract_params(onnx_l, onnx_wbs)
        if pt_l.weight.data.size() != weights.size():
            raise ValueError(f"Dimension Mismatch. pt size:{pt_l.weight.data.size()} onnx size:{weights.size()}")
        # set weights
        pt_l.weight.data = weights
        if bias is not None:
            pt_l.bias.data = bias
        # check if model still runs
        count+=1

    y = pt_model(x)
    print(f"post-load result: {y}")

    # saving pytorch model
    save_model(pt_model,args.output_path,file_prefix=f'onnx-import-{onnx_model_name}_ResNet8')

    # outputting onnx model to check against original
    new_onnx_path = to_onnx(pt_model,
            [3,32,32], # shape of input e.g. [1,28,28]
            batch_size=1, # size of batch input
            path=args.output_path, # output path to save to
            fname=f'onnx-import-{onnx_model_name}_ResNet8-VERIF_ONNX.onnx', # filename of onnx op
            test_in=None) # pytorch tensor input to put thru exporter

    # running original onnx
    # reload model
    og_model = onnx.load(args.onnx_model_path)
    for imp in og_model.opset_import:
        if imp.domain == "":
            imp.version = 13
            break
    print(og_model.opset_import)

    # convert to op set 13
    #og_onnx13 = version_converter.convert_version(og_model, 12)
    print("model converted")
    # save the model ffs
    new_og_path = os.path.join(args.output_path, "og_onnx_tmp.onnx")
    onnx.save_model(og_model,new_og_path)
    og_res = run_onnx(new_og_path, x)[0]
    print("og model:",np.argmax(og_res, axis=1))

    # running new onnx
    new_res = run_onnx(new_onnx_path, x)
    new_res_tensor = torch.from_numpy(new_res[0][0])
    sm = torch.nn.functional.softmax(new_res_tensor,dim=1)
    # comparing onnx values (first just doing vis inspection)
    print(sm)
    print("new model:",np.argmax(sm, axis=1))

    print("TARGET",y_gnd)

    return

def main():
    parser = argparse.ArgumentParser(description="ONNX CLI")

    parser.add_argument('-omp','--onnx_model_path',metavar='PATH',type=path_check,
            help='Path to onnx model to load, the same type as model name')
    parser.add_argument('-op','--output_path',metavar='PATH',type=str,
            help='Path to output directory to save pytorch model')
    # parse the arguments
    args = parser.parse_args()

    # run importing for onnx parameters
    onnx_param_import_resnet8(args)

if __name__ == "__main__":
    main()
