# TODO refactor

'''
!!!OLD!!!

Testing the onnx lib with pytorch branchynet early exit model.

- Saving a model to onnx
- Running a loaded onnx model against same pytorch model


- Check what the differences are with diff batch sizes
- Loading a model from pytorch and changing to onnx
'''

#importing pytorch models to test
from models.Branchynet import B_Lenet, B_Lenet_fcn, B_Lenet_se, ConvPoolAc, B_Alexnet_cifar, TW_SmallCNN, C_Alexnet_SVHN
from models.Lenet import Lenet
from models.Testnet import Testnet, BrnFirstExit, BrnSecondExit, BrnFirstExit_se, Backbone_se, Backbone_Alex, TW_BB_SmallCNN, SDN_BB_ResNet
#from main import pull_mnist_data, train_backbone

import torch
import torch.nn as nn
import torch.optim as optim

#to get inputs for testing
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
import torchvision.transforms as transforms

import os
import numpy as np
from datetime import datetime as dt
import argparse

#to test onnx methods
import io
import torch.onnx

import onnx
import onnxruntime

from PIL import Image, ImageOps

from earlyexitnet.onnx_tools.onnx_helpers import to_onnx


#'/home/localadmin/phd/earlyexitnet/outputs/pre_Trn_bb_2021-07-09_141616/pretrn-joint-8-2021-07-09_142311.pth'
#'brn-top1ee-bsf-lessOps-trained.onnx'

def brn_main(md_pth, save_name):

    print("Running BranchyNet Test")
    e_thr = 0.996
    bs = 1
    shape = [1,28,28]
    #set up model
    #model = B_Lenet(fast_inf_batch_size=bs, exit_threshold=0.1)
    #model = B_Lenet(exit_threshold=0.9)
    #model = B_Lenet_se(exit_threshold=0.996)
    model = B_Lenet_se(exit_threshold=e_thr)
    print(f"Using model b lenet se, thr={e_thr}")

    checkpoint = torch.load(md_pth)
    model.load_state_dict(checkpoint['model_state_dict'])
    if save_name[-5:] != '.onnx':
        save_name += '.onnx'

    #fast inf pytorch
    model.set_fast_inf_mode()
    print("Finished loading model parameters")

    #generate input
    #test_x = torch.ones(1, *shape)
    #test_x = torch.randn(1, *shape)

    #pull real input example from MNIST data set
    tfs = transforms.Compose([
        transforms.ToTensor()
        ])

    mnist_dl = DataLoader( torchvision.datasets.MNIST('../data/mnist',
                                    download=True, train=False, transform=tfs),
                batch_size=1, drop_last=True, shuffle=False)

    mnistiter = iter(mnist_dl)
    xb, yb = mnistiter.next() #MNIST example stored in xb

    #torchvision.utils.save_image(xb, 'dat_img.png')
    #print("SHAPE",xb.shape)
    #torchvision.utils.save_image(test_x, 'random_input_example.png') #comment until required

    tx_raw = np.array(ImageOps.grayscale(Image.open(
        'data_test/1/random_input_example.png')),dtype=np.float32)

    if len(tx_raw.shape) == 2:
        tx_raw = np.expand_dims(tx_raw,axis=0)
    tx_raw = np.expand_dims(tx_raw,axis=0)
    tx_raw = tx_raw / np.amax(tx_raw)
    test_x = torch.from_numpy(tx_raw)

    ie_raw = np.array(ImageOps.grayscale(Image.open(
        'data_test/1/input_example.png')),dtype=np.float32)

    if len(ie_raw.shape) == 2:
        ie_raw = np.expand_dims(ie_raw,axis=0)
    ie_raw = np.expand_dims(ie_raw,axis=0)
    ie_ten = torch.from_numpy(ie_raw)

    #ie = torchvision.datasets.ImageFolder('helpme', transform=tfs)
    #iedl=DataLoader(ie, batch_size=1)
    #testiter = iter(iedl)
    #xie, yie = testiter.next()

    combi = torch.cat((test_x, test_x, xb), 0) #NOTE currently not used, combines into batch
    #print(combi)

    #print("STARTING RUN OF PYTORCH MODEL WITH INPUTS")
    #output = model(xb)
    #print("PT OUT:", output)
    #output2 = model(test_x)
    #print("PT OUT2:", output2)
    #output3 = model(ie_ten) #model(xie)
    #print("PT OUT3:", output3)

    ###         #print(torch.max(ie_ten), torch.max(xb))
    ###         def load_inputs(self,filepath):
    ###         img_ls = os.listdir(filepath)
    ###         img_ls.sort()
    ###
    ###         np_ls=[]
    ###         for s in range(self.partition.batch_size):
    ###         current_path = os.path.join(filepath,img_ls[s])
    ###         # load in the numpy array
    ###         #img = np.array(Image.open(current_path),dtype=np.float32)
    ###         img = np.load(current_path)
    ###         # scale images
    ###         data_max = np.amax(img)
    ###         img = img / data_max
    ###         if len(img.shape) == 2:
    ###         img = np.expand_dims(img,axis=0)
    ###         np_ls.append(img)
    ###         self.data = np.concatenate(np_ls, axis=0 )
    ###         print("Input data shape:",self.data.shape)

    ### TESTING THE IMAGES USED FOR BOARD ###
    #npy_path = "../fpgaconvnet-hls/test/partitions/ee_80rsc/IMAGES/"
    #bs=64
    npy_path = "./IMAGES_1024e/"
    bs = 1024

    img_ls = os.listdir(npy_path)
    img_ls.sort()

    output4 = []
    np_ls = []
    offset = 0
    for samp_idx in range(offset, bs+offset):

        current_path = os.path.join(npy_path,img_ls[samp_idx])
        img = np.load(current_path)
        print(f"image {samp_idx} loaded from : {current_path}")
        data_max = np.amax(img)
        img = img / data_max
        if len(img.shape) == 2:
            img = np.expand_dims(img,axis=0)
        np_ls.append(img)

        # convert from np to pyt
        test_img = torch.from_numpy(img)

        #print(f"Running img")
        tmp = model(test_img)
        output4.append(tmp)
        #print(f"Network run")

    #print("PT OUT4:", output4)
    ### TESTING THE IMAGES USED FOR BOARD ###

    print("LE Count:", model.le_cnt)

    print("Returning without saving anything.")

    #FIXME remove when done playing with test sets
    return

    #save to onnx
    print("SAVING MODEL TO ONNX: ", save_name)
    save_path = to_onnx(model, shape, batch_size=bs, fname=save_name, test_in=test_x)
    print("SAVED TO: ",save_path)

    #load from onnx
    print("IMPORTING MODEL FROM ONNX for comparison")

    #onnx_model = onnx.load(save_path)
    #onnx.checker.check_model(onnx_model) #running model checker
    #TODO add more onnx checks
    #print("IMPORTED")

    #onnx runtime model

    ort_session = onnxruntime.InferenceSession(save_path) #start onnx runtime session
    def to_numpy(tensor): #function to convert tensor to numpy format
        return tensor.detach().cpu().numpy() if tensor.requires_grad else \
            tensor.cpu().numpy()

    print("RUNNING ONNX")
    # compute ONNX Runtime (ort) output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(xb)}
    ort_outs = ort_session.run(None, ort_inputs)
    print("ONNX_OUT", ort_outs)

    ort_inputs2 = {ort_session.get_inputs()[0].name: to_numpy(test_x)}
    ort_outs2 = ort_session.run(None, ort_inputs2)
    print("ONNX_OUT2", ort_outs2)

    ort_inputs3 = {ort_session.get_inputs()[0].name: to_numpy(ie_ten)}
    ort_outs3 = ort_session.run(None, ort_inputs3)
    print("ONNX_OUT3", ort_outs3)

    # compare ONNX Runtime and PyTorch results
    #outlist=[]
    #for out in output:
        #olist=[]
        #for o in out:
        #    if isinstance(o, torch.Tensor):
        #        olist.append(to_numpy(o))
        #outlist.append(olist)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(output),
        ort_outs[0], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(to_numpy(output2),
        ort_outs2[0], rtol=1e-03, atol=1e-05)

def balexnet(md_pth, save_name):
    print("Running BranchyAlexNet Test")
    e_thr = 0.6
    bs = 1
    shape = [3,32,32]
    model = B_Alexnet_cifar(exit_threshold=e_thr)
    print(f"Using model b alexnet cifar, thr={e_thr}")

    test_x = torch.ones(1, *shape)
    if save_name[-5:] != '.onnx':
        save_name += '.onnx'

    model.set_fast_inf_mode()
    print("Finished loading model parameters")

    #save to onnx
    print("SAVING MODEL TO ONNX: ", save_name)
    save_path = to_onnx(model, shape, batch_size=bs, fname=save_name, test_in=test_x)
    print("SAVED TO: ",save_path)

def tw_smallcnn(md_pth, save_name):
    print("Running Triplewins small cnn Test")
    e_thr = 0.6
    bs = 1
    shape = [1,28,28]
    model = TW_SmallCNN(exit_threshold=e_thr)
    print(f"Using model triplewins small cnn, thr={e_thr}")

    test_x = torch.ones(1, *shape)
    if save_name[-5:] != '.onnx':
        save_name += '.onnx'

    model.set_fast_inf_mode()
    print("Finished loading model parameters")

    #save to onnx
    print("SAVING MODEL TO ONNX: ", save_name)
    save_path = to_onnx(model, shape, batch_size=bs, fname=save_name, test_in=test_x)
    print("SAVED TO: ",save_path)

def calexsvhn(md_pth, save_name):
    print("Running C AlexNet Test")
    e_thr = 0.6
    bs = 1
    shape = [3,32,32]
    model = C_Alexnet_SVHN(exit_threshold=e_thr)
    print(f"Using model c alexnet svhn, thr={e_thr}")

    test_x = torch.ones(1, *shape)
    if save_name[-5:] != '.onnx':
        save_name += '.onnx'

    model.set_fast_inf_mode()
    print("Finished loading model parameters")

    #save to onnx
    print("SAVING MODEL TO ONNX: ", save_name)
    save_path = to_onnx(model, shape, batch_size=bs, fname=save_name, test_in=test_x)
    print("SAVED TO: ",save_path)

####################################################################
#######                                                      #######
#######     MAIN FUNCTION FOR RUNNING LeNet or TestNets      #######
#######                                                      #######
####################################################################

def lenet_main(args, save_name, train):
    print("Running LeNet/TestNet Test")
    bs = 1#64
    #shape = [5,16,16]
    shape = [1,28,28]
    # set up model
    #model = Lenet()
    #model = Testnet()

    # other networks for testing
    #model = BrnFirstExit()
    #model = BrnSecondExit()
    model = Backbone_se()

    if train:
        #briefly train/load the model
        batch_size = 512 #training bs in branchynet
        train_dl, valid_dl, test_dl = pull_mnist_data(batch_size, normalise=True)
        lr = 0.001 #Adam algo - step size alpha=0.001
        #exponetial decay rates for 1st & 2nd moment: 0.99, 0.999
        exp_decay_rates = [0.99, 0.999]
        opt = optim.Adam(model.parameters(), betas=exp_decay_rates, lr=lr)
        path_str = f'outputs/{save_name}/'
        save_path = train_backbone(model, train_dl, valid_dl,
                batch_size=batch_size, save_path=path_str, epochs=20,
                loss_f=nn.CrossEntropyLoss(), opt=opt, dat_norm=True)

    else:
        md_pth = args.trained_path
        checkpoint = torch.load(md_pth)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    print("Model done")

    #feed same input to both
    test_x = torch.ones(1, *shape)#torch.randn(1, *shape)

    tfs = transforms.Compose([
        transforms.ToTensor()
        ])
    mnist_dl = DataLoader( torchvision.datasets.MNIST('../data/mnist',
                                    download=True, train=False, transform=tfs),
                batch_size=1, drop_last=True, shuffle=False)

    mnistiter = iter(mnist_dl)
    xb, yb = mnistiter.next()

    #combi = torch.cat((test_x, test_x, xb), 0)

    print("STARTING RUN")
    #output = model(xb)
    output2 = model(test_x)
    #print("PT OUT:", output)
    print("PT OUT2:", output2)

    #save to onnx
    print("SAVING")

    path='outputs/onnx'
    if save_name[-5:] != '.onnx':
        save_name += '.onnx'
    sv_pnt = os.path.join(path, save_name)
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
        opset_version=13,          # the ONNX version to export the model to
        do_constant_folding=True,  # t/f execute constant folding for optimization
        #example_outputs=ex_out,
        input_names = ['lenet_in'],   # the model's input names
        output_names = ['lenet_out'],#, 'eeF'], # the model's output names
        #dynamic_axes={'lenet_in' : {0 : 'batch_size'},    # variable length axes
        #              'lenet_out' : {0 : 'exit_size'}
        #              }
    )

    print("SAVED")

    #load from onnx
    import onnx
    #onnx_model_ni = onnx.load(sv_pnt)
    #onnx.checker.check_model(onnx_model_ni)
    #print("IMPORTED")

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
    #onnx_model = onnx_model_ni


    #onnx runtime model
    import onnxruntime
    ort_session = onnxruntime.InferenceSession(sv_pnt)
    def to_numpy(tensor):
        if isinstance(tensor, list):
            tensor = tensor[0]
        return tensor.detach().cpu().numpy() if tensor.requires_grad else \
            tensor.cpu().numpy()

    print("RUNNING ONNX")
    # compute ONNX Runtime output prediction
    #ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(xb)}
    #ort_outs = ort_session.run(None, ort_inputs)

    #print("ONNX_OUT", ort_outs)
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

    #np.testing.assert_allclose(to_numpy(output),
    #    ort_outs[0], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(to_numpy(output2),
        ort_outs2[0], rtol=1e-03, atol=1e-05)

def alexnet(args, save_name, train):
    print("Running LeNet/TestNet Test")
    bs = 1#64
    shape = [3,32,32]

    model = Backbone_Alex()
    model.eval()
    print("Model done")

    path='outputs/onnx'
    if save_name[-5:] != '.onnx':
        save_name += '.onnx'
    sv_pnt = os.path.join(path, save_name)
    if not os.path.exists(path):
        os.makedirs(path)

    x = torch.randn(bs, *shape)

    torch.onnx.export(
        #scr_model,      # model being run
        model,
        x,              # model input (or a tuple for multiple inputs)
        sv_pnt,         # where to save the model (can be a file or file-like object)
        export_params=True, # store the trained parameter weights inside the model file
        opset_version=13,          # the ONNX version to export the model to
        do_constant_folding=True,  # t/f execute constant folding for optimization
        #example_outputs=ex_out,
        input_names = ['a_in'],   # the model's input names
        output_names = ['a_out'],#, 'eeF'], # the model's output names
        #dynamic_axes={'lenet_in' : {0 : 'batch_size'},    # variable length axes
        #              'lenet_out' : {0 : 'exit_size'}
        #              }
    )
    print("SAVED")

# triple wins small cnn
def triplewins(args, save_name, train):
    print("Gen ONNX model for triple wins small cnn mnist")
    bs = 1#64
    shape = [1,28,28]

    model = TW_BB_SmallCNN()
    print("Model done")

    path='outputs/onnx'
    if save_name[-5:] != '.onnx':
        save_name += '.onnx'
    sv_pnt = os.path.join(path, save_name)
    if not os.path.exists(path):
        os.makedirs(path)

    x = torch.randn(bs, *shape)

    torch.onnx.export(
        #scr_model,      # model being run
        model,
        x,              # model input (or a tuple for multiple inputs)
        sv_pnt,         # where to save the model (can be a file or file-like object)
        export_params=True, # store the trained parameter weights inside the model file
        opset_version=13,          # the ONNX version to export the model to
        do_constant_folding=True,  # t/f execute constant folding for optimization
        #example_outputs=ex_out,
        input_names = ['tw_in'],   # the model's input names
        output_names = ['tw_out'],#, 'eeF'], # the model's output names
        #dynamic_axes={'lenet_in' : {0 : 'batch_size'},    # variable length axes
        #              'lenet_out' : {0 : 'exit_size'}
        #              }
    )
    print("SAVED")

# sdn/l2stop resnet backbone
def sdn_bb(args, save_name, train):
    print("Gen ONNX model for sdn/l2stop resnet18 BB - cifar100")
    bs = 1#64
    shape = [3,32,32]

    model = SDN_BB_ResNet()
    print("Model done")

    path='outputs/onnx'
    if save_name[-5:] != '.onnx':
        save_name += '.onnx'
    sv_pnt = os.path.join(path, save_name)
    if not os.path.exists(path):
        os.makedirs(path)

    x = torch.randn(bs, *shape)

    torch.onnx.export(
        #scr_model,      # model being run
        model,
        x,              # model input (or a tuple for multiple inputs)
        sv_pnt,         # where to save the model (can be a file or file-like object)
        export_params=True, # store the trained parameter weights inside the model file
        opset_version=13,          # the ONNX version to export the model to
        do_constant_folding=True,  # t/f execute constant folding for optimization
        #example_outputs=ex_out,
        input_names = ['sdn_in'],   # the model's input names
        output_names = ['sdn_out'],#, 'eeF'], # the model's output names
        #dynamic_axes={'lenet_in' : {0 : 'batch_size'},    # variable length axes
        #              'lenet_out' : {0 : 'exit_size'}
        #              }
    )
    print("SAVED")

def path_check(string): #checks for valid path
    if os.path.exists(string):
        return string
    else:
        raise FileNotFoundError(string)

def main():
    parser = argparse.ArgumentParser(description="script for running pytorch-onnx tests")
    parser.add_argument('--model',choices=['brn','lenet', 'alexnet', 'balexnet',
        'triplewins','tw_smallcnn','sdn_bb', 'calexsvhn'],
                        help='choose the model')
    parser.add_argument('--trained_path', type=path_check,
                        help='path to trained model')
    parser.add_argument('--save_name', type=str,
                        help='path to trained model')
    parser.add_argument('-t', '--train', action='store_true',
            help='train lenet model or not') #default is false
    args = parser.parse_args()

    if args.model == 'brn':
        brn_main(md_pth=args.trained_path, save_name=args.save_name)
    elif args.model == 'lenet':
        print("ignorning model path provided")
        lenet_main(args, save_name=args.save_name, train=args.train)
    elif args.model == 'alexnet':
        print("running Alexnet-bb")
        alexnet(args, save_name=args.save_name, train=False) #false doesnt do anything here
    elif args.model == 'triplewins':
        print("running triple wins bb")
        triplewins(args, save_name=args.save_name, train=False) #false doesnt do anything here
    elif args.model == 'sdn_bb':
        print("running sdn/l2stop sdn resnet bb")
        sdn_bb(args, save_name=args.save_name, train=False) #false doesnt do anything here
    elif args.model == 'balexnet':
        balexnet(md_pth=None, save_name=args.save_name) #false doesnt do anything here
    elif args.model == 'tw_smallcnn':
        tw_smallcnn(md_pth=None, save_name=args.save_name) #false doesnt do anything here
    elif args.model == 'calexsvhn':
        calexsvhn(md_pth=None, save_name=args.save_name) #false doesnt do anything here

if __name__ == "__main__":
    main()
