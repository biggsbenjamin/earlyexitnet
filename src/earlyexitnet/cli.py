"""
CLI for training and testing early-exit and normal CNNs.
"""

import random
import os
import argparse

# import training functions
from earlyexitnet.training_tools.train import Trainer,get_model

# import testing class and functions
from earlyexitnet.testing_tools.test import Tester

# import dataloaders from tools
from earlyexitnet.tools import \
    MNISTDataColl,CIFAR10DataColl,load_model,path_check

from earlyexitnet.onnx_tools.onnx_helpers import \
    to_onnx

# import nn for loss function
import torch.nn as nn
# torch for cuda check
import torch
# general imports
import os
from datetime import datetime as dt, timedelta

from time import perf_counter
import json
from tqdm import tqdm
import numpy as np


def get_exits(model_str):
    # NOTE only used if there is no exit num constant
    # set number of exits
    if model_str in ['lenet','testnet','brnfirst',
                     'brnsecond','brnfirst_se','backbone_se']:
        exits = 1
    elif model_str in ['b_lenet','b_lenet_fcn',
                       'b_lenet_se','b_lenet_cifar']:
        exits = 2
    else:
        raise NameError("Model not supported, check name:",model_str)

    return exits

def get_save_path(model_name, notes_path, timestamp=True, desc=None, show=True, filetype='json'):
    save_path = model_name
    nope = os.path.split(notes_path)[0]
    if desc is not None:
        save_path += f"_{desc}"
    if timestamp:
        ts = dt.now().strftime("%Y-%m-%d_%H%M%S")
        save_path += f"_{ts}"

    save_path += f'.{filetype}'

    save_path = os.path.join(nope,save_path)
    #if show:
    print("Storing the test results at", save_path)
    return save_path

def test_only(args):
    """Given the parameters passed in the command line, run tests on a given model

    Args:
        args: CLI arguments

    Returns:
        None
    """
    model = get_model(args.model_name)
    # get number of exits
    if hasattr(model,'exit_num'):
        exits=model.exit_num
    else:
        exits=get_exits(args.model_name)
    print("Model:", args.model_name)
    #set loss function
    loss_f = nn.CrossEntropyLoss() # combines log softmax and negative log likelihood
    print("Loss default function set")
    batch_size_test = args.batch_size_test #test bs in branchynet
    print("Setting up for testing")
    #load in the model from the path

    # Device setup
    if torch.cuda.is_available() and args.gpu_target is not None:
        device = torch.device(f"cuda:{args.gpu_target}")
        pin_mem = True
    else:
        device = torch.device("cpu")
        pin_mem = False
    print("Device:", device)

    load_model(model, args.trained_model_path, device=device)

    num_workers = 1 if args.num_workers is None else args.num_workers
    print(f"Number of workers: {num_workers}")

    # check if there are thresholds provided
    if args.top1_threshold is None and \
            args.entr_threshold is None and \
            exits > 1:
        # no thresholds provided, skip testing
        print("WARNING: No Thresholds provided, skipping testing.")
        return model
    #skip to testing
    if args.dataset == 'mnist':
        datacoll = MNISTDataColl(batch_size_test=batch_size_test,num_workers=num_workers,pin_mem=pin_mem)
    elif args.dataset == 'cifar10':
        datacoll = CIFAR10DataColl(batch_size_test=batch_size_test,num_workers=num_workers, no_scaling=args.no_scaling,pin_mem=pin_mem)
    else:
        raise NameError("Dataset not supported, check name:",
                        args.dataset)
    # path to notes write up
    if args.notes_path is not None:
        notes_path = args.notes_path
    else:
        notes_path = f'./outputs/{args.model_name}/jsons/'
        if not os.path.exists(notes_path):
            os.makedirs(notes_path)

    if args.threshold_range is not None and args.threshold_step is not None:
        if len(args.threshold_range) == 2:
            test_multiple(datacoll,model,exits,loss_f,notes_path,args)
        else:
            raise NameError("Invalid amount of value for the threshold range:", len(args.threshold_range))
    else:
        # RUN THE MODEL OVER TEST DATASET
        test_single(datacoll,model,exits,loss_f,notes_path,args)

def test_multiple(datacoll,model,exits,loss_f,notes_path,args):
    """Run multiples tests on a given model, varying the ealy-exit threshold linearly in a given range,
    with a given step. Save the test results in JSON file

    Args:
        datacoll: Dataloader object
        model : Loader model object
        exits (_int_) : Number of exits for the given model
        loss_f : Loss function (unused)
        notes_path : Optional custom path for saving the output of the test
        args : CLI arguments
    """
    step = args.threshold_step
    min_thr, max_thr = args.threshold_range

    num_tests = int((max_thr - min_thr)/step)
    total_time = 0
    running_time = 0

    results = []

    for i, thresh in enumerate(np.arange(min_thr, max_thr, step)):
        rt_string = dt.utcfromtimestamp(timedelta(seconds=int(running_time)).total_seconds()).strftime("%M:%S")
        tt_string = dt.utcfromtimestamp(timedelta(seconds=total_time).total_seconds()).strftime("%M:%S")
        print(f"\nRunning test {i}/{num_tests}, thresh: {thresh} \t\t [{rt_string}/{tt_string}]")

        elapsed_time, test_stats = run_test(datacoll,model,exits,[float(thresh)], None,loss_f,args) # ignore entropy threshold

        results.append(test_stats)

        running_time += elapsed_time
        total_time = int((float(running_time) / (i+1)) * num_tests)

    final_object = {}
    final_object["model"] = args.model_name
    final_object["dataset"] = args.dataset
    final_object["thresholds"] = {"min_thr":min_thr, "max_thr":max_thr, "step":step}

    final_object["test_vals"] = results

    save_path = get_save_path(args.model_name, notes_path, desc="multiple")


    with open(save_path, 'a') as output:
        output.write(json.dumps(final_object, indent=2))


def run_test(datacoll,model,exits,top1_thr,entr_thr,loss_f,args, save_raw = False):
    """Run and time an individual test on entire dataset with given test parameters

    Args:
        datacoll: Dataloader object
        model : Loader model object
        exits (_int_) : Number of exits for the given model
        top1_thr (_type_): Early-exit threshold used for softmax-like confidence functions
        entr_thr (_type_): Early-exit threshold for entropy-like confidence functions
        loss_f : Loss function (unused)
        args : CLI arguments
        save_raw (bool, optional): If True, final activation layer and softmax vectors are saved in return dict. Defaults to False.

    Returns:
        _tuple[float, dict]_: Elapsed time and dictionary containing all the test information
    """

    # Device setup
    if torch.cuda.is_available() and args.gpu_target is not None:
        device = torch.device(f"cuda:{args.gpu_target}")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}, pin memory = {datacoll.pin_mem}" )

    if top1_thr is None and entr_thr is None and exits > 2:
        # no thresholds provided, skip testing
        print("WARNING: No Thresholds provided, skipping testing.")
        return
    elif top1_thr is None:
        # set useless threshold
        top1_thr=[0]
    elif entr_thr is None:
        # set useless threshold
        entr_thr=[1000000]
    # set up test class then write results
    if exits>1:
        if len(top1_thr)+1 != exits or len(entr_thr)+1 != exits:
            raise ValueError(f"Not enough arguments for threshold. Expecting {exits-1}")
        # Adding final exit thr - must exit here so tiny/huge depending on criteria
        top1_thr.append(0)
        entr_thr.append(1000000)

    test_dl = datacoll.get_test_dl()
    net_test = Tester(model,test_dl,loss_f,exits, top1_thr,entr_thr,args.confidence_function,device, save_raw)

    start = perf_counter()
    net_test.test()
    stop = perf_counter()
    elapsed_time = stop-start

    return elapsed_time, net_test.get_stats()

def test_single(datacoll,model,exits,loss_f,notes_path,args):
    """Run single test on a given model. Save the test results in JSON file

    Args:
        datacoll: Dataloader object
        model : Loader model object
        exits (_int_) : Number of exits for the given model
        loss_f : Loss function (unused)
        notes_path : Optional custom path for saving the output of the test
        args : CLI arguments
    """

    save_raw = args.save_raw

    elapsed_time, test_stats = run_test(datacoll,model,exits,args.top1_threshold, args.entr_threshold,loss_f,args,save_raw)

    print("top1 thrs: {},  entropy thrs: {}".format(args.top1_threshold, args.entr_threshold))
    print("Total time elapsed:", elapsed_time, "s")

    ts = dt.now().strftime("%Y-%m-%d_%H%M%S")
    if not save_raw: # when saving raw output, txt file doesn't make sense
        with open(get_save_path("test_notes", notes_path, filetype='txt', timestamp=False), 'a') as notes:
        #with open(notes_path, 'a') as notes:
            notes.write("\n#######################################\n")
            notes.write(f"\nTesting results: for {args.model_name} @ {ts} ")
            notes.write(f"on dataset {args.dataset}:\n")

            notes.write(json.dumps(test_stats, indent=2))
            notes.write('\n')

            if args.run_notes is not None:
                notes.write(args.run_notes+"\n")
        notes.close()

    final_object = {}
    final_object["model"] = args.model_name
    if hasattr(args,'trained_model_path') and args.trained_model_path is not None:
        final_object["model_path"] = args.trained_model_path
    final_object["dataset"] = args.dataset

    final_object["test_vals"] = test_stats

    save_path = get_save_path(args.model_name, os.path.split(notes_path)[0], desc="single")

    with open(save_path, 'a') as output:
        if save_raw:
            output.write(json.dumps(final_object))
        else:
            output.write(json.dumps(final_object, indent=2))

def train_n_test(args):
    """
    Main training and testing function run from the cli
    """
    #set up the model specified in args
    model = get_model(args.model_name)
    # get number of exits
    if hasattr(model,'exit_num'):
        exits=model.exit_num
    else:
        exits=get_exits(args.model_name)
    print("Model done:", args.model_name)
    # Device setup
    if torch.cuda.is_available() and args.gpu_target is not None:
        device = torch.device(f"cuda:{args.gpu_target}")
        pin_mem = True
    else:
        device = torch.device("cpu")
        pin_mem = False
    print("Device:", device)
    num_workers = 1 if args.num_workers is None else args.num_workers
    print(f"Number of workers: {num_workers}")
    #set loss function
    loss_f = nn.CrossEntropyLoss() # combines log softmax and negative log likelihood
    print("Loss function set")
    print("Training new network")
    #get data and load if not already exiting - MNIST for no w
    #training bs in branchynet
    batch_size_train = args.batch_size_train
    # split the training data into training and validation (test is separate)
    validation_split = 0.2
    batch_size_test = args.batch_size_test #test bs in branchynet
    normalise=False     #normalise the training data or not
    #sort into training, and test data
    if args.dataset == 'mnist':
        datacoll = MNISTDataColl(batch_size_train=batch_size_train,
                batch_size_test=batch_size_test,normalise=normalise,
                v_split=validation_split,num_workers=num_workers,
                pin_mem=pin_mem)
    elif args.dataset == 'cifar10':
        datacoll = CIFAR10DataColl(batch_size_train=batch_size_train,
                batch_size_test=batch_size_test,normalise=normalise,
                v_split=validation_split,num_workers=num_workers,
                pin_mem=pin_mem,no_scaling=args.no_scaling)
    else:
        raise NameError("Dataset not supported, check name:",args.dataset)
    train_dl = datacoll.get_train_dl()
    valid_dl = datacoll.get_valid_dl()
    print("Got training data, batch size:",batch_size_train)

    #start training loop for epochs - at some point add recording points here
    path_str = f'outputs/{args.model_name}/'
    pretrain_backbone=True
    if args.bb_epochs == 0:
        # if no model provided, joint from scratch
        pretrain_backbone=False

    print("backbone epochs: {} joint epochs: {}".format(args.bb_epochs, args.jt_epochs))

    # Set up training class
    net_trainer = Trainer(
        model, train_dl, valid_dl, batch_size_train,
        path_str,loss_f=loss_f, exits=exits,
        # set epochs
        backbone_epochs=args.bb_epochs,
        exit_epochs=args.ex_epochs,
        joint_epochs=args.jt_epochs,
        # set opt cfg strings
        backbone_opt_cfg=args.bb_opt_cfg,
        exit_opt_cfg=args.ex_opt_cfg,
        joint_opt_cfg=args.jt_opt_cfg,
        device=device,
        pretrained_path=args.trained_model_path,
        validation_frequency=args.validation_frequency
    )
    print(f"using bb optimiser -> {args.bb_opt_cfg}")
    print(f"using jt optimiser -> {args.jt_opt_cfg}")
    if exits > 1:
        best,last=net_trainer.train_joint(pretrain_backbone=pretrain_backbone)
    else:
        ts = dt.now().strftime("%Y-%m-%d_%H%M%S")
        intr_path = f'bb_only_time_{ts}'
        print("Saving to:",intr_path)
        # training backbone only using same method
        best,last=net_trainer.train_backbone(
            internal_folder=intr_path)
    # get path to network savepoints
    save_path = os.path.split(last)[0]
    #save some notes about the run
    notes_path = os.path.join(save_path,'notes.txt')
    with open(notes_path, 'w') as notes:
        notes.write("bb epochs {}, jt epochs {}\n".format(args.bb_epochs, args.jt_epochs))
        notes.write("Training batch-size {}, Test batch-size {}\n".format(batch_size_train,
                                                                       batch_size_test))
        notes.write(f"Optimiser bb info: {net_trainer.backbone_opt_cfg}\n")
        notes.write(f"Optimiser jt info: {net_trainer.joint_opt_cfg}\n")
        notes.write(f"Dataset: {args.dataset}\n")
        # record exit weighting (if model has it)
        if hasattr(net_trainer.model,'exit_loss_weights'):
            ex_loss_w=str(net_trainer.model.exit_loss_weights)
            notes.write(f"model training exit weights:{ex_loss_w}\n")
        notes.write("Path to last model:"+str(last)+"\n")
        notes.write("Path to best model:"+str(best)+"\n")
        # store backbone training data, NOTE for now, just for resnet
        notes.write(f"bb_train_epcs: {net_trainer.bb_train_epcs}\n")
        notes.write(f"bb_train_loss: {net_trainer.bb_train_loss}\n")
        notes.write(f"bb_train_accu: {net_trainer.bb_train_accu}\n")
        notes.write(f"bb_valid_epcs: {net_trainer.bb_valid_epcs}\n")
        notes.write(f"bb_valid_loss: {net_trainer.bb_valid_loss}\n")
        notes.write(f"bb_valid_accu: {net_trainer.bb_valid_accu}\n")
    notes.close()

    #TODO graph training data
    #separate graphs for pre training and joint training

    # loading best model
    print(f"Loading best model: {best}")
    load_model(net_trainer.model, best)

    #once trained, run it on the test data
    test_single(datacoll,net_trainer.model,exits,loss_f,notes_path,args)
    return net_trainer.model,best


def main():
    """
    Main function that sorts out the CLI args and runs training and testing function.
    """

    # this formatter prevents the help text description automatically wrapping to fit in terminal
    # requires the help text to have manual line breaks
    parser = argparse.ArgumentParser(description="Early Exit CLI", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-m','--model_name',
            required=True, help='select the model name - see training model')

    parser.add_argument('-mp','--trained_model_path',metavar='PATH',type=path_check,
            required=False,
            help='Path to previously trained model to load, the same type as model name')

    parser.add_argument('-bstr','--batch_size_train',type=int,default=500,
                        help='batch size for the training of the network')
    parser.add_argument('-bbe','--bb_epochs', metavar='N',type=int, default=0, required=False,
            help='Epochs to train backbone separately, or non ee network')
    parser.add_argument('-jte','--jt_epochs', metavar='n',type=int, default=0, required=False,
            help='epochs to train exits jointly with backbone')
    parser.add_argument('-exe','--ex_epochs', metavar='n',type=int, default=0, required=False,
            help='epochs to train exits with frozen backbone')
    parser.add_argument('-vf','--validation_frequency',type=int,default=1,required=False,
            help='Validation and save frequency. Number of epochs to wait for before valid,saving.')
    # opt selection
    parser.add_argument('-bbo','--bb_opt_cfg',type=str,default='adam-brn',required=False,
            help='Selection string to pick backbone optimiser configuration from training_tools')
    parser.add_argument('-jto','--jt_opt_cfg',type=str,default='adam-brn',required=False,
            help='Selection string to pick joint optimiser configuration from training_tools')
    parser.add_argument('-exo','--ex_opt_cfg',type=str,default='adam-brn',required=False,
            help='Selection string to pick exit-only optimiser configuration from training_tools')
    # run notes
    parser.add_argument('-rn', '--run_notes', type=str, required=False,
            help='Some notes to add to the train/test information about the model or otherwise')

    parser.add_argument('-np', '--notes_path', type=path_check, required=False,
            help='Path to location for notes to be saved')
    parser.add_argument('-cf','--confidence_function',required=False,nargs='+', type=int,
            help='Choose which function to be used when determining the confidence of the network at a given exit, pick one or many.\n0 Entropy\n1 Softmax\n2 Trunc Base-2 Softmax\n3 Non-Trunc Base-2 Softmax\n4 Base-2 Sub-Softmax\n')


    #parser.add_argument('--seed', metavar='N', type=int, default=random.randint(0,2**32-1),
    #    help='Seed for training, NOT CURRENTLY USED')

    parser.add_argument('-d','--dataset',
            choices=['mnist','cifar10','cifar100'],
            required=False, default='mnist',
            help='select the dataset, default is mnist')
    parser.add_argument('--no_scaling',action='store_true',
                        help='Prevents data being scaled to between 0,1')

    # choose the cuda device to target
    parser.add_argument('-gpu','--gpu_target',type=int,required=False,
            help="GPU acceleration target, int val for torch.device( cuda:[?] )")
    parser.add_argument('-nw','--num_workers',type=int,required=False,
            help="Number of workers for data loaders")

    #threshold inputs for TESTING
    parser.add_argument('-bste','--batch_size_test',type=int,default=1,
                        help='batch size for the testing of the network')
    #threshold inputs for testing, 1 or more args - user should know model
    parser.add_argument('-t1','--top1_threshold', nargs='+',type=float,required=False)
    parser.add_argument('-entr','--entr_threshold', nargs='+',type=float,required=False)

    parser.add_argument('-tr', '--threshold_range', nargs='+', type=float, required=False)
    parser.add_argument('-ts', '--threshold_step', type=float, required=False)

    parser.add_argument('-sr', '--save_raw', action=argparse.BooleanOptionalAction, default=False, required=False,
                        help='Save the value of the final activation vector and confidence functions')

    # generate onnx graph for the model
    parser.add_argument('-go', '--generate_onnx',metavar='PATH',type=path_check,
                        required=False,
                        help='Generate onnx from loaded or trained Pytorch model, specify the directory of the output onnx')

    #TODO arguments to add
        #training loss function
        #some kind of testing specification

    # parse the arguments
    args = parser.parse_args()
    if args.trained_model_path is not None and (args.bb_epochs==0 and args.jt_epochs==0):
        model = test_only(args)
        model_path = args.trained_model_path
    else:
        model,model_path = train_n_test(args)

    if args.generate_onnx is not None:
        # get input shape for graph gen
        if args.dataset == 'mnist':
            shape = [1,28,28]
        elif args.dataset in ['cifar10','cifar100']:
            shape = [3,32,32]
        else:
            raise NameError("Unknown input shape for model.")
        # generate model name
        pt_path = os.path.splitext(os.path.basename(model_path))[0]
        fname = f'{args.model_name}_{pt_path}.onnx'
        # convert to onnx and save to op
        to_onnx(model,shape,batch_size=1,
                path=args.generate_onnx,
                fname=fname)

if __name__ == "__main__":
    main()

