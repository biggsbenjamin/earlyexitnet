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
from earlyexitnet.tools import MNISTDataColl,CIFAR10DataColl,load_model

# import nn for loss function
import torch.nn as nn
# general imports
import os
from datetime import datetime as dt

def get_exits(model_str):
    # set number of exits
    if model_str in ['lenet','testnet','brnfirst',
                     'brnsecond','brnfirst_se','backbone_se']:
        exits = 1
    elif model_str in ['b_lenet','b_lenet_fcn',
                       'b_lenet_se','b_lenet_cifar']:
        exits = 2
    else:
        raise NameError("Model not supported, check name:",args.model_name)

    return exits

def test_only(args):
    model = get_model(args.model_name)
    exits = get_exits(args.model_name)
    print("Model:", args.model_name)
    #set loss function
    loss_f = nn.CrossEntropyLoss() # combines log softmax and negative log likelihood
    print("Loss default function set")
    batch_size_test = args.batch_size_test #test bs in branchynet
    print("Setting up for testing")
    #load in the model from the path
    load_model(model, args.trained_model_path)
    #skip to testing
    if args.dataset == 'mnist':
        datacoll = MNISTDataColl(batch_size_test=batch_size_test)
    elif args.dataset == 'cifar10':
        datacoll = CIFAR10DataColl(batch_size_test=batch_size_test)
    else:
        raise NameError("Dataset not supported, check name:",
                        args.dataset)
    # path to notes write up
    notes_path = os.path.join(
        os.path.split(args.trained_model_path)[0],'notes.txt')
    # path to the model (already trained)
    save_path = args.trained_model_path
    # RUN THE MODEL OVER TEST DATASET
    test(datacoll,model,exits,loss_f,save_path,notes_path,args)

def test(datacoll,model,exits,loss_f,
         save_path,notes_path,args):
    # set up test class then write results
    test_dl = datacoll.get_test_dl()
    if exits>1:
        top1_thr = [args.top1_threshold, 0]
        entr_thr = [args.entr_threshold, 1000000]
        net_test = Tester(model,test_dl,loss_f,exits,
                top1_thr,entr_thr)
    else:
        net_test = Tester(model,test_dl,loss_f,exits)

    top1_thr = net_test.top1acc_thresholds
    entr_thr = net_test.entropy_thresholds
    net_test.test()
    #get test results
    test_size = net_test.sample_total
    top1_pc = net_test.top1_pc
    entropy_pc = net_test.entr_pc
    top1acc = net_test.top1_accu
    entracc = net_test.entr_accu
    t1_tot_acc = net_test.top1_accu_tot
    ent_tot_acc = net_test.entr_accu_tot
    full_exit_accu = net_test.full_exit_accu
    #get percentage exits and avg accuracies, add some timing etc.
    print("top1 thrs: {},  entropy thrs: {}".format(top1_thr, entr_thr))
    print("top1 exit %s {},  entropy exit %s {}".format(top1_pc, entropy_pc))
    print("Accuracy over exited samples:")
    print("top1 exit acc % {}, entropy exit acc % {}".format(top1acc, entracc))
    print("Accuracy over network:")
    print("top1 acc % {}, entr acc % {}".format(t1_tot_acc,ent_tot_acc))
    print("Accuracy of the individual exits over full set: {}".format(full_exit_accu))

    ts = dt.now().strftime("%Y-%m-%d_%H%M%S")
    with open(notes_path, 'a') as notes:
        notes.write("\n#######################################\n")
        notes.write(f"\nTesting results: for {args.model_name} @ {ts}\n  ")
        notes.write(f"on dataset {args.dataset}\n")
        notes.write("Test sample size: {}\n".format(test_size))
        notes.write("top1 thrs: {},  entropy thrs: {}\n".format(top1_thr, entr_thr))
        notes.write("top1 exit %s {}, entropy exit %s {}\n".format(top1_pc, entropy_pc))
        notes.write("best* model "+save_path+"\n")
        notes.write("Accuracy over exited samples:\n")
        notes.write("top1 exit acc % {}, entropy exit acc % {}\n".format(top1acc, entracc))
        notes.write("Accuracy over EE network:\n")
        notes.write("top1 acc % {}, entr acc % {}\n".format(t1_tot_acc,ent_tot_acc))
        notes.write("Accuracy of the individual exits over full set: {}\n".format(full_exit_accu))

        if args.run_notes is not None:
            notes.write(args.run_notes+"\n")
    notes.close()

"""
Main training and testing function run from the cli
"""
def train_n_test(args):
    #set up the model specified in args
    model = get_model(args.model_name)
    exits = get_exits(args.model_name)
    print("Model done:", args.model_name)
    #set loss function - og bn used "softmax_cross_entropy" unclear if this is the same
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
                v_split=validation_split)
    elif args.dataset == 'cifar10':
        datacoll = CIFAR10DataColl(batch_size_train=batch_size_train,
                batch_size_test=batch_size_test,normalise=normalise,
                v_split=validation_split)
    else:
        raise NameError("Dataset not supported, check name:",args.dataset)
    train_dl = datacoll.get_train_dl()
    valid_dl = datacoll.get_valid_dl()
    print("Got training data, batch size:",batch_size_train)

    #start training loop for epochs - at some point add recording points here
    path_str = 'outputs/'
    print("backbone epochs: {} joint epochs: {}".format(args.bb_epochs, args.jt_epochs))

    # Set up training class
    net_trainer = Trainer(
        model, train_dl, valid_dl, batch_size_train,
        path_str,loss_f=loss_f, exits=exits,
        backbone_epochs=args.bb_epochs,
        exit_epochs=args.ex_epochs,
        joint_epochs=args.jt_epochs
    )
    if exits > 1:
        best,last=net_trainer.train_joint(pretrain_backbone=True)
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
        notes.write("Training batch size {}, Test batchsize {}\n".format(batch_size_train,
                                                                       batch_size_test))
        # record exit weighting (if model has it)
        if hasattr(model,'exit_loss_weights'):
            notes.write("model training exit weights:"+str(net_trainer.model.exit_loss_weights)+"\n")
        notes.write("Path to last model:"+str(last)+"\n")
    notes.close()

    #TODO graph training data
    #separate graphs for pre training and joint training

    #once trained, run it on the test data
    test(datacoll,net_trainer.model,exits,loss_f,save_path,notes_path,args)


def path_check(string): #checks for valid path
    if os.path.exists(string):
        return string
    else:
        raise FileNotFoundError(string)

"""
Main function that sorts out the CLI args and runs training and testing function.
"""
def main():
    parser = argparse.ArgumentParser(description="Early Exit CLI")

    parser.add_argument('-m','--model_name',
            choices=[   'b_lenet',
                        'b_lenet_fcn',
                        'b_lenet_se',
                        'lenet',
                        'testnet',
                        'brnfirst',
                        'brnfirst_se',
                        'brnsecond',
                        'backbone_se',
                        'b_lenet_cifar',
                        ],
            required=True, help='select the model name')

    parser.add_argument('-mp','--trained_model_path',metavar='PATH',type=path_check,
            required=False,
            help='Path to previously trained model to load, the same type as model name')

    parser.add_argument('-bstr','--batch_size_train',type=int,default=512,
                        help='batch size for the training of the network')
    parser.add_argument('-bbe','--bb_epochs', metavar='N',type=int, default=1, required=False,
            help='Epochs to train backbone separately, or non ee network')
    parser.add_argument('-jte','--jt_epochs', metavar='n',type=int, default=1, required=False,
            help='epochs to train exits jointly with backbone')
    parser.add_argument('-exe','--ex_epochs', metavar='n',type=int, default=1, required=False,
            help='epochs to train exits with frozen backbone')
    parser.add_argument('-rn', '--run_notes', type=str, required=False,
            help='Some notes to add to the train/test information about the model or otherwise')

    #parser.add_argument('--seed', metavar='N', type=int, default=random.randint(0,2**32-1),
    #    help='Seed for training, NOT CURRENTLY USED')

    parser.add_argument('-d','--dataset',
            choices=['mnist','cifar10','cifar100'],
            required=False, default='mnist',
            help='select the dataset, default is mnist')
    #threshold inputs for TESTING
    parser.add_argument('-bste','--batch_size_test',type=int,default=1,
                        help='batch size for the testing of the network')
    parser.add_argument('-t1','--top1_threshold',type=float,required=False)
    parser.add_argument('-entr','--entr_threshold',type=float,required=False)

    #TODO arguments to add
        #training loss function
        #some kind of testing specification

    # parse the arguments
    args = parser.parse_args()

    if args.trained_model_path is not None:
        test_only(args)
    else:
        train_n_test(args)

if __name__ == "__main__":
    main()
