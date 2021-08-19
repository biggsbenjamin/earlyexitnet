#training, testing for branchynet-pytorch version
#testing fit with onnx

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

#TODO might merge exit+backbone for code reuse
def train_backbone(model, train_dl, valid_dl, save_path, epochs=50,
                    loss_f=nn.CrossEntropyLoss(), opt=None):
    #train network backbone

    if opt is None:
        #set to branchynet default
        #Adam algo - step size alpha=0.001
        lr = 0.001
        #exponetial decay rates for 1st & 2nd moment: 0.99, 0.999
        exp_decay_rates = [0.99, 0.999]
        backbone_params = [
                {'params': model.backbone.parameters()},
                {'params': model.exits[-1].parameters()}
                ]

        opt = optim.Adam(backbone_params, betas=exp_decay_rates, lr=lr)

    #probe params to double check only backbone run
    #probe_params(model) #satisfied that setting specific params works right

    #timestamp = dt.now().strftime("%Y-%m-%d_%H%M%S")
    #bb_folder_path = 'backbone-' + timestamp

    best_val_loss = [1.0, '']
    for epoch in range(epochs):
        model.train()
        print("Starting epoch:", epoch+1, end="... ", flush=True)

        #training loop
        for xb, yb in train_dl:
            results = model(xb)
            #loss for backbone ignores other exits
            #Wasting some forward compute of early exits
            #but shouldn't be included in backward step
            #since params not looked at by optimiser
            #TODO add backbone only method to bn class
            loss = loss_f(results[-1], yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

        #validation
        model.eval()
        with torch.no_grad():
            valid_losses = np.sum(np.array(
                    [[loss_f(exit, yb) for exit in model(xb)]
                        for xb, yb in valid_dl]), axis=0)

        val_loss_avg = valid_losses[-1] / len(valid_dl)
        print("V Loss:", val_loss_avg)
        #probe_params(model)
        savepoint = save_model(model, save_path, file_prefix='backbone-'+str(epoch+1), opt=opt)

        if val_loss_avg < best_val_loss[0]:
            best_val_loss[0] = val_loss_avg
            best_val_loss[1] = savepoint
    print("BEST VAL LOSS: ", best_val_loss[0], " for epoch: ", best_val_loss[1])
    return savepoint #link to best val loss model

def train_exits(model, epochs=100):
    #train the exits

    #Adam algo - step size alpha=0.001
    #exponetial decay rates for 1st & 2nd moment: 0.99, 0.999
    return #something trained

def train_joint(model, train_dl, valid_dl, save_path, opt=None,
                loss_f=nn.CrossEntropyLoss(), backbone_epochs=50,
                joint_epochs=100, pretrain_backbone=True):

    timestamp = dt.now().strftime("%Y-%m-%d_%H%M%S")

    if pretrain_backbone:
        print("PRETRAINING BACKBONE FROM SCRATCH")
        folder_path = 'pre_Trn_bb_' + timestamp
        best_bb_path = train_backbone(model, train_dl,
                valid_dl, os.path.join(save_path, folder_path),
                epochs=backbone_epochs, loss_f=loss_f)
        #train the rest...
        print("LOADING BEST BACKBONE")
        load_model(model, best_bb_path)
        print("JOINT TRAINING WITH PRETRAINED BACKBONE")

        prefix = 'pretrn-joint'
    else:
        #jointly trains backbone and exits from scratch
        print("JOINT TRAINING FROM SCRATCH")
        folder_path = 'jnt_fr_scrcth' + timestamp
        prefix = 'joint'

    spth = os.path.join(save_path, folder_path)

    #set up the joint optimiser
    if opt is None: #TODO separate optim function to reduce code, maybe pass params?
        #set to branchynet default
        lr = 0.001 #Adam algo - step size alpha=0.001
        #exponetial decay rates for 1st & 2nd moment: 0.99, 0.999
        exp_decay_rates = [0.99, 0.999]

        opt = optim.Adam(model.parameters(), betas=exp_decay_rates, lr=lr)

    best_val_loss = [[1.0,1.0], ''] #TODO make sure list size matches num of exits
    for epoch in range(joint_epochs):
        model.train()
        print("starting epoch:", epoch+1, end="... ", flush=True)

        #training loop
        for xb, yb in train_dl:
            results = model(xb)

            losses = [weighting * loss_f(res, yb)
                        for weighting, res in zip(model.exit_loss_weights,results)]

            opt.zero_grad()
            #backward
            for loss in losses[:-1]:
                loss.backward(retain_graph=True)
            losses[-1].backward()

            opt.step()

        #validation
        model.eval()
        with torch.no_grad():
            valid_losses = np.sum(np.array(
                    [[loss_f(exit, yb) for exit in model(xb)]
                        for xb, yb in valid_dl]), axis=0)


        val_loss_avg = valid_losses / len(valid_dl)
        print("v loss:", val_loss_avg)
        savepoint = save_model(model, spth, file_prefix=prefix+'-'+str(epoch+1), opt=opt)

        el_total=0.0
        bl_total=0.0
        for exit_loss, best_loss in zip(val_loss_avg,best_val_loss[0]):
            el_total+=exit_loss
            bl_total+=best_loss
        if el_total < bl_total:
            best_val_loss[0] = val_loss_avg
            best_val_loss[1] = savepoint

    print("BEST* VAL LOSS: ", best_val_loss[0], " for epoch: ", best_val_loss[1])
    return #something trained

def test():
    return #some test stats

def pull_mnist_data(batch_size=64):
    #transforms - TODO check if branchynet normalises data
    tfs = transforms.Compose([
        transforms.ToTensor()
        ])

    mnist_train_dl = DataLoader( torchvision.datasets.MNIST('../data/mnist',
                                    download=True, train=True, transform=tfs),
                batch_size=batch_size, drop_last=True, shuffle=True)

    mnist_valid_dl = DataLoader( torchvision.datasets.MNIST('../data/mnist',
                                    download=True, train=False, transform=tfs),
                batch_size=batch_size, drop_last=True, shuffle=True)

    return mnist_train_dl, mnist_valid_dl

def save_model(model, path, file_prefix='', seed=None, epoch=None, opt=None, loss=None):
    #TODO add saving for inference only
    #TODO add bool to save as onnx - remember about fixed bs
    #saves the model in pytorch format to the path specified
    timestamp = dt.now().strftime("%Y-%m-%d_%H%M%S")
    filenm = file_prefix + '-' + timestamp
    save_dict ={'timestamp': timestamp,
                'model_state_dict': model.state_dict()
                }

    if seed is not None:
        save_dict['seed'] = seed
    if epoch is not None:
        save_dict['epoch'] = epoch
        filenm += f'{epoch:03d}'
    if opt is not None:
        save_dict['opt_state_dict'] = opt.state_dict()
    if loss is not None:
        save_dict['loss'] = loss

    if not os.path.exists(path):
        os.makedirs(path)

    filenm += '.pth'
    file_path = os.path.join(path, filenm)

    torch.save(save_dict, file_path)

    print("Saved to:", file_path)
    return file_path


def load_model(model, path):
    #TODO add "warmstart" - partial reloading of model, useful for backbone pre_training
    #loads the model from the path specified
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    #TODO optionals
    #opt.load_state_dict(checkpoint['opt_state_dict'])
    #epoch = checkpoint['epoch']
    #loss = checkpoint['loss']

def main():
    #vis_backprop_graphs() #for visualising the backprop graph shape
    #shape testing
    #print(shape_test(model, [1,28,28], [1])) #output is not one hot encoded

    #set up the model
    model = Branchynet()
    print("Model done")

    #get data and load if not already exiting - MNIST for now
        #sort into training, and test data
    batch_size = 512 #training bs in branchynet
    train_dl, valid_dl = pull_mnist_data(batch_size)
    print("Got training and test data")

    #set loss function - og bn used "softmax_cross_entropy" unclear if this is the same
    loss_f = nn.CrossEntropyLoss() # combines log softmax and negative log likelihood
    print("Loss function set")

    #start training loop for epochs - at some point add recording points here
    bb_epochs = 5 #50 for backbone
    jt_epochs = 10 #100 for joint with exits
    path_str = 'outputs/'

    #train_backbone(model, train_dl, valid_dl, path_str, epochs=epochs, loss_f=loss_f)

    train_joint(model, train_dl, valid_dl, path_str, backbone_epochs=bb_epochs,
            joint_epochs=jt_epochs, loss_f=loss_f, pretrain_backbone=True)


    #once trained, run it on the test data
    #be nice to have comparison against pytorch pretrained LeNet from pytorch
    #get percentage exits and avg accuracies, add some timing etc.

if __name__ == "__main__":
    main()
