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

def probe_params(model):
    #probe params to double check only backbone run
    print("backbone 1st conv")
    print([param for param in model.backbone[0].parameters()])
    print("backbone last linear")
    print([param for param in model.exits[-1].parameters()])
    print("exit 1")
    print([param for param in model.exits[0].parameters()])

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

        print("V Loss:", valid_losses[-1] / len(valid_dl))
        #probe_params(model)
        save_model(model, save_path, file_prefix='backbone', opt=opt)

    return #something trained

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
        train_backbone(model, train_dl, valid_dl, os.path.join(save_path, folder_path),
                epochs=backbone_epochs, loss_f=loss_f)
        #train_backbone(model, save_path, epochs=backbone_epochs)

        #train the rest...
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
        #Adam algo - step size alpha=0.001
        lr = 0.001
        #exponetial decay rates for 1st & 2nd moment: 0.99, 0.999
        exp_decay_rates = [0.99, 0.999]

        opt = optim.Adam(model.parameters(), betas=exp_decay_rates, lr=lr)

    for epoch in range(joint_epochs):
        model.train()
        print("starting epoch:", epoch+1, end="... ", flush=True)

        #training loop
        for xb, yb in train_dl:
            results = model(xb)

            loss = 0.0
            for res in results:
                loss += loss_f(res, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

        #validation
        model.eval()
        with torch.no_grad():
            valid_losses = np.sum(np.array(
                    [[loss_f(exit, yb) for exit in model(xb)]
                        for xb, yb in valid_dl]), axis=0)

        print("v loss:", valid_losses / len(valid_dl))
        save_model(model, spth, file_prefix=prefix, opt=opt)

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


def shape_test(model, dims_in, dims_out, loss_function=nn.CrossEntropyLoss()):
    rand_in = torch.rand(tuple([1, *dims_in]))
    rand_out = torch.rand(tuple([*dims_out])).long()

    model.eval()
    with torch.no_grad():
        results = model(rand_in)
        if isinstance(results, list):
            losses = [loss_function(res, rand_out) for res in results ]
        else:
            losses = [loss_function(results, rand_out)]
    return losses

# Our drawing graph functions. We rely / have borrowed from the following
# python libraries:
# https://github.com/szagoruyko/pytorchviz/blob/master/torchviz/dot.py
# https://github.com/willmcgugan/rich
# https://graphviz.readthedocs.io/en/stable/
def draw_graph(start, watch=[]):
    from graphviz import Digraph
    node_attr = dict(style='filled',
                    shape='box',
                    align='left',
                    fontsize='12',
                    ranksep='0.1',
                    height='0.2')
    graph = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    assert(hasattr(start, "grad_fn"))
    if start.grad_fn is not None:
        label = str(type(start.grad_fn)).replace("class", "").replace("'", "").replace(" ", "")
        print(label) #missing first item
        graph.node(label, str(start.grad_fn), fillcolor='red')

        _draw_graph(start.grad_fn, graph, watch=watch, pobj=label)#str(start.grad_fn))
        size_per_element = 0.15
    min_size = 12    # Get the approximate number of nodes and edges
    num_rows = len(graph.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    graph.graph_attr.update(size=size_str)
    graph.render(filename='net_graph.jpg')

def _draw_graph(var, graph, watch=[], seen=[], indent=".", pobj=None):
    ''' recursive function going through the hierarchical graph printing off
    what we need to see what autograd is doing.'''
    from rich import print
    if hasattr(var, "next_functions"):
        for fun in var.next_functions:
            joy = fun[0]
            if joy is not None:
                #if joy not in seen:

                label = str(type(joy)).replace("class", "").replace("'", "").replace(" ", "")
                label_graph = label
                colour_graph = ""
                seen.append(joy)
                if hasattr(joy, 'variable'):
                    happy = joy.variable
                    if happy.is_leaf:
                        label += " \U0001F343"
                        colour_graph = "green"
                        vv = []
                        for (name, obj) in watch:
                            if obj is happy:
                                label += " \U000023E9 " + \
                                    "[b][u][color=#FF00FF]" + name + \
                                    "[/color][/u][/b]"
                                label_graph += name
                                colour_graph = "blue"
                                break
                            vv = [str(obj.shape[x]) for x in range(len(obj.shape))]
                        label += " [["
                        label += ', '.join(vv)
                        label += "]]"
                        label += " " + str(happy.var())
                        graph.node(str(joy), label_graph, fillcolor=colour_graph)
                print(indent + label)
                _draw_graph(joy, graph, watch, seen, indent + ".", joy)
                if pobj is not None:
                    graph.edge(str(pobj), str(joy))

def main():
    vis_backprop_graphs() #for visualising the backprop graph shape
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
    epochs = 2 #50 for backbone, 100 for joint with exits

    path_str = 'outputs/'
    if not os.path.exists(path_str):
        os.makedirs(path_str)

    #train_backbone(model, train_dl, valid_dl, path_str, epochs=epochs, loss_f=loss_f)

    train_joint(model, train_dl, valid_dl, path_str, backbone_epochs=epochs,
            joint_epochs=epochs, loss_f=loss_f, pretrain_backbone=True)


    #once trained, run it on the test data
    #be nice to have comparison against pytorch pretrained LeNet from pytorch
    #get percentage exits and avg accuracies, add some timing etc.

if __name__ == "__main__":
    main()
