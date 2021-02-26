#training, testing for branchynet-pytorch version
#testing fit with onnx

from models.Branchynet import Branchynet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
import torchvision.transforms as transforms

#import os
import numpy as np

def train():
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

def main():
    #set up the model
    model = Branchynet()
    print("Model done")

    #get data and load if not already exiting - MNIST for now
        #sort into training, and test data
    batch_size = 512 #training bs in branchynet
    train_dl, valid_dl = pull_mnist_data(batch_size)
    print("Got training and test data")

    #set optimiser
    lr = 0.5
    opt = optim.SGD(model.parameters(), lr=lr)
    print("Optimiser set")

    #set loss function - og bn used "softmax_cross_entropy" unclear if this is the same
    loss_f = nn.CrossEntropyLoss() # combines log softmax and negative log likelihood
    print("Loss function set")

    #shape testing
    #print(shape_test(model, [1,28,28], [1])) #output is not one hot encoded

    #start training loop for epochs - at some point add recording points here
    epochs = 10

    for epoch in range(epochs):
        model.train()
        print("Starting epoch:", epoch+1, end="... ", flush=True)
        for xb, yb in train_dl:
            results = model(xb)

            loss = 0.0
            for res in results:
                loss += loss_f(res, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            valid_losses = np.sum(np.array(
                    [[loss_f(exit, yb) for exit in model(xb)]
                        for xb, yb in valid_dl]), axis=0)

        print("V Loss:", valid_losses / len(valid_dl))

    #once trained, run it on the test data
    #be nice to have comparison against pytorch pretrained LeNet from pytorch
    #get percentage exits and avg accuracies, add some timing etc.

if __name__ == "__main__":
    main()
