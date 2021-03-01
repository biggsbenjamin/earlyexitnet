import torch
import torch.nn as nn

#import numpy as np
from scipy.stats import entropy


class ConvPoolAc(nn.Module):
    def __init__(self, chanIn, chanOut, kernel=3, stride=1, padding=1, p_ceil_mode=False):
        super(ConvPoolAc, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(chanIn, chanOut, kernel_size=kernel,
                stride=stride, padding=padding, bias=False),
            nn.MaxPool2d(2, stride=2, ceil_mode=p_ceil_mode), #ksize, stride
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.layer(x)

#Main Network
class Branchynet(nn.Module):

    def __init__(self):
        super(Branchynet, self).__init__()

        # call function to build layers
            #probably need to fragment the model into a moduleList
            #having distinct indices to compute the classfiers/branches on
        #function for building the branches
            #this includes the individual classifier layers, can keep separate
            #last branch/classifer being terminal linear layer - included here not main net

        self.fast_inference_mode = False
        #self.exit_criterion = entropy
        self.exit_threshold = 0.5 #TODO make input, learnable, better default value

        self.backbone = nn.ModuleList()
        self.exits = nn.ModuleList()

        self.chansIn = [5,10]
        self.chansOut = [10,20]

        #weight initialisiation - for standard layers this is done automagically
        self._build_backbone()
        self._build_exits()

    def _build_backbone(self):
        #Starting conv2d layer
        self.backbone.append(nn.Conv2d(1, 5, kernel_size=5, stride=1, padding=3))

        #after first exit
        post_exit= nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        #strt_bl = ConvPoolAc(1, 5, kernel=5, stride=1, padding=3)
        #self.backbone.append(strt_bl)

        #adding ConvPoolAc blocks - remaining backbone
        bb_layers = [post_exit] #include post exit 1 layers
        #bb_layers = []
        for cI, cO in zip(self.chansIn, self.chansOut): #TODO make input variable
            bb_layer = ConvPoolAc(cI, cO, kernel=5, stride=1, padding=3, p_ceil_mode=True)
            bb_layers.append(bb_layer)

        bb_layers.append(nn.Flatten())
        bb_layers.append(nn.Linear(720, 84))

        remaining_backbone_layers = nn.Sequential(*bb_layers)
        self.backbone.append(remaining_backbone_layers)

    def _build_exits(self):
        #adding early exits/branches

        #early exit 1
        ee1 = nn.Sequential(
            nn.MaxPool2d(2, stride=2), #ksize, stride
            nn.ReLU(True),
            ConvPoolAc(5, 10, kernel=3, stride=1, padding=1, p_ceil_mode=True),
            nn.Flatten(),
            nn.Linear(640,10) #insize,outsize - make variable on num of classes
        )
        self.exits.append(ee1)

        #final exit
        eeF = nn.Sequential(
            nn.Flatten(),
            nn.Linear(84,10)
        )
        self.exits.append(eeF)

    def exit_criterion(self, x): #not for batches atm
        #evaluate the exit criterion on the result provided
        #return true if it can exit, false if it can't
        softmax = nn.functional.softmax(x)
        #apply scipy.stats.entropy for branchynet, when they do theirs, its on a batch
        return entropy(softmax) < self.exit_threshold

    def forward(self, x):
        #std forward function - add var to distinguish be test and inf

        res = []
        if self.fast_inference_mode: #TODO fix for batches, dont think its trivial
            for i in range(len(self.backbone)):
                x = self.backbone[i](x)
                ec = self.exits[i](x)
                res.append(ec)
                if self.exit_criterion(ec):
                    return res #return the results early if in inference mode

        else:
            #calculate all exits
            for i in range(len(self.backbone)):
                x = self.backbone[i](x)
                res.append(self.exits[i](x))

        return res

    def set_fast_inf_mode(mode=True):
        if mode:
            self.eval()
        self.fast_inference_mode = mode

