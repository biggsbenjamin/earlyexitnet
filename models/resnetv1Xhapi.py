"""
Combining MLPERF (https://github.com/mlcommons/tiny) tiny CIFAR10 ResNetv1 benchmark from eembc
with HAPI (https://steliosven10.github.io/papers/[2020]_iccad_hapi_hardware_aware_progressive_inference.pdf)
exit training method + MSDNet-style classifiers.

"""

import torch
import torch.nn as nn
import numpy as np

def get_output_shape(module, img_dim):
    # returns output shape
    dims = module(torch.rand(*(img_dim))).data.shape
    return dims

class ConvBasic(nn.Module):
        def __init__(self, chanIn, chanOut, k=3, s=1,
                    p=1, p_ceil_mode=False, bias=True):
            super(ConvBasic, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(chanIn, chanOut, kernel_size=k, stride=s,
                    padding=p, bias=False),
                nn.BatchNorm2d(chanOut),
                nn.ReLU(True) #in place
            )

        def forward(self, x):
            return self.conv(x)

class IntrClassif(nn.Module):
    # intermediate classifer head to be attached along the backbone
    # Inpsired by MSDNet classifiers (from HAPI):
    # https://github.com/kalviny/MSDNet-PyTorch/blob/master/models/msdnet.py

    def __init__(self,
            chanIn, input_shape, classes, bb_index):
        super(IntrClassif, self).__init__()

        # index for the position in the backbone layer
        self.bb_index = bb_index
        # input shape to automatically size linear layer
        self.input_shape = input_shape

        # intermediate conv channels
        #interChans = 128 # TODO reduce size for smaller nets
        interChans = 64
        # conv, bnorm, relu 1
        self.conv1 = ConvBasic(chanIn,interChans, k=3, s=2, p=[1,1])
        # conv, bnorm, relu 2
        self.conv2 = ConvBasic(interChans,interChans, k=3, s=2, p=[1,1])
        # avg pool - TODO check if global or local
        self.pool = nn.AvgPool2d(2) # NOTE - NOT in fpgaconvnet, only global

        self.linear_dim = np.prod(self._get_linear_size())
        print(f"Classif @ {self.bb_index} linear dim: {self.linear_dim}")
        # linear layer
        self.linear = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.linear_dim, classes)
                )

    def _get_linear_size(self):
        c1out = get_output_shape(self.conv1, self.input_shape)
        c2out = get_output_shape(self.conv2, c1out)
        pout = get_output_shape(self.pool, c2out)
        return pout

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        return self.linear(x)


class ResNet8v1(nn.Module):
    # FIXME Keras differences:
    # conv2d kernel init, kernel regularizer
    def __init__(self):
        super(ResNet8v1,self).__init__()
        # NOTE for CIFAR10 only (locked shape ip and op)
        self.input_shape=[1,3,32,32]
        self.num_classes=10
        self.num_filters=[16,32,64] # apparently 64 for standard resnet

        self.exit_threshold = 0.8

        self.fast_inference_mode = False

        # store backbone layers in list so can attach classifs at any point
        self.backbone = nn.ModuleList()
        self.exits = nn.ModuleList()

        self.exit_num = self._build_backbone()
        self._build_exits()

        # array of attachment points on the bb
        self.exit_array = [1]*self.exit_num
        # NOTE planning not to joint train exits
        # this is for testing purposes
        self.exit_loss_weights = [1.0]*self.exit_num

    # Builds the backbone for the ResNet model defined in ml perf benchmark
    def _build_backbone(self):
        # Flexible exit placement requires every layer be separate module
        # resnet structure:
        # input 0,1,2
        # b1 3,4,5,6,7
        # b1out (2out+7out) -> 8
        # b2 9,10,11,12,13
        # b2out (14+13out) -> 15
        # b3 16,17,18,19,20
        # b3out (21+20out) -> 22
        # pl 23
        bb_block_list = []
        bb_layers = [
            ### input conv ###
            nn.Conv2d(self.input_shape[1],self.num_filters[0],  #0
                kernel_size=3,stride=1,padding='same'),
            nn.BatchNorm2d(self.num_filters[0]),                #1
            nn.ReLU(),                                          #2
            ### block 1 ###
            nn.Conv2d(self.num_filters[0],self.num_filters[0],  #3
                kernel_size=3,stride=1,padding='same'),
            nn.BatchNorm2d(self.num_filters[0]),                #4
            nn.ReLU(),                                          #5
            nn.Conv2d(self.num_filters[0],self.num_filters[0],  #6
                kernel_size=3,stride=1,padding='same'),
            nn.BatchNorm2d(self.num_filters[0]),                #7
            # 2 res add
            nn.ReLU(),                                          #8
            ### block 2 ###
            nn.Conv2d(self.num_filters[0],self.num_filters[1],  #9
                kernel_size=3,stride=2,padding=[1,1]),
            nn.BatchNorm2d(self.num_filters[1]),                #10
            nn.ReLU(),                                          #11
            nn.Conv2d(self.num_filters[1],self.num_filters[1],  #12
                kernel_size=3,stride=1,padding='same'),
            nn.BatchNorm2d(self.num_filters[1]),                #13
            # b2 res conv
            nn.Conv2d(self.num_filters[0],self.num_filters[1],  #14
                kernel_size=1,stride=2,padding=0),
            # b2 res add here
            nn.ReLU(),                                          #15
            ### block 3 ###
            nn.Conv2d(self.num_filters[1],self.num_filters[2],  #16
                kernel_size=3,stride=2,padding=[1,1]),
            nn.BatchNorm2d(self.num_filters[2]),                #17
            nn.ReLU(),                                          #18
            nn.Conv2d(self.num_filters[2],self.num_filters[2],  #19
                kernel_size=3,stride=1,padding='same'),
            nn.BatchNorm2d(self.num_filters[2]),                #20
            # b2 res conv
            nn.Conv2d(self.num_filters[1],self.num_filters[2],  #21
                kernel_size=1,stride=2,padding=0),
            # b2 res add here
            nn.ReLU(),                                          #22
            # final pooling
            nn.AvgPool2d(8) # kernel 8,8 stride 8,8 -onnx model #23
            ]

        # add layers to module list
        for lyr in bb_layers:
            self.backbone.append(lyr)
        return len(bb_layers)

    def _build_exits(self):
        # exits - generate one per layer (this is many...)
        prev_shape = self.input_shape
        block_entry_shape = None
        for i,lyr in enumerate(self.backbone[0:-1]):
            if isinstance(lyr, nn.Conv2d):
                classif_channels_in = lyr.out_channels
            elif isinstance(lyr, nn.BatchNorm2d):
                classif_channels_in = lyr.num_features
            # determine output shape for classifier
            if i in [14,21]: # residual conv conns
                prev_shape = get_output_shape(lyr, block_entry_shape)
            else:
                prev_shape = get_output_shape(lyr, prev_shape)
                if i in [2,8,15]: #22
                    block_entry_shape = prev_shape
            #print(f"BB layer {i} shapes:",prev_shape)
            self.exits.append( IntrClassif(classif_channels_in,
                                    prev_shape,
                                    self.num_classes,i) )

        # append the final exit
        #pool_chans = self.backbone[-1].num_features # FIXME value is 64
        self.exits.append( nn.Sequential(
                nn.Flatten(),
                nn.Linear(64,self.num_classes,bias=True)
            ))
        return

    @torch.jit.unused
    def _forward_training(self, x):
        #TODO make jit compatible - not urgent
        #broken because returning list()
        res = []

        # connecting network in resnet order
        # input
        for idx in range(3):
            x = self.backbone[idx](x)
            res.append( self.exits[idx](x) )
        # b1
        xa=x
        for idx in range(3,8):
            xa = self.backbone[idx](xa)
            res.append( self.exits[idx](xa) )
        x = x + xa
        x = self.backbone[8](x)
        res.append( self.exits[8](x) )
        # b2
        xa=x
        for idx in range(9,14):
            xa = self.backbone[idx](xa)
            res.append( self.exits[idx](xa) )
        x = self.backbone[14](x) # res conv
        res.append( self.exits[14](x) )
        x = x + xa
        x = self.backbone[15](x)
        res.append( self.exits[15](x) )
        # b3
        xa=x
        for idx in range(16,21):
            xa = self.backbone[idx](xa)
            res.append( self.exits[idx](xa) )
        x = self.backbone[21](x) # res conv
        res.append( self.exits[21](x) )
        x = x + xa
        x = self.backbone[22](x)
        res.append( self.exits[22](x) )
        # classif
        x = self.backbone[23](x)
        res.append( self.exits[23](x) )
        # return final result containing all EE (and final exit)
        return res

    def exit_criterion_top1(self, x): #NOT for batch size > 1
        # evaluate the exit criterion on the result provided
        # return true if it can exit, false if it can't
        with torch.no_grad():
            pk = nn.functional.softmax(x, dim=-1)
            top1 = torch.max(pk) #x)
            return top1 > self.exit_threshold

    def forward(self, x):
        #std forward function - add var to distinguish be test and inf
        if self.fast_inference_mode:
            # resnet linking, exits everywhere
            # input
            for idx in range(3):
                x = self.backbone[idx](x)
                res = self.exits[idx](x)
                if self.exit_criterion_top1(res):
                    return res
            # b1
            xa=x
            for idx in range(3,8):
                xa = self.backbone[idx](xa)
                res = self.exits[idx](xa)
                if self.exit_criterion_top1(res):
                    return res
            x = x + xa
            x = self.backbone[8](x)
            res = self.exits[8](x)
            if self.exit_criterion_top1(res):
                return res
            # b2
            xa=x
            for idx in range(9,14):
                xa = self.backbone[idx](xa)
                res = self.exits[idx](xa)
                if self.exit_criterion_top1(res):
                    return res
            x = self.backbone[14](x) # res conv
            res = self.exits[14](x)
            if self.exit_criterion_top1(res):
                return res
            x = x + xa
            x = self.backbone[15](x)
            res = self.exits[15](x)
            if self.exit_criterion_top1(res):
                return res
            # b3
            xa=x
            for idx in range(16,21):
                xa = self.backbone[idx](xa)
                res = self.exits[idx](xa)
                if self.exit_criterion_top1(res):
                    return res
            x = self.backbone[21](x) # res conv
            res = self.exits[21](x)
            if self.exit_criterion_top1(res):
                return res
            x = x + xa
            x = self.backbone[22](x)
            res = self.exits[22](x)
            if self.exit_criterion_top1(res):
                return res
            # classif
            x = self.backbone[23](x)
            res = self.exits[23](x)
            return res
        else: #used for training
            #calculate all exits
            return self._forward_training(x)
