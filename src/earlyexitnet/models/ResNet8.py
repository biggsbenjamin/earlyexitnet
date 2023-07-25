import torch
import torch.nn as nn

from earlyexitnet.tools import get_output_shape

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        layers = nn.ModuleList()
        conv_layer = []

        # manually matching padding
        if stride != 1:
            c0 = nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=stride, padding=0, bias=True)
            conv_layer.append(nn.ZeroPad2d((0,1,0,1)))
        else:
            c0 = nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=stride, padding=1, bias=True)
        conv_layer.append(c0)
        conv_layer.append(nn.ReLU(inplace=True))

        c1= nn.Conv2d(out_channels, out_channels,
                  kernel_size=3, stride=1, padding=1, bias=True)
        conv_layer.append(c1)
        layers.append(nn.Sequential(*conv_layer))

        #shortcut
        shortcut = nn.Sequential()
        if stride != 1:
            c2=nn.Conv2d(in_channels, out_channels,
                         kernel_size=1, stride=stride,
                         padding=0,bias=True)
            shortcut = nn.Sequential(c2)
            self.conv_list = [c2,c0,c1]
        else:
            self.conv_list = [c0,c1]

        layers.append(shortcut)
        # activation after add
        layers.append(nn.ReLU(inplace=True))

        self.layers = layers

    def forward(self, x):
        fwd = self.layers[0](x) # conv layers
        fwd += self.layers[1](x) # shortcut
        fwd = self.layers[2](fwd) # activation
        return fwd


#mlperf tiny cifar10 model
class ResNet8(nn.Module):
    def __init__(self):
        super(ResNet8, self).__init__()
        self.exit_num=1

        self.input_size=32
        self.num_classes=10

        self.in_chans=16

        # module list of all convolutions
        self.wb_list = nn.ModuleList()

        c0 = nn.Conv2d(3, self.in_chans,
                      kernel_size=3, stride=1,
                      padding=1, bias=True)

        self.init_conv = nn.Sequential(
            c0,
            nn.ReLU(inplace=True)
        )

        self.wb_list.append(c0)

        self.backbone = nn.ModuleList()
        self.in_chan_sizes = [16,16,32]
        self.out_chan_sizes = [16,32,64]
        self.strides = [1,2,2]

        self._build_backbone()

        end_layers = []
        end_layers.append(nn.AvgPool2d(kernel_size=8))
        #end_layers.append(nn.MaxPool2d(kernel_size=8))
        end_layers.append(nn.Flatten())
        l0=nn.Linear(64, self.num_classes)
        end_layers.append(l0)
        #end_layers.append(nn.Linear(256, self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)
        self.wb_list.append(l0)

        print("w&b layer len",len(self.wb_list))

    def _build_backbone(self):
        for ic,oc,s in \
                zip(self.in_chan_sizes,self.out_chan_sizes,self.strides):
            block=BasicBlock(
                in_channels=ic,
                out_channels=oc,
                stride=s
            )
            self.backbone.append(block)
            for c in block.conv_list:
                self.wb_list.append(c)
        return

    def forward(self, x):
        y = self.init_conv(x)
        for b in self.backbone:
            y = b(y)
        y = self.end_layers(y)
        return [y]


# ResNet8 backbone for training experiments - weights should be transferable
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        layers = nn.ModuleList()
        conv_layer=[]

        c0 = nn.Conv2d(in_channels, out_channels,
                  kernel_size=3, stride=stride, padding=1, bias=True)
        conv_layer.append(c0)
        # BN layer to be fused
        bn0 = nn.BatchNorm2d(out_channels)
        conv_layer.append(bn0)
        conv_layer.append(nn.ReLU(inplace=True))

        c1= nn.Conv2d(out_channels, out_channels,
                  kernel_size=3, stride=1, padding=1, bias=True)
        conv_layer.append(c1)
        # another BN layer to be fused
        bn1 = nn.BatchNorm2d(out_channels)
        conv_layer.append(bn1)
        # append main conv branch
        layers.append(nn.Sequential(*conv_layer))
        # Pair the conv and batchnorm in list for later fusion
        self.conv_bn_pairs = [[c0,bn0],[c1,bn1]]

        # add residual path
        shortcut = nn.Sequential()
        if stride != 1:
            c2=nn.Conv2d(in_channels, out_channels,
                         kernel_size=1, stride=stride,
                         padding=0,bias=True)
            shortcut = nn.Sequential(c2)

        layers.append(shortcut)
        # activation after add
        layers.append(nn.ReLU(inplace=True))
        # save block layer module list
        self.layers = layers

    def forward(self, x):
        fwd = self.layers[0](x) # conv layers
        fwd += self.layers[1](x) # shortcut
        fwd = self.layers[2](fwd) # activation
        return fwd

class ResNet8_backbone(nn.Module):
    def __init__(self):
        super(ResNet8_backbone, self).__init__()
        self.exit_num=1

        self.input_size=32
        self.num_classes=10
        self.in_chans=16

        c0 = nn.Conv2d(3, self.in_chans,
                      kernel_size=3, stride=1,
                      padding=1, bias=True)

        self.init_conv = nn.Sequential(
            c0,
            nn.BatchNorm2d(self.in_chans),
            nn.ReLU(inplace=True)
        )

        self.backbone = nn.ModuleList()

        self.in_chan_sizes = [16,16,32]
        self.out_chan_sizes = [16,32,64]
        self.strides = [1,2,2]

        self._build_backbone()

        end_layers = []
        end_layers.append(nn.AvgPool2d(kernel_size=8))
        end_layers.append(nn.Flatten())
        l0=nn.Linear(64, self.num_classes)
        end_layers.append(l0)
        #end_layers.append(nn.Linear(256, self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)
        # init weights+biases according to mlperf tiny
        self._weight_init()

    def _build_backbone(self):
        for ic,oc,s in \
                zip(self.in_chan_sizes,self.out_chan_sizes,self.strides):
            block = ResBlock(in_channels=ic,out_channels=oc,stride=s)
            self.backbone.append(block)
        return

    def _weight_init(self):
        def _apply_init(m):
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
        self.apply(_apply_init)

    def forward(self, x):
        y = self.init_conv(x)
        for b in self.backbone:
            y = b(y)
        y = self.end_layers(y)
        return [y]

# Early-Exit ResNet8 and classifier structures
## FlexDNN Classifier
# from paper:
# 3 dw sep conv (with activation lyrs)
# 2 pool layers
# 1 fc layer
# part of the arch search removes conv blocks when theres no impact on accu
    # I guess this would be later in the network

## MSDNet Classifier
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

    def __init__(self,chanIn,input_shape,classes,bb_index):
        super(IntrClassif, self).__init__()

        # index for the position in the backbone layer
        self.bb_index = bb_index
        # input shape to automatically size linear layer
        self.input_shape = input_shape

        # intermediate conv channels
        #interChans = 128 # TODO reduce size for smaller nets
        interChans = 32
        # conv, bnorm, relu 1
        self.conv1 = ConvBasic(chanIn,interChans, k=3, s=2, p=[1,1])
        # conv, bnorm, relu 2
        self.conv2 = ConvBasic(interChans,interChans, k=3, s=2, p=[1,1])
        # avg pool - TODO check if global or local
        self.pool = nn.AvgPool2d(2) # NOTE - NOT in fpgaconvnet, only global

        self.linear_dim = int(torch.prod(torch.tensor(self._get_linear_size())))
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

class ResNet8_2EE(ResNet8_backbone):
    # basic early exit network for resnet8
    def __init__(self):
        super(ResNet8_2EE, self).__init__()

        # NOTE structure:
        # init conv -> exit1
        # self.backbone
        # self.end_layer (avg pool, flatten, linear)

        self.exits = nn.ModuleList()
        # weighting for each exit when summing loss
        self.input_shape=[1,3,32,32]

        self.exit_num=2
        self.fast_inference_mode = False
        self.exit_loss_weights = [1.0, 1.0]
        self.exit_threshold = torch.tensor([0.8], dtype=torch.float32)
        self._build_exits()

    def _build_exits(self): #adding early exits/branches
        # TODO generalise exit placement for multi exit
        # early exit 1
        previous_shape = get_output_shape(self.init_conv,self.input_shape)
        ee1 = IntrClassif(16,previous_shape,self.num_classes,0)
        self.exits.append(ee1)

        #final exit
        self.exits.append(self.end_layers)

    # @torch.jit.unused #decorator to skip jit comp
    # def _forward_training(self, x):
    #     # TODO make jit compatible - not urgent
    #     # NOTE broken because returning list()
    #     res = []
    #     y = self.init_conv(x)
    #     res.append(self.exits[0](y))
    #     # compute remaining backbone layers
    #     for b in self.backbone:
    #         y = b(y)
    #     # final exit
    #     y = self.end_layers(y)
    #     res.append(y)

    #     return res
    
    
    def _forward_training(self, x):
        res = None
        num_batch = x.size(0)
        for bb, ee in zip(self.backbone, self.exits):
            x = bb(x)
            tmp = ee(x)
            num_classes = tmp.size(1)
            
            tmp = tmp.reshape(1, num_batch, num_classes) # resize from [B, C] to [1, B, C] to then stack it along the first dimension
            res = tmp if res is None else torch.cat((res,tmp), dim=0)
        return res

    def exit_criterion_top1(self, x): #NOT for batch size > 1
        with torch.no_grad():
            pk = nn.functional.softmax(x, dim=-1)
            top1 = torch.max(pk)
            return top1 > self.exit_threshold

    def forward(self, x):
        #std forward function
        if self.fast_inference_mode:
            #for bb, ee in zip(self.backbone, self.exits):
            #    x = bb(x)
            #    res = ee(x) #res not changed by exit criterion
            #    if self.exit_criterion_top1(res):
            #        return res
            y = self.init_conv(x)
            res = self.exits[0](y)
            if self.exit_criterion_top1(res):
                return res
            # compute remaining backbone layers
            for b in self.backbone:
                y = b(y)
            # final exit
            res = self.exits[1](y)
            return res

        else: # NOTE used for training
            # calculate all exits
            return self._forward_training(x)

    def set_fast_inf_mode(self, mode=True):
        if mode:
            self.eval()
        self.fast_inference_mode = mode
