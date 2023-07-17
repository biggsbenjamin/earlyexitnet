import torch
import torch.nn as nn
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

