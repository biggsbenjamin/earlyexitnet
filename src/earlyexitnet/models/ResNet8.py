import torch
import torch.nn as nn
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        layers = nn.ModuleList()
        conv_layer = []
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
