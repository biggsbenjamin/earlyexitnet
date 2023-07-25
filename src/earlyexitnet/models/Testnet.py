import torch
import torch.nn as nn

class BrnFirstExit(nn.Module):
    # backbone up to first exit of branchy-lenet model, and first exit
    #fcn version
    def __init__(self):
        super().__init__()
        self.first_exit = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5,stride=1,padding=4,bias=False),
            nn.MaxPool2d(2,stride=2,ceil_mode=False),
            nn.ReLU(),
            nn.Conv2d(5, 10, kernel_size=3,stride=1,padding=1,bias=False),
            nn.MaxPool2d(2,stride=2,ceil_mode=False),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(640, 10)#, bias=False)
        )
        #NOTE early and late exits
        self.exit_num=1

    def forward(self, x):
        y = [self.first_exit(x)] #NOTE put in list to reuse training code
        return y

class BrnSecondExit(nn.Module):
    # backbone up to second exit of branchy-lenet model, and second exit
    #fcn version (and se version) #NOTE not se version
    def __init__(self):
        super().__init__()
        self.second_exit = nn.Sequential(
            #bb common to both exits - removed if only exit required
            nn.Conv2d(1, 5, kernel_size=5,stride=1,padding=4),#,bias=False),
            nn.MaxPool2d(2,stride=2,ceil_mode=False),
            nn.ReLU(),

            nn.Conv2d(5, 10, kernel_size=5,stride=1,padding=4),#,bias=False),
            nn.MaxPool2d(2,stride=2,ceil_mode=False),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5,stride=1,padding=3),#,bias=False),
            nn.MaxPool2d(2,stride=2,ceil_mode=False),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(720, 84),#, bias=False),
            nn.Linear(84,10)#, bias=False)
        )
        #NOTE early and late exits
        self.exit_num=1

    def forward(self, x):
        y = [self.second_exit(x)] #NOTE put in list to reuse training code
        return y

class BrnFirstExit_se(nn.Module):
    # backbone up to first exit of branchy-lenet model, and first exit
    #se version, 2nd exit is same as fcn version
    #NOTE 2nd exit no longer same as fcn version
    def __init__(self):
        super().__init__()
        self.first_exit = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5,stride=1,padding=4),
            nn.MaxPool2d(2,stride=2,ceil_mode=False),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1280, 10)#, bias=False)
        )
        #NOTE early and late exits
        self.exit_num=1

    def forward(self, x):
        y = [self.first_exit(x)] #NOTE put in list to reuse training code
        return y

class Backbone_se(nn.Module):
    # backbone from start to end
    #se version 2nd exit
    def __init__(self):
        super().__init__()
        self.second_exit = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5,stride=1,padding=4),
            nn.MaxPool2d(2,stride=2,ceil_mode=False),
            nn.ReLU(),
            nn.Conv2d(5, 10, kernel_size=5,stride=1,padding=4),#,bias=False),
            nn.MaxPool2d(2,stride=2,ceil_mode=False),
            nn.ReLU(),
            #NOTE swapped a conv for a linear (more hw friendly)
            nn.Conv2d(10, 20, kernel_size=5,stride=1,padding=3),#,bias=False),
            nn.MaxPool2d(2,stride=2,ceil_mode=False),
            nn.ReLU(),
            #NOTE
            nn.Flatten(),
            #nn.Linear(1000, 84), #bias=False
            nn.Linear(720, 10)
        )
        #NOTE early and late exits
        self.exit_num=1

    def forward(self, x):
        y = [self.second_exit(x)] #NOTE put in list to reuse training code
        return y

class Testnet(nn.Module):
    #test network
    def __init__(self):
        super().__init__()
        self.testnet = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5,stride=1,padding=4,bias=False),
            #nn.MaxPool2d(2,stride=2,ceil_mode=False),
            #nn.ReLU(),
            #nn.Conv2d(5, 10, kernel_size=3,stride=1,padding=1,bias=False),
            #nn.MaxPool2d(2,stride=2,ceil_mode=False),
            #nn.ReLU(),
            nn.Flatten(),
            nn.Linear(5120, 10)#, bias=False)
        )
        #NOTE early and late exits
        self.exit_num=1

    def forward(self, x):
        y = [self.testnet(x)]#self.conv1(x)
        #y = self.relu1(y)
        #y = self.pool1(y)
        #y = self.conv2(y)
        #y = self.relu2(y)
        #y = self.pool2(y)
        #y = self.fltn(y)
        #y = self.fc1(y)
        #y = self.relu3(y)
        #y = self.fc2(y)
        #y = self.relu4(y)
        #y = self.fc3(y)
        #y = self.relu5(y)
        return y

############## ALEXNET BB ################
#FIXME
class Backbone_Alex(nn.Module):
    # Alexnet (no lrn, dropout)
    # CIFAR 10
    def __init__(self):
        super().__init__()
        self.second_exit = nn.Sequential(
            #conv1
            nn.Conv2d(3, 32, kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3,stride=2,ceil_mode=False),
            #LRN
            #conv2
            nn.Conv2d(32, 64, kernel_size=5,stride=1,padding=2),
            #original branch1 here
            nn.ReLU(),
            nn.MaxPool2d(3,stride=2,ceil_mode=False),
            #LRN
            #branch 1 moved here
            #conv3
            nn.Conv2d(64, 96, kernel_size=3,stride=1,padding=1),
            #original branch 2
            nn.ReLU(),
            #branch 2
            #conv4
            nn.Conv2d(96, 96, kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 64, kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3,stride=2,ceil_mode=False),
            nn.Flatten(),
            nn.Linear(576, 256), # originally 1024 (different ceil mode)
            nn.ReLU(),
            #dropout
            nn.Linear(256, 128),
            nn.ReLU(),
            #dropout
            nn.Linear(128, 10)
        )
        #NOTE early and late exits
        self.exit_num=1

    def forward(self, x):
        #NOTE put in list to reuse training code
        y = [self.second_exit(x)]
        return y

class TW_BB_SmallCNN(nn.Module):
    # MNIST
    def __init__(self):
        super().__init__()
        self.bb = nn.Sequential(
                nn.Conv2d(1, 32, 3),
                nn.ReLU(True),
                #branch1
                nn.Conv2d(32,32,3),
                nn.ReLU(True),
                nn.MaxPool2d(2,2),
                nn.Conv2d(32,64,3),
                nn.ReLU(True),
                #branch2
                nn.Conv2d(64,64,3),
                nn.ReLU(True),
                nn.MaxPool2d(2,2),
                nn.Flatten(),
                nn.Linear(64*4*4, 200),
                nn.ReLU(True),
                # drop
                nn.Linear(200,200),
                nn.ReLU(True),
                nn.Linear(200,10))
        #NOTE early and late exits
        self.exit_num=1

    def forward(self, x):
        return [self.bb(x)]

### SDN and l2stop - smallest net is resnet56... VGG won't fit on larger board


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super(BasicBlock, self).__init__()

        layers = nn.ModuleList()

        conv_layer = []
        conv_layer.append(nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False))
        #conv_layer.append(nn.BatchNorm2d(channels))
        conv_layer.append(nn.ReLU(inplace=True))
        conv_layer.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False))
        #conv_layer.append(nn.BatchNorm2d(channels))
        layers.append(nn.Sequential(*conv_layer))

        # going to have to cut this out...
        shortcut = nn.Sequential()

        if stride != 1 or in_channels != self.expansion*channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*channels, kernel_size=1, stride=stride, bias=False),
                #nn.BatchNorm2d(self.expansion*channels)
            )

        layers.append(shortcut)

        layers.append(nn.ReLU(inplace=True))

        self.layers = layers

    def forward(self, x):
        fwd = self.layers[0](x) # conv layers
        fwd += self.layers[1](x) # shortcut
        fwd = self.layers[2](fwd) # activation
        return fwd

class SDN_BB_ResNet(nn.Module):
    def __init__(self):
        super(SDN_BB_ResNet, self).__init__()
        #NOTE early and late exits
        self.exit_num=1

        params = {}

        ### taken from ResNet net arch ###
        #model_params = get_task_params(task)
        params['block_type'] = 'basic'
        params['num_blocks'] = [9]#,9,9]
        params['add_ic'] = [
              [0, 0, 0, 1, 0, 0, 0, 1, 0],
              [0, 0, 1, 0, 0, 0, 1, 0, 0],
              [0, 1, 0, 0, 0, 1, 0, 0, 0]] # 15, 30, 45, 60, 75, 90 percent of GFLOPs

        #model_name = '{}_resnet56'.format(task)

        params['network_type'] = 'resnet56'
        params['augment_training'] = True
        params['init_weights'] = True

        #params['task'] = 'cifar100'
        params['input_size'] = 32
        params['num_classes'] = 100
        ### taken from ResNet net arch ###


        self.num_blocks = params['num_blocks']
        self.num_classes = int(params['num_classes'])
        self.augment_training = params['augment_training']
        self.input_size = int(params['input_size'])
        self.block_type = params['block_type']
        #self.train_func = mf.cnn_train
        #self.test_func = mf.cnn_test
        self.in_channels = 16
        self.num_output =  1

        if self.block_type == 'basic':
            self.block = BasicBlock

        init_conv = []

        if self.input_size ==  32: # cifar10 and cifar100
            init_conv.append(nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False))
        elif self.input_size == 64: # tiny imagenet
            init_conv.append(nn.Conv2d(3, self.in_channels, kernel_size=3, stride=2, padding=1, bias=False))

        #init_conv.append(nn.BatchNorm2d(self.in_channels))
        init_conv.append(nn.ReLU(inplace=True))

        self.init_conv = nn.Sequential(*init_conv)

        self.layers = nn.ModuleList()
        self.layers.extend(self._make_layer(self.in_channels, block_id=0, stride=1))
        #self.layers.extend(self._make_layer(32, block_id=1, stride=2))
        #self.layers.extend(self._make_layer(64, block_id=2, stride=2))

        end_layers = []

        #end_layers.append(nn.AvgPool2d(kernel_size=8))
        end_layers.append(nn.MaxPool2d(kernel_size=8))
        end_layers.append(nn.Flatten())
        #end_layers.append(nn.Linear(64*self.block.expansion, self.num_classes))
        end_layers.append(nn.Linear(256*self.block.expansion, self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

        #self.initialize_weights()

    def _make_layer(self, channels, block_id, stride):
        num_blocks = int(self.num_blocks[block_id])
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(self.block(self.in_channels, channels, stride))
            self.in_channels = channels * self.block.expansion
        return layers

    def forward(self, x):
        out = self.init_conv(x)
        for layer in self.layers:
            out = layer(out)
        out = self.end_layers(out)
        return out

    #def initialize_weights(self):
    #    for m in self.modules():
    #        if isinstance(m, nn.Conv2d):
    #            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #        elif isinstance(m, nn.BatchNorm2d):
    #            m.weight.data.fill_(1)
    #            m.bias.data.zero_()
    #        elif isinstance(m, nn.Linear):
    #            m.weight.data.normal_(0, 0.01)
    #            m.bias.data.zero_()

### l2stop VGG  additional blocks
#class ConvBlock(nn.Module):
#    def __init__(self, conv_params):
#        super(ConvBlock, self).__init__()
#        input_channels = conv_params[0]
#        output_channels = conv_params[1]
#        max_pool_size = conv_params[2]
#        batch_norm = conv_params[3]
#
#        conv_layers = []
#        conv_layers.append(nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3,padding=1))
#
#        #if batch_norm:
#        #    conv_layers.append(nn.BatchNorm2d(output_channels))
#
#        conv_layers.append(nn.ReLU())
#
#        if max_pool_size > 1:
#            conv_layers.append(nn.MaxPool2d(kernel_size=max_pool_size))
#
#        self.layers = nn.Sequential(*conv_layers)
#
#    def forward(self, x):
#        fwd = self.layers(x)
#        return fwd
#
#class FcBlock(nn.Module):
#    def __init__(self, fc_params, flatten):
#        super(FcBlock, self).__init__()
#        input_size = int(fc_params[0])
#        output_size = int(fc_params[1])
#
#        fc_layers = []
#        if flatten:
#            fc_layers.append(nn.Flatten())
#
#        fc_layers.append(nn.Linear(input_size, output_size))
#        fc_layers.append(nn.ReLU())
#        #fc_layers.append(nn.Dropout(0.5))
#        self.layers = nn.Sequential(*fc_layers)
#
#    def forward(self, x):
#        fwd = self.layers(x)
#        return fwd
#
#class L2Stop_VGG_BB(nn.Module):
#    # cifar 10/100 or # tiny img net
#    def __init__(self):
#        super().__init__()
#
#        ###  taken from network_architectures.py (VGG) ###
#        #model_params = get_task_params(task)
#        params = {} # CIFAR100
#        params['input_size'] = 32
#        params['num_classes'] = 100
#
#        if params['input_size'] == 32:
#            params['fc_layers'] = [512, 512]
#        elif params['input_size'] == 64:
#            params['fc_layers'] = [2048, 1024]
#
#        params['conv_channels']  = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
#        #model_name = '{}_vgg16bn'.format(task)
#
#        # architecture params
#        params['network_type'] = 'vgg16'
#        params['max_pool_sizes'] = [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
#        params['conv_batch_norm'] = False #True
#        params['init_weights'] = False #True
#        params['augment_training'] = True #check for the stop rule policy
#        # ic is additional outputs
#        #model_params['add_ic'] = [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0]
#        ###  taken from network_architectures.py (VGG) ###
#
#        # read necessary parameters
#        self.input_size = int(params['input_size'])
#        self.num_classes = int(params['num_classes'])
#        self.conv_channels = params['conv_channels'] # the first element is input dimension
#        self.fc_layer_sizes = params['fc_layers']
#
#        # read or assign defaults to the rest
#        self.max_pool_sizes = params['max_pool_sizes']
#        self.conv_batch_norm = params['conv_batch_norm']
#        self.augment_training = params['augment_training']
#        self.init_weights = params['init_weights']
#        #self.train_func = mf.cnn_train
#        #self.test_func = mf.cnn_test
#        self.num_output = 1
#
#        self.init_conv = nn.Sequential() # just for compatibility with other models
#
#        self.layers = nn.ModuleList()
#        # add conv layers
#        input_channel = 3
#        cur_input_size = self.input_size
#        for layer_id, channel in enumerate(self.conv_channels):
#            if self.max_pool_sizes[layer_id] == 2:
#                cur_input_size = int(cur_input_size/2)
#            conv_params =  (input_channel, channel, self.max_pool_sizes[layer_id], self.conv_batch_norm)
#            self.layers.append(ConvBlock(conv_params))
#            input_channel = channel
#
#        fc_input_size = cur_input_size*cur_input_size*self.conv_channels[-1]
#
#        for layer_id, width in enumerate(self.fc_layer_sizes[:-1]):
#            fc_params = (fc_input_size, width)
#            flatten = False
#            if layer_id == 0:
#                flatten = True
#
#            self.layers.append(FcBlock(fc_params, flatten=flatten))
#            fc_input_size = width
#
#        end_layers = []
#        end_layers.append(nn.Linear(fc_input_size, self.fc_layer_sizes[-1]))
#        #end_layers.append(nn.Dropout(0.5))
#        end_layers.append(nn.Linear(self.fc_layer_sizes[-1], self.num_classes))
#        self.end_layers = nn.Sequential(*end_layers)
#
#        #if self.init_weights:
#            #self.initialize_weights()
#
#    def forward(self, x):
#        fwd = self.init_conv(x)
#
#        for layer in self.layers:
#            fwd = layer(fwd)
#
#        fwd = self.end_layers(fwd)
#        return [fwd]
#
#    #def initialize_weights(self):
#    #    for m in self.modules():
#    #        if isinstance(m,nn.Conv2d):
#    #            n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
#    #            m.weight.data.normal_(0,math.sqrt(2. / n))
#    #            if m.bias is not None:
#    #                m.bias.data.zero_()
#    #        elif isinstance(m,nn.BatchNorm2d):
#    #            m.weight.data.fill_(1)
#    #            m.bias.data.zero_()
#    #        elif isinstance(m,nn.Linear):
#    #            m.weight.data.normal_(0,0.01)
#    #            m.bias.data.zero_()
