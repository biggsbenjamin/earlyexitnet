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

    def forward(self, x):
        y = [self.second_exit(x)] #NOTE put in list to reuse training code
        return y
