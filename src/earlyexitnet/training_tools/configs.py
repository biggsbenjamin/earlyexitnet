"""
Selection of configurations for training.

Including class for definitions of known pytorch optimiser configs.
"""

from torch import optim

from dataclasses import dataclass, field
from typing import List, Any

def select_optimiser(opt_str):
    # select optimiser configuration based on cli string input
    if opt_str == 'adam-brn':
        opt_cfg = OptBranchyNetAdam()
    elif opt_str == 'adam-wd':
        opt_cfg = OptAdamWeightDecay()
    elif opt_str == 'adam-wd-mul-sched':
        opt_cfg = OptAdamWDMulSched()
    elif opt_str == 'sgd-plat-sched':
        opt_cfg = OptSGDPlateauSched()
    elif opt_str == 'adam-wd-plat-sched':
        opt_cfg=OptAdamWDPlatSched()
    else:
        raise NotImplementedError(f"Optimiser config:{opt_str} not implement.")
    return opt_cfg

@dataclass
class OptConfig:
    name: str = 'placeholder'
    lr: float = 0.001
    opt_class: Any = optim.SGD
    lr_sched_class: Any = None

    # NOTE replace me in child configs, TODO method auto assign?
    def get_opt(self,params):
        opt = self.opt_class(params,lr=self.lr)
        lr_sched = None
        return opt, lr_sched

# NOTE based on config from branchynet paper
@dataclass
class OptBranchyNetAdam(OptConfig):
    name: str ='adam-brn'
    betas: List[float] = field(default_factory=lambda: [0.99,0.999])
    opt_class: Any = optim.Adam

    def get_opt(self,params):
        opt = self.opt_class(params,lr=self.lr,betas=self.betas)
        lr_sched = None
        return opt, lr_sched

@dataclass
class OptAdamWeightDecay(OptBranchyNetAdam):
    name: str ='adam-wd'
    lr: float = 0.00094
    weight_decay: float =0.00011

    def get_opt(self,params):
        opt = self.opt_class(params,lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay)
        lr_sched = None
        return opt, lr_sched

# NOTE based on information from:
# https://github.com/mlcommons/tiny/blob/master/benchmark/training/image_classification/train.py
@dataclass
class OptAdamWDMulSched(OptAdamWeightDecay):
    name: str ='adam-wd-mul-sched'

    lr_sched_class: Any =optim.lr_scheduler.MultiplicativeLR
    lr_lambda: Any =lambda epoch:0.99
    verbose: bool =True

    def get_opt(self,params):
        opt = self.opt_class(params,lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay)
        lr_sched = self.lr_sched_class(opt,lr_lambda=lr_lambda,verbose=self.verbose)
        return opt, lr_sched

@dataclass
class OptAdamWDPlatSched(OptAdamWeightDecay):
    name: str ='adam-wd-plat-sched'
    lr: float = 0.1

    lr_sched_class: Any = optim.lr_scheduler.ReduceLROnPlateau
    factor: float =0.1
    patience: int =3
    threshold: float=0.001
    mode: str ='max'
    verbose: bool =True

    def get_opt(self,params):
        opt = self.opt_class(params,lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay)
        lr_sched = self.lr_sched_class(opt,
                factor=self.factor,
                patience=self.patience,
                threshold=self.threshold,
                mode=self.mode,
                verbose=self.verbose)
        return opt, lr_sched

# NOTE taken from resnet18 cifar 10 configuration:
# https://huggingface.co/edadaltocg/resnet18_cifar10
@dataclass
class OptSGDPlateauSched(OptConfig):
    name: str ='sgd-plat-sched'
    lr: float = 0.1
    momentum: float =0.9
    weight_decay: float =0.0005
    nesterov: bool =True
    opt_class: Any=optim.SGD

    lr_sched_class: Any =optim.lr_scheduler.ReduceLROnPlateau
    factor: float =0.1
    patience: int =3
    threshold: float=0.001
    mode: str ='max'
    verbose: bool =True

    def get_opt(self,params):
        opt = self.opt_class(params,lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=self.nesterov
                )
        lr_sched = self.lr_sched_class(opt,
                factor=self.factor,
                patience=self.patience,
                threshold=self.threshold,
                mode=self.mode,
                verbose=self.verbose)
        return opt, lr_sched

