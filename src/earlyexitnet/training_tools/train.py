'''
Training functions for early exit networks.

Includes backbone only training, joint training and early-exit only training.

Added non-ee training from other script for testing purposes.
'''

# importing early exit models
from earlyexitnet.models.Branchynet import \
    ConvPoolAc,B_Lenet,B_Lenet_fcn,B_Lenet_se, B_Lenet_cifar

# importing non EE models
# NOTE models that don't output a list (over exits) won't work
#from earlyexitnet.models.Lenet import Lenet
from earlyexitnet.models.Testnet import \
    Testnet, BrnFirstExit, BrnSecondExit, Backbone_se
# Non ee models
from earlyexitnet.models.ResNet8 import ResNet8,ResNet8_backbone
# EE resnet models for cifar10
from earlyexitnet.models.ResNet8 import ResNet8_2EE

# Import optimiser configs
from earlyexitnet.training_tools import configs

# importing accu + loss trackers and dataloader classes
from earlyexitnet.tools import LossTracker, AccuTracker
from earlyexitnet.tools import save_model, load_model, shape_test

# pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import \
    DataLoader, Dataset, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms

# general imports
import os
import numpy as np
from datetime import datetime as dt

# get number of correct values from predictions and labels
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def get_model(model_str):
    #set up the model specified in args
    if model_str == 'lenet':
        # FIXME
        raise NameError("Training this model not supported")
        model = Lenet()
    elif model_str == 'testnet':
        model = Testnet()
    elif model_str == 'brnfirst': #fcn version
        model = BrnFirstExit()
    elif model_str == 'brnsecond': #fcn version
        model = BrnSecondExit()
    elif model_str == 'brnfirst_se': #se version
        model = BrnFirstExit_se()
    elif model_str == 'backbone_se': #se backbone (for baseline)
        model = Backbone_se()
    elif model_str == 'b_lenet':
        model = B_Lenet()
    elif model_str == 'b_lenet_fcn':
        model = B_Lenet_fcn()
    elif model_str == 'b_lenet_se':
        model = B_Lenet_se()
    elif model_str == 'b_lenet_cifar':
        model = B_Lenet_cifar()
        #print(shape_test(model, [3,32,32], [1])) #output is not one hot encoded
    elif model_str == 'resnet8':
        model = ResNet8()
    elif model_str == 'resnet8_bb':
        model = ResNet8_backbone()
    elif model_str == 'resnet8_2ee':
        model = ResNet8_2EE()
    else:
        raise NameError("Model not supported, check name:",model_str)
    print("Model done:", model_str)
    return model

class Trainer:
    def __init__(self, model, train_dl, valid_dl, batch_size, save_path,
                 loss_f=nn.CrossEntropyLoss(), exits=1,
                 backbone_epochs=50, exit_epochs=50,joint_epochs=100,
                 backbone_opt_cfg='adam-brn',exit_opt_cfg='adam-brn',
                 joint_opt_cfg='adam-brn',
                 device=None,
                 pretrained_path=None,
                 validation_frequency=1,
                 #dat_norm=False):
                 ):
        # assign nn model to train
        self.model=model
        # assign training and validation data loaders
        self.train_dl=train_dl
        self.train_len=len(self.train_dl)
        self.valid_dl=valid_dl
        self.valid_len=len(self.valid_dl)
        # assign training batch size
        self.batch_size=batch_size
        # assign loss function (cross entropy loss)
        self.loss_f=loss_f
        # number of exits
        self.exits=exits

        # saving other configs to self
        self.save_path = save_path
        # epochs to train bb only
        self.backbone_epochs=backbone_epochs
        # epochs to train exits
        self.exit_epochs = exit_epochs
        # epochs to train exits and bb jointly
        self.joint_epochs = joint_epochs
        # device to train on
        self.device = device
        # path to pretrained model
        self.pretrained_path = pretrained_path
        # TODO different operations if model is full or just bb
        # sets frequency model runs validation AND is checkpointed
        self.validation_frequency = validation_frequency

        # sets optimiser configs
        self.backbone_opt_cfg   = configs.select_optimiser(backbone_opt_cfg)
        self.exit_opt_cfg       = configs.select_optimiser(exit_opt_cfg)
        self.joint_opt_cfg      = configs.select_optimiser(joint_opt_cfg)

        self.ee_flag=False
        if self.exits > 1:
            # early-exit network!
            self.ee_flag=True

        # set up trackers and best values in self
        self.best_val_loss  = None
        self.best_val_accu  = None
        self.train_loss_trk = None
        self.train_accu_trk = None
        self.valid_loss_trk = None
        self.valid_accu_trk = None

        # Epoch plots
        self.bb_train_epcs = [i+1 for i in \
                range(self.backbone_epochs)]
        self.bb_train_loss = [0]*len(self.bb_train_epcs)
        self.bb_train_accu = [0]*len(self.bb_train_epcs)

        self.bb_valid_epcs = []
        self.bb_valid_loss = []
        self.bb_valid_accu = []

    # set up the trackers generic
    def _tracker_init(self, training_exits):
        # training exits bool flag
        if training_exits:
            e_num = self.exits
        else:
            e_num = 1
        self.best_val_loss = {"loss": [1.0]*e_num,
                "savepoint":'',"epoch":-1}
        self.best_val_accu = {"accuracy": [0.0]*e_num,
                "savepoint":'',"epoch":-1}
        self.train_loss_trk = LossTracker(
            self.train_dl.batch_size,bins=e_num)
        self.train_accu_trk = AccuTracker(
            self.train_dl.batch_size,bins=e_num)
        self.valid_loss_trk = LossTracker(
            self.valid_dl.batch_size,bins=e_num)
        self.valid_accu_trk = AccuTracker(
            self.valid_dl.batch_size,bins=e_num)

    def _train_loop_loss_ex(self,opt,results,yb):
        # calculate loss, ba
        raw_losses = [self.loss_f(res,yb) \
                      for res in results]
        losses = [weighting * raw_loss
                    for weighting, raw_loss in \
                  zip(self.model.exit_loss_weights,
                      raw_losses)]
        opt.zero_grad()
        #backprop
        for loss in losses[:-1]:
            #ee losses need to keep graph
            loss.backward(retain_graph=True)
        #final loss, graph not required
        losses[-1].backward()
        opt.step()
        #raw losses
        self.train_loss_trk.add_loss(
            [exit_loss.item() for \
             exit_loss in raw_losses])
        self.train_accu_trk.update_correct(
            results,yb)

    def _train_loop_loss_bb(self,opt,results,yb):
        #TODO add backbone only method to brn class
        loss = self.loss_f(results[-1], yb)
        #print(f"Loss: {loss}")
        opt.zero_grad()
        loss.backward()
        opt.step()
        # update trackers
        self.train_loss_trk.add_loss(loss.item())
        self.train_accu_trk.update_correct(
            results[-1],yb)
    def _val_loop_get_best_ex(self,val_loss_avg,val_accu_avg,
            savepoint,curr_epoch):
        el_total=0.0
        bl_total=0.0
        for exit_loss,best_loss,l_w in \
                zip(val_loss_avg,
                    self.best_val_loss["loss"],
                    self.model.exit_loss_weights):
            el_total+=exit_loss*l_w
            bl_total+=best_loss*l_w
        #selecting "best" network
        if el_total < bl_total:
            self.best_val_loss["loss"] = val_loss_avg
            self.best_val_loss["savepoint"] = savepoint
            self.best_val_loss["epoch"] = curr_epoch
        # determining exit accuracy, current and best
        ea_total=0.0
        ba_total=0.0
        for exit_accu, best_accu,l_w in \
                zip(val_accu_avg,
                    self.best_val_accu["accuracy"],
                    self.model.exit_loss_weights):
            ea_total+=exit_accu*l_w
            ba_total+=best_accu*l_w
        #selecting "best" network
        if ea_total > ba_total:
            self.best_val_accu["accuracy"]=val_accu_avg
            self.best_val_accu["savepoint"]=savepoint
            self.best_val_accu["epoch"] = curr_epoch
    # get best validation accu and loss - bb ver
    def _val_loop_get_best_bb(self,val_loss_avg,val_accu_avg,
            savepoint,curr_epoch):
        if val_loss_avg < self.best_val_loss["loss"][0]:
            self.best_val_loss["loss"][0] = val_loss_avg
            self.best_val_loss["savepoint"]=savepoint
            self.best_val_loss["epoch"] = curr_epoch
        if val_accu_avg > self.best_val_accu["accuracy"][0]:
            self.best_val_accu["accuracy"][0] = val_accu_avg
            self.best_val_accu["savepoint"] = savepoint
            self.best_val_accu["epoch"] = curr_epoch

    def _train_ee(self, training_exits, max_epochs, epoch_thresh,
            opt,internal_folder='',prefix='blank_prefix',
            # bb vs exit&jnt functions for loss, etc.
            loss_calc_f=None,validation_get_best_f=None,
            lr_sched=None # learning rate scheduler
            ):
        # training exits bool
        if training_exits:
            e_num = self.exits
            print(f"Training {e_num} exits")
        else:
            e_num = 1
            print(f"Training final exit")

        # set device model - should be cpu as default
        if self.device is None:
            self.device = torch.device("cpu")
        self.model.to(self.device)

        for epoch in range(max_epochs):
            self.model.train()
            print("Starting epoch: {}".format(epoch+1),
                  end="... ", flush=True)
            self.train_loss_trk.reset_tracker()
            self.train_accu_trk.reset_tracker()

            #training loop
            for xb, yb in self.train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                results = self.model(xb)
                # calculate and back prop loss for exit(s)
                loss_calc_f(opt,results,yb)
            # update training loss and accuracy averages
            tr_loss_avg = self.train_loss_trk.get_avg(
                return_list=True)
            t1acc = self.train_accu_trk.get_accu(
                return_list=True)
            if not training_exits:
                tr_loss_avg = tr_loss_avg[-1]
                t1acc = t1acc[-1]
                # 'log' the training loss
                self.bb_train_loss[epoch] = tr_loss_avg
                self.bb_train_accu[epoch] = t1acc

            # print the training info
            print("raw t loss:{} t1acc:{}".format(tr_loss_avg,t1acc))

            if epoch % self.validation_frequency == 0 or \
                    (epoch+1) == max_epochs:
                #### validation and saving loop ####
                self.valid_loss_trk.reset_tracker()
                self.valid_accu_trk.reset_tracker()
                self.model.eval()
                with torch.no_grad():
                    for xb,yb in self.valid_dl:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        res_v = self.model(xb)
                        if training_exits:
                            self.valid_loss_trk.add_loss(
                                [self.loss_f(exit, yb) \
                                 for exit in res_v])
                            self.valid_accu_trk.update_correct(
                                res_v,yb)
                        else:
                            vloss=self.loss_f(res_v[-1], yb)
                            self.valid_loss_trk.add_loss(vloss)
                            self.valid_accu_trk.update_correct(
                                res_v[-1],yb)
                # average validation loss and accuracy
                val_loss_avg = self.valid_loss_trk.get_avg(
                    return_list=True)
                val_accu_avg = self.valid_accu_trk.get_accu(
                    return_list=True)
                if not training_exits: # should be last of 1
                    val_loss_avg = val_loss_avg[-1]
                    val_accu_avg = val_accu_avg[-1]
                    # 'log' the values TODO add non bb method
                    self.bb_valid_epcs.append(epoch)
                    self.bb_valid_loss.append(val_loss_avg)
                    self.bb_valid_accu.append(val_accu_avg)

                    if isinstance(lr_sched, optim.lr_scheduler.ReduceLROnPlateau):
                        lr_sched.step(val_accu_avg)

                # debugging print - TODO log this rather than print
                print("raw v loss:{} v accu:{}".format(val_loss_avg,val_accu_avg))
                # saving state of network
                savepoint = save_model(
                    self.model,
                    os.path.join(self.save_path,internal_folder),
                    file_prefix=prefix+'-e'+str(epoch+1),
                    opt=opt,tloss=tr_loss_avg,vloss=val_loss_avg,
                    taccu=t1acc,vaccu=val_accu_avg)

                # determining exit loss, current and best
                validation_get_best_f(val_loss_avg,val_accu_avg,savepoint,epoch)
                # TODO log tr and vl data against epoch to file
                # print to some file, csv or otherwise

                #### validation and saving loop ####

            # add lr sched step here for adam-wd
            if isinstance(lr_sched, optim.lr_scheduler.MultiplicativeLR):
                lr_sched.step()

            # Early-terminate training when accuracy stops improving
            # TODO add this back in
            #if epoch - self.best_val_accu["epoch"] > epoch_thresh*self.validation_frequency:
            #    print(f"EARLY TERMINATION OF TRAINING  @ {epoch}")
            #    savepoint = save_model(
            #        self.model,
            #        os.path.join(self.save_path,internal_folder),
            #        file_prefix=prefix+'-e'+str(epoch+1),
            #        opt=opt,tloss=tr_loss_avg,vloss=val_loss_avg,
            #        taccu=t1acc,vaccu=val_accu_avg)
            #    break

        # final print - TODO log this
        print("BEST* VAL LOSS: ",self.best_val_loss["loss"],
              " for epoch: ",self.best_val_loss["savepoint"])
        print("BEST* VAL ACCU: ",self.best_val_accu["accuracy"],
              " for epoch: ",self.best_val_accu["savepoint"])
        # return highest val accuracy and most recent savepoint
        return self.best_val_accu["savepoint"], savepoint

    def train_backbone(self, internal_folder=None):
        prefix='backbone'
        if self.pretrained_path is not None:
            # load previous model to continue training
            load_model(self.model, self.pretrained_path)
            prefix='bb-exist'
        if self.exits>1:
            # selecting only backbone params
            params = [{'params': self.model.backbone.parameters()},
                    {'params': self.model.exits[-1].parameters()}
                    ]
        else:
            params = self.model.parameters()

        # get optimiser and shceduler from config
        opt,lr_sched = self.backbone_opt_cfg.get_opt(params)
        # set up single exit tracker
        self._tracker_init(training_exits=False)
        # run the actual training loop
        best_bb_pth, last_pth = self._train_ee(
            training_exits=False,
            max_epochs=self.backbone_epochs,
            epoch_thresh=100,
            opt=opt,internal_folder=internal_folder,
            prefix=prefix,
            loss_calc_f=self._train_loop_loss_bb,
            validation_get_best_f=self._val_loop_get_best_bb,
            lr_sched=lr_sched
            )
        return best_bb_pth, last_pth

    def train_exits(self):
        # TODO specify exits to train - list indices?, bool list mask?
        #train the exits alone

        #Adam algo - step size alpha=0.001
        #exponetial decay rates for 1st & 2nd moment: 0.99, 0.999

        raise NotImplementedError("Not method training exit")
        return #something trained

    def train_joint(self, pretrain_backbone=True):
        timestamp = dt.now().strftime("%Y-%m-%d_%H%M%S")
        if self.pretrained_path is None:
            if pretrain_backbone:
                print("PRETRAINING BACKBONE FROM SCRATCH")
                folder_path = 'pre_Trn_bb_' + timestamp
                # Training backbone!
                best_bb_path,_ = self.train_backbone(
                    internal_folder=folder_path)
                #train the rest...
                print("LOADING BEST BACKBONE:",best_bb_path)
                load_model(self.model, best_bb_path)
                print("JOINT TRAINING WITH PRETRAINED BACKBONE")
                prefix = 'pretrn-joint'
            else:
                #jointly trains backbone and exits from scratch
                print("JOINT TRAINING FROM SCRATCH")
                folder_path = 'jnt_fr_scrcth' + timestamp
                prefix = 'joint'
        else: # pretrained model
            # NOTE assuming just backbone trained
            load_model(self.model, self.pretrained_path,strict=False)
            print("JOINT TRAINING USING EXISTING BB")
            folder_path = 'jnt_fr_exstng' + timestamp
            prefix = 'bbexst-joint'

        #set up the joint optimiser - NOTE branchynet default
        params=self.model.parameters()
        opt,lr_sched = self.joint_opt_cfg.get_opt(params)
        # save path - folder and distinct prefix
        #spth = os.path.join(self.save_path,folder_path)
        # set up trackers multi exit
        self._tracker_init(training_exits=True)
        # run the actual training loop
        best_bb_pth, last_pth = self._train_ee(
            training_exits=True,
            max_epochs=self.joint_epochs,
            epoch_thresh=10,
            opt=opt, internal_folder=folder_path,
            prefix=prefix,
            loss_calc_f=self._train_loop_loss_ex,
            validation_get_best_f=self._val_loop_get_best_ex,
            lr_sched=lr_sched
            )
        return best_bb_pth, last_pth
