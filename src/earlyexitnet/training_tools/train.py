'''
Training functions for early exit networks.

Includes backbone only training, joint training and early-exit only training.

Added non-ee training from other script for testing purposes.
'''

# importing early exit models
from earlyexitnet.models.Branchynet import ConvPoolAc,B_Lenet,B_Lenet_fcn,B_Lenet_se, B_Lenet_cifar

# importing non EE models
from earlyexitnet.models.Lenet import Lenet
from earlyexitnet.models.Testnet import Testnet, BrnFirstExit, BrnSecondExit, Backbone_se

# importing accu + loss trackers and dataloader classes
from earlyexitnet.tools import LossTracker, AccuTracker
from earlyexitnet.tools import save_model, load_model, shape_test

# pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms

import os
import numpy as np
from datetime import datetime as dt

# get number of correct values from predictions and labels
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


class Trainer:
    def __init__(self, model, train_dl, valid_dl, batch_size, save_path,
                 loss_f=nn.CrossEntropyLoss(), exits=1,
                 backbone_epochs=50, exit_epochs=50,joint_epochs=100,
                 #pretrain_backbone=True, dat_norm=False):
                 ):
        self.model=model
        self.train_dl=train_dl
        self.valid_dl=valid_dl
        self.batch_size=batch_size
        self.loss_f=loss_f
        self.exits=exits

        # saving to self
        self.save_path = save_path
        self.backbone_epochs=backbone_epochs
        self.exit_epochs = exit_epochs
        self.joint_epochs = joint_epochs

        self.ee_flag=False
        if self.exits > 1:
            # early-exit network!
            self.ee_flag=True

        # set up trackers and best values in self
        self.best_val_loss = None
        self.best_val_accu = None
        self.train_loss_trk = None
        self.train_accu_trk = None
        self.valid_loss_trk = None
        self.valid_accu_trk = None

    # set up the trackers generic
    def _tracker_init(self, training_exits):
        # training exits bool
        if training_exits:
            e_num = self.exits
        else:
            e_num = 1
        self.best_val_loss = {"loss": [1.0]*e_num, "save_point":''}
        self.best_val_accu = {"accuracy": [0.0]*e_num, "save_point":''}
        self.train_loss_trk = LossTracker(train_dl.batch_size,bins=e_num)
        self.train_accu_trk = AccuTracker(train_dl.batch_size,bins=e_num)
        self.valid_loss_trk = LossTracker(valid_dl.batch_size,bins=e_num)
        self.valid_accu_trk = AccuTracker(valid_dl.batch_size,bins=e_num)

    def _train_ee(self, training_exits, epochs, opt, internal_folder=''):
        # training exits bool
        if training_exits:
            e_num = self.exits
            print(f"Training {e_num} exits")
        else:
            e_num = 1
            print(f"Training final exit")

        for epoch in range(epochs):
            self.model.train()
            print("starting epoch: {}".format(epoch+1), end="... ", flush=True)
            self.train_loss_trk.reset_tracker()
            self.train_accu_trk.reset_tracker()

            #training loop
            for xb, yb in train_dl:
                results = self.model(xb)

                if training_exits:
                    raw_losses = [self.loss_f(res,yb) for res in results]
                    losses = [weighting * raw_loss
                                for weighting, raw_loss in zip(model.exit_loss_weights,raw_losses)]
                    opt.zero_grad()
                    #backward
                    for loss in losses[:-1]: #ee losses need to keep graph
                        loss.backward(retain_graph=True)
                    losses[-1].backward() #final loss, graph not required
                    opt.step()

                    #raw losses
                    train_loss_trk.add_loss([exit_loss.item() for exit_loss in raw_losses])
                    train_accu_trk.update_correct(results,yb)
                else:
                    #TODO add backbone only method to bn class
                    loss = loss_f(results[-1], yb)
                    opt.zero_grad()
                    opt.step()

                    trainloss_trk.add_loss(loss.item())
                    trainaccu_trk.update_correct(results[-1],yb)

            # update training loss and accuracy averages
            tr_loss_avg = train_loss_trk.get_avg(return_list=True)
            t1acc = train_accu_trk.get_accu(return_list=True)
            if not training_exits:
                tr_loss_avg = tr_loss_avg[-1]
                t1acc = t1acc[-1]

            # validation loop - TODO needs to be different for multi and single
            model.eval()
            with torch.no_grad():
                for xb,yb in valid_dl:
                    res = model(xb)
                    valid_loss_trk.add_loss([loss_f(exit, yb) for exit in res])
                    valid_accu_trk.update_correct(res,yb)

            val_loss_avg = valid_loss_trk.get_avg(return_list=True)
            val_accu_avg = valid_accu_trk.get_accu(return_list=True)

            # debugging print - TODO log this
            print("raw t loss:{} t1acc:{}\nraw v loss:{} v accu:{}".format(
                tr_loss_avg,t1acc,val_loss_avg,val_accu_avg))
            if dat_norm:
                prefix = "dat_norm-"+prefix
            savepoint = save_model(model, spth, file_prefix=prefix+'-'+str(epoch+1), opt=opt,
                tloss=tr_loss_avg,vloss=val_loss_avg,taccu=t1acc,vaccu=val_accu_avg)

            # determining exit lost, current and best
            el_total=0.0
            bl_total=0.0
            for exit_loss, best_loss,l_w in zip(val_loss_avg,best_val_loss[0],model.exit_loss_weights):
                el_total+=exit_loss*l_w
                bl_total+=best_loss*l_w
            #selecting "best" network
            if el_total < bl_total:
                best_val_loss[0] = val_loss_avg
                best_val_loss[1] = savepoint

            # determining exit accuracy, current and best
            ea_total=0.0
            ba_total=0.0
            for exit_accu, best_accu,l_w in zip(val_accu_avg,best_val_accu[0],model.exit_loss_weights):
                ea_total+=exit_accu*l_w
                ba_total+=best_accu*l_w
            #selecting "best" network
            if ea_total > ba_total:
                best_val_accu[0] = val_accu_avg
                best_val_accu[1] = savepoint

        print("BEST* VAL LOSS: ", best_val_loss[0], " for epoch: ", best_val_loss[1])
        print("BEST* VAL ACCU: ", best_val_accu[0], " for epoch: ", best_val_accu[1])
        #return best val loss path link
        #return best_val_loss[1],savepoint
        return best_val_accu[1],savepoint

    def train_backbone(self, internal_folder=None):
        #set to branchynet default
        #Adam algo - step size alpha=0.001
        lr = 0.001
        #exponetial decay rates for 1st & 2nd moment: 0.99, 0.999
        exp_decay_rates = [0.99, 0.999]
        backbone_params = [
                {'params': model.backbone.parameters()},
                {'params': model.exits[-1].parameters()}
                ]

        opt = optim.Adam(backbone_params, betas=exp_decay_rates, lr=lr)

        # set up single exit tracker
        self._tracker_init(training_exits=False)

        # run the actual training loop
        self._train(epochs=self.backbone_epochs, opt=opt, internal_folder=internal_folder)

    def train_exits(self):
        # TODO specify exits to train - list indices?, bool list mask?
        #train the exits alone

        #Adam algo - step size alpha=0.001
        #exponetial decay rates for 1st & 2nd moment: 0.99, 0.999
        return #something trained

    def train_joint(self, pretrain_backbone=True)
        timestamp = dt.now().strftime("%Y-%m-%d_%H%M%S")

        if pretrain_backbone:
            print("PRETRAINING BACKBONE FROM SCRATCH")
            folder_path = 'pre_Trn_bb_' + timestamp

            best_bb_path,_ = self.train_backbone(internal_folder=folder_path)
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

        #set up the joint optimiser
        if opt is None: #TODO separate optim function to reduce code, maybe pass params?
            #set to branchynet default
            lr = 0.001 #Adam algo - step size alpha=0.001
            #exponetial decay rates for 1st & 2nd moment: 0.99, 0.999
            exp_decay_rates = [0.99, 0.999]

            # all parameters are being used ()all exits)
            opt = optim.Adam(model.parameters(), betas=exp_decay_rates, lr=lr)

        # set up trackers multi exit
        self._tracker_init(training_exits=True)

        # run the actual training loop
        self._train(epochs=self.joint_epochs, opt=opt, internal_folder=folder_path)

### OLD CODE - REFACTORING ###

# train network backbone
def train_backbone(model, train_dl, valid_dl, batch_size, save_path, epochs=50,
                    loss_f=nn.CrossEntropyLoss(), opt=None, dat_norm=False):

    if opt is None:
        #set to branchynet default
        #Adam algo - step size alpha=0.001
        lr = 0.001
        #exponetial decay rates for 1st & 2nd moment: 0.99, 0.999
        exp_decay_rates = [0.99, 0.999]
        backbone_params = [
                {'params': model.backbone.parameters()},
                {'params': model.exits[-1].parameters()}
                ]

        opt = optim.Adam(backbone_params, betas=exp_decay_rates, lr=lr)

    best_val_loss = [1.0, '']
    best_val_accu = [0.0, '']
    trainloss_trk = LossTracker(batch_size,1)
    trainaccu_trk = AccuTracker(batch_size,1)
    validloss_trk = LossTracker(batch_size,1)
    validaccu_trk = AccuTracker(batch_size,1)

    for epoch in range(epochs):
        model.train()
        print("Starting epoch:", epoch+1, end="... ", flush=True)
        correct_count=0

        trainloss_trk.reset_tracker()
        trainaccu_trk.reset_tracker()
        validloss_trk.reset_tracker()
        validaccu_trk.reset_tracker()

        #training loop
        for xb, yb in train_dl:
            results = model(xb)
            #loss for backbone ignores other exits
            #Wasting some forward compute of early exits
            #but shouldn't be included in backward step
            #since params not looked at by optimiser
            #TODO add backbone only method to bn class
            loss = loss_f(results[-1], yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            trainloss_trk.add_loss(loss.item())
            trainaccu_trk.update_correct(results[-1],yb)

        tr_loss_avg = trainloss_trk.get_avg(return_list=True)[-1]
        t1acc = trainaccu_trk.get_avg(return_list=True)[-1]

        #validation
        model.eval()
        with torch.no_grad():
            for xb,yb in valid_dl:
                res_v = model(xb)
                validloss_trk.add_loss(loss_f(res_v[-1], yb))
                validaccu_trk.update_correct(res_v[-1],yb)

        val_loss_avg = validloss_trk.get_avg(return_list=True)[-1]#should be last of 1
        val_accu_avg = validaccu_trk.get_avg(return_list=True)[-1]

        print(  "T Loss:",tr_loss_avg,
                "T T1 Acc: ", t1acc,
                "V Loss:", val_loss_avg,
                "V T1 Acc:", val_accu_avg)
        if dat_norm:
            file_prefix = "dat_norm-backbone-"
        else:
            file_prefix = "backbone-"
        savepoint = save_model(model, save_path, file_prefix=file_prefix+str(epoch+1), opt=opt,
                tloss=tr_loss_avg,vloss=val_loss_avg,taccu=t1acc,vaccu=val_accu_avg)

        if val_loss_avg < best_val_loss[0]:
            best_val_loss[0] = val_loss_avg
            best_val_loss[1] = savepoint
        if val_accu_avg > best_val_accu[0]:
            best_val_accu[0] = val_accu_avg
            best_val_accu[1] = savepoint
    print("BEST VAL LOSS: ", best_val_loss[0], " for epoch: ", best_val_loss[1])
    print("BEST VAL ACCU: ", best_val_accu[0], " for epoch: ", best_val_accu[1])
    #return best_val_loss[1], savepoint #link to best val loss model
    return best_val_accu[1], savepoint #link to best val accu model - trying for now

def train_joint(model, train_dl, valid_dl, batch_size, save_path, opt=None,
                loss_f=nn.CrossEntropyLoss(), backbone_epochs=50,
                joint_epochs=100, pretrain_backbone=True, dat_norm=False):

    timestamp = dt.now().strftime("%Y-%m-%d_%H%M%S")

    if pretrain_backbone:
        print("PRETRAINING BACKBONE FROM SCRATCH")
        folder_path = 'pre_Trn_bb_' + timestamp
        best_bb_path,_ = train_backbone(model, train_dl,
                valid_dl, batch_size, os.path.join(save_path, folder_path),
                epochs=backbone_epochs, loss_f=loss_f,dat_norm=dat_norm)
        #train the rest...
        print("LOADING BEST BACKBONE:",best_bb_path)
        load_model(model, best_bb_path)
        print("JOINT TRAINING WITH PRETRAINED BACKBONE")

        prefix = 'pretrn-joint'
    else:
        #jointly trains backbone and exits from scratch
        print("JOINT TRAINING FROM SCRATCH")
        folder_path = 'jnt_fr_scrcth' + timestamp
        prefix = 'joint'

    spth = os.path.join(save_path, folder_path)

    #set up the joint optimiser
    if opt is None: #TODO separate optim function to reduce code, maybe pass params?
        #set to branchynet default
        lr = 0.001 #Adam algo - step size alpha=0.001
        #exponetial decay rates for 1st & 2nd moment: 0.99, 0.999
        exp_decay_rates = [0.99, 0.999]

        opt = optim.Adam(model.parameters(), betas=exp_decay_rates, lr=lr)

    best_val_loss = [[1.0,1.0], ''] #TODO make sure list size matches num of exits
    best_val_accu = [[0.0,0.0], ''] #TODO make sure list size matches num of exits
    train_loss_trk = LossTracker(train_dl.batch_size,bins=2)
    train_accu_trk = AccuTracker(train_dl.batch_size,bins=2)
    valid_loss_trk = LossTracker(valid_dl.batch_size,bins=2)
    valid_accu_trk = AccuTracker(valid_dl.batch_size,bins=2)
    for epoch in range(joint_epochs):
        model.train()
        print("starting epoch:", epoch+1, end="... ", flush=True)
        train_loss = [0.0,0.0]
        correct_count = [0,0]
        train_loss_trk.reset_tracker()
        train_accu_trk.reset_tracker()
        #training loop
        for xb, yb in train_dl:
            results = model(xb)

            raw_losses = [loss_f(res,yb) for res in results]

            losses = [weighting * raw_loss
                        for weighting, raw_loss in zip(model.exit_loss_weights,raw_losses)]

            opt.zero_grad()
            #backward
            for loss in losses[:-1]: #ee losses need to keep graph
                loss.backward(retain_graph=True)
            losses[-1].backward() #final loss, graph not required
            opt.step()

            for i,_ in enumerate(train_loss):
                #weighted losses
                train_loss[i]+=losses[i].item()
            #raw losses
            train_loss_trk.add_loss([exit_loss.item() for exit_loss in raw_losses])
            train_accu_trk.update_correct(results,yb)


        tr_loss_avg_weighted = [loss/(len(train_dl)*batch_size) for loss in train_loss]
        tr_loss_avg = train_loss_trk.get_avg(return_list=True)
        t1acc = train_accu_trk.get_accu(return_list=True)

        #validation
        model.eval()
        with torch.no_grad():
            for xb,yb in valid_dl:
                res = model(xb)
                valid_loss_trk.add_loss([loss_f(exit, yb) for exit in res])
                valid_accu_trk.update_correct(res,yb)

        val_loss_avg = valid_loss_trk.get_avg(return_list=True)
        val_accu_avg = valid_accu_trk.get_accu(return_list=True)

        print("raw t loss:{} t1acc:{}\nraw v loss:{} v accu:{}".format(
            tr_loss_avg,t1acc,val_loss_avg,val_accu_avg))
        if dat_norm:
            prefix = "dat_norm-"+prefix
        savepoint = save_model(model, spth, file_prefix=prefix+'-'+str(epoch+1), opt=opt,
            tloss=tr_loss_avg,vloss=val_loss_avg,taccu=t1acc,vaccu=val_accu_avg)

        # determining exit lost, current and best
        el_total=0.0
        bl_total=0.0
        for exit_loss, best_loss,l_w in zip(val_loss_avg,best_val_loss[0],model.exit_loss_weights):
            el_total+=exit_loss*l_w
            bl_total+=best_loss*l_w
        #selecting "best" network
        if el_total < bl_total:
            best_val_loss[0] = val_loss_avg
            best_val_loss[1] = savepoint

        # determining exit accuracy, current and best
        ea_total=0.0
        ba_total=0.0
        for exit_accu, best_accu,l_w in zip(val_accu_avg,best_val_accu[0],model.exit_loss_weights):
            ea_total+=exit_accu*l_w
            ba_total+=best_accu*l_w
        #selecting "best" network
        if ea_total > ba_total:
            best_val_accu[0] = val_accu_avg
            best_val_accu[1] = savepoint

    print("BEST* VAL LOSS: ", best_val_loss[0], " for epoch: ", best_val_loss[1])
    print("BEST* VAL ACCU: ", best_val_accu[0], " for epoch: ", best_val_accu[1])
    #return best val loss path link
    #return best_val_loss[1],savepoint
    return best_val_accu[1],savepoint