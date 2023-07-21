"""
Class for testing early exit and normal CNNs.
Includes requires accuracy and loss trackers from tools
"""

# importing trackers for loss + accuracy, and generic tracker for exit distribution
from earlyexitnet.tools import Tracker, LossTracker, AccuTracker

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

class Tester:
    def __init__(self,model,test_dl,loss_f=nn.CrossEntropyLoss(),exits=2,
            top1acc_thresholds=[],entropy_thresholds=[],device=None):
        self.model=model
        self.test_dl=test_dl
        self.loss_f=loss_f
        self.exits=exits
        self.sample_total = len(test_dl)
        self.top1acc_thresholds = top1acc_thresholds
        self.entropy_thresholds = entropy_thresholds
        if device is None or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = device

        self.batch_size = test_dl.batch_size
        if exits > 1:
            #TODO make thresholds a more flexible param
            #setting top1acc threshold for exiting (final exit set to 0)
            #self.top1acc_thresholds = [0.995,0]
            #setting entropy threshold for exiting (final exit set to LARGE)
            #self.entropy_thresholds = [0.025,1000000]
            #set up stat trackers
            #samples exited
            self.exit_track_top1 = Tracker(self.batch_size,exits,self.sample_total)
            self.exit_track_entr = Tracker(self.batch_size,exits,self.sample_total)
            #individual accuracy over samples exited
            self.accu_track_top1 = AccuTracker(self.batch_size,exits)
            self.accu_track_entr = AccuTracker(self.batch_size,exits)

        #total exit accuracy over the test data
        self.accu_track_totl = AccuTracker(
            self.batch_size,exits,self.sample_total)

        self.top1_pc = None # % exit for top1 confidence
        self.entr_pc = None # % exit for entropy confidence
        self.top1_accu = None #accuracy of exit over exited samples
        self.entr_accu = None #accuracy of exit over exited samples
        self.full_exit_accu = None #accuracy of the exits over all samples
        self.top1_accu_tot = None #total accuracy of network given exit strat
        self.entr_accu_tot = None #total accuracy of network given exit strat

    def _thr_max_softmax(self,exit_results, thr):
        ### NOTE DEFINING TOP1 of SOFTMAX DECISION
        sftmax = nn.functional.softmax(exit_results,dim=-1)
        # getting maximum values from softmax op
        sftmx_max = torch.max(sftmax, dim=-1).values
        # comparing to threshold to get boolean tensor mask for exit
        exit_mask = sftmx_max.gt(thr)
        return exit_mask

    def _thr_entropy(self,exit_results, thr):
        ### NOTE DEFINING entropy less than threshold
        sftmax = nn.functional.softmax(exit_results,dim=-1)
        entr = sftmax.log().mul(sftmax).nan_to_num().sum(dim=-1).mul(-1)
        exit_mask = entr.lt(thr)
        return exit_mask

    def _thr_compare_(self, exit_track, accu_track,
            results, gnd_trth, thrs, thr_func):
        # generate all false mask
        prev_mask = torch.tensor([False]*self.batch_size,
                dtype=torch.bool,device=self.device)
        for i,(exit,thr) in enumerate(zip(results,thrs)):
            # call function to generate mask
            exit_mask = thr_func(exit,thr)
            # mask out values that previously exited
            exit_mask = exit_mask.logical_and(prev_mask.logical_not())
            # get number that are exiting here
            exit_num = exit_mask.sum()
            # updated the number exiting
            exit_track.add_val(exit_num,bin_index=i)
            # update accuracy, along with number exiting here
            accu_track.update_correct(
                    exit[exit_mask],gnd_trth[exit_mask],
                    accum_count=exit_num,bin_index=i)
            # update exit mask
            prev_mask = exit_mask

    def _test_multi_exit(self):
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            for xb,yb in self.test_dl:
                xb,yb = xb.to(self.device),yb.to(self.device)
                res = self.model(xb)
                # accuracy of exits over everything
                self.accu_track_totl.update_correct(res,yb)
                # maximum value of softmax (top1) GREATER than thr
                self._thr_compare_(self.exit_track_top1,
                        self.accu_track_top1,
                        res, yb, self.top1acc_thresholds,
                        self._thr_max_softmax)
                # entropy of softmax is LOWER than threshold
                self._thr_compare_(self.exit_track_entr,
                        self.accu_track_entr,
                        res, yb, self.entropy_thresholds,
                        self._thr_entropy)

    def _test_single_exit(self):
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            for xb,yb in self.test_dl:
                xb,yb = xb.to(self.device),yb.to(self.device)
                res = self.model(xb)
                self.accu_track_totl.update_correct(res,yb)

    def debug_values(self):
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            for xb,yb in test_dl:
                xb,yb = xb.to(self.device),yb.to(self.device)
                res = self.model(xb)
                for i,exit in enumerate(res):
                    #print("raw exit {}: {}".format(i, exit))
                    softmax = nn.functional.softmax(exit,dim=-1)
                    #print("softmax exit {}: {}".format(i, softmax))
                    sftmx_max = torch.max(softmax)
                    print("exit {} max softmax: {}".format(i, sftmx_max))
                    entr = -torch.sum(torch.nan_to_num(softmax * torch.log(softmax)))
                    print("exit {} entropy: {}".format(i, entr))
                    #print("exit CE loss: {}".format(loss_f(exit,yb)))

    def test(self):
        print(f"Test of length {self.sample_total} starting")
        if self.exits > 1:
            self._test_multi_exit()
            self.top1_pc = self.exit_track_top1.get_avg(return_list=True)
            self.entr_pc = self.exit_track_entr.get_avg(return_list=True)
            self.top1_accu = self.accu_track_top1.get_accu(return_list=True)
            self.entr_accu = self.accu_track_entr.get_accu(return_list=True)
            self.top1_accu_tot = np.sum(self.accu_track_top1.val_bins)/(self.sample_total*self.batch_size)
            self.entr_accu_tot = np.sum(self.accu_track_entr.val_bins)/(self.sample_total*self.batch_size)
        else:
            self._test_single_exit()
        #accuracy of each exit over FULL data set
        self.full_exit_accu = self.accu_track_totl.get_accu(return_list=True)
        #TODO save test stats along with link to saved model
