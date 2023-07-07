"""
Class for testing early exit and normal CNNs.
Includes requires accuracy and loss trackers from tools
"""

# import custom funcions to simulate hardware
import earlyexitnet.testing_tools.hw_sim as hw_sim

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

        if exits > 1:
            #TODO make thresholds a more flexible param
            #setting top1acc threshold for exiting (final exit set to 0)
            #self.top1acc_thresholds = [0.995,0]
            #setting entropy threshold for exiting (final exit set to LARGE)
            #self.entropy_thresholds = [0.025,1000000]
            #set up stat trackers
            #samples exited
            self.exit_track_top1 = Tracker(test_dl.batch_size,exits,self.sample_total)
            self.exit_track_entr = Tracker(test_dl.batch_size,exits,self.sample_total)
            self.exit_track_fast = Tracker(test_dl.batch_size,exits,self.sample_total)
            #individual accuracy over samples exited
            self.accu_track_top1 = AccuTracker(test_dl.batch_size,exits)
            self.accu_track_entr = AccuTracker(test_dl.batch_size,exits)
            self.accu_track_fast = AccuTracker(test_dl.batch_size,exits)

        #total exit accuracy over the test data
        self.accu_track_totl = AccuTracker(
            test_dl.batch_size,exits,self.sample_total)

        self.top1_pc = None # % exit for top1 confidence
        self.entr_pc = None # % exit for entropy confidence
        self.fast_pc = None
        self.top1_accu = None #accuracy of exit over exited samples
        self.entr_accu = None #accuracy of exit over exited samples
        self.fast_accu = None
        self.full_exit_accu = None #accuracy of the exits over all samples
        self.top1_accu_tot = None #total accuracy of network given exit strat
        self.entr_accu_tot = None #total accuracy of network given exit strat
        self.fast_accu_tot = None


    def _entropy_comparison(self, results : list[list[float]], correct_results : list[float]):
        for i,(exit,thr) in enumerate(zip(results,self.entropy_thresholds)):
            softmax = nn.functional.softmax(exit,dim=-1)
            entr = -torch.sum(torch.nan_to_num(softmax * torch.log(softmax)))
            if entr < thr:
                #print("entr exited at exit {}".format(i))
                self.exit_track_entr.add_val(1,i)
                self.accu_track_entr.update_correct(exit,correct_results,bin_index=i)
                break
            
    def _softmax_comparison(self, results : list[list[float]], correct_results : list[float]):
        for i,(exit,thr) in enumerate(zip(results,self.top1acc_thresholds)):
        ### NOTE DEFINING TOP1 of SOFTMAX DECISION
            softmax = nn.functional.softmax(exit,dim=-1)
            sftmx_max = torch.max(softmax)
            if sftmx_max > thr:
                #print("top1 exited at exit {}".format(i))
                self.exit_track_top1.add_val(1,i)
                self.accu_track_top1.update_correct(exit,correct_results,bin_index=i)
                break

    def _fast_softmax_comparison(self, results : list[list[float]], correct_results : list[float]):
        for i,(exit,thr) in enumerate(zip(results,self.top1acc_thresholds)):
        
            softmax = hw_sim.fast_softmax(exit)
            sftmx_max = max(softmax)           
            
            if sftmx_max > thr:
                #print("top1 exited at exit {}".format(i))
                self.exit_track_fast.add_val(1,i)
                self.accu_track_fast.update_correct(exit,correct_results,bin_index=i)
                break

    def _test_multi_exit(self):
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            for xb,yb in self.test_dl:
                xb,yb = xb.to(self.device),yb.to(self.device)
                res = self.model(xb) # implicitly calls forward and returns array of arrays of the final layer for each exit (techically list of tensors for each exit)
                self.accu_track_totl.update_correct(res,yb) # DOUBT should it not be update_correct_list?
                
                self._softmax_comparison(res,yb)
                self._entropy_comparison(res,yb)
                self._fast_softmax_comparison(res,yb)

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
            self.fast_pc = self.exit_track_fast.get_avg(return_list=True)
            self.top1_accu = self.accu_track_top1.get_accu(return_list=True)
            self.entr_accu = self.accu_track_entr.get_accu(return_list=True)
            self.entr_fast = self.accu_track_fast.get_accu(return_list=True)
            self.top1_accu_tot = np.sum(self.accu_track_top1.val_bins)/self.sample_total
            self.entr_accu_tot = np.sum(self.accu_track_entr.val_bins)/self.sample_total
            self.entr_accu_fast = np.sum(self.accu_track_fast.val_bins)/self.sample_total
        else:
            self._test_single_exit()
        #accuracy of each exit over FULL data set
        self.full_exit_accu = self.accu_track_totl.get_accu(return_list=True)
        #TODO save test stats along with link to saved model
