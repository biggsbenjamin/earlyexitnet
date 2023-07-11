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
import functorch # technically deprecated

# general imports
import os
import numpy as np
from datetime import datetime as dt
from typing import Callable

class Comparison:
    def __init__(
            self,
            name: str, 
            compare_func: Callable[[torch.Tensor],torch.Tensor], 
            exit_track: Tracker, 
            accu_track: AccuTracker,
            exit_thresholds: list[float]
        ):
        self.name = name
        self.compare_func = compare_func
        self.exit_track = exit_track
        self.accu_track = accu_track
        self.exit_thresholds = exit_thresholds
    
    def compare(self, results: torch.Tensor) -> tuple[int, torch.Tensor]:
        for exit_pos, (exit_layer,thr) in enumerate(zip(results,self.exit_thresholds)):
            if self.compare_func(exit_layer, thr):
                return (exit_pos, exit_layer)
    
    def eval(self, batched_results: torch.Tensor, batched_correct_results: torch.Tensor):
        # FAILED BECAUSE CONDITIONAL OPERATIONS ARE NOT SUPPORTED
        # batched results is batched along the second dimension [num_exits, batch_size, num_classes]
        # batched_compare = functorch.vmap(self.compare_func, in_dims=1) # [E, B, C] -> [B, 2] instead of [E,C] -> [2]
        
        # exit_location = batched_compare(batched_results)
        
        # iterate over all batches
        batched_results = batched_results.to(torch.device('cpu'))
        for i in range(batched_results.size(dim=1)):
            index = torch.Tensor([i]).to(batched_results.device).type(torch.int32)
            # breakpoint()
            result_layer = torch.index_select(batched_results, 1, index) # get specific batch in format [E,C]
            exit_pos, result = self.compare(result_layer)
            
            self.exit_track.add_val(1, exit_pos)
            # self.accu_track.update_correct(result, batched_correct_results[i], bin_index=exit_pos)
    
    def print_tracker_info(self):
        print(self.name)
        print(self.exit_track.get_avg(return_list=True))
        # print(self.accu_track.get_accu(return_list=True))
        
        
            # self.top1_pc = self.exit_track_top1.get_avg(return_list=True)
            # self.entr_pc = self.exit_track_entr.get_avg(return_list=True)
            # # self.fast_pc = self.exit_track_fast.get_avg(return_list=True)
            # self.top1_accu = self.accu_track_top1.get_accu(return_list=True)
            # self.entr_accu = self.accu_track_entr.get_accu(return_list=True)
            # # self.fast_accu = self.accu_track_fast.get_accu(return_list=True)
            # self.top1_accu_tot = np.sum(self.accu_track_top1.val_bins / self.test_dl.batch_size)/self.sample_total

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
            # self.exit_track_top1 = Tracker(test_dl.batch_size,exits,self.sample_total)
            # self.exit_track_entr = Tracker(test_dl.batch_size,exits,self.sample_total)
            # self.exit_track_fast = Tracker(test_dl.batch_size,exits,self.sample_total)
            # #individual accuracy over samples exited
            # self.accu_track_top1 = AccuTracker(1,exits)
            # self.accu_track_entr = AccuTracker(1,exits)
            # self.accu_track_fast = AccuTracker(1,exits)

            self.entropy_cmp = Comparison(
                "Entropy",
                self._entropy_comparison,
                Tracker(test_dl.batch_size,exits,self.sample_total),
                AccuTracker(1,exits),
                self.entropy_thresholds
            )
            
            self.softmax_cmp = Comparison(
                "Softmax",
                self._softmax_comparison,
                Tracker(test_dl.batch_size,exits,self.sample_total),
                AccuTracker(1,exits),
                self.top1acc_thresholds
            )
            
            self.fast_softmax_cmp = Comparison(
                "Quick Softmax",
                self._fast_softmax_comparison,
                Tracker(test_dl.batch_size,exits,self.sample_total),
                AccuTracker(1,exits),
                self.top1acc_thresholds
            )

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
            
    def _softmax_comparison(self, layer: torch.Tensor, thresh: float) -> bool:
        # breakpoint()
        ### NOTE DEFIING TOP1 of SOFTMAX DECISION
        softmax = nn.functional.softmax(layer,dim=-1)
        
        sftmx_max = torch.max(softmax)
        
        return sftmx_max > thresh

    def _fast_softmax_comparison(self, results : list[list[float]], correct_results : list[float]):
        
        for i,(exit,thr) in enumerate(zip(results,self.top1acc_thresholds)):
        
            softmax = hw_sim.base2_softmax(exit)
            # softmax = hw_sim.subMax_softmax(exit)            
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
                # res has dimension [num_exits, batch_size, num_classes]
                
                # self.accu_track_totl.update_correct(res,yb)
            
                self.softmax_cmp.eval(res, yb)
                

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
            
            self.softmax_cmp.print_tracker_info()
            
            # self.top1_pc = self.exit_track_top1.get_avg(return_list=True)
            # self.entr_pc = self.exit_track_entr.get_avg(return_list=True)
            # # self.fast_pc = self.exit_track_fast.get_avg(return_list=True)
            # self.top1_accu = self.accu_track_top1.get_accu(return_list=True)
            # self.entr_accu = self.accu_track_entr.get_accu(return_list=True)
            # # self.fast_accu = self.accu_track_fast.get_accu(return_list=True)
            # self.top1_accu_tot = np.sum(self.accu_track_top1.val_bins / self.test_dl.batch_size)/self.sample_total
            # self.entr_accu_tot = np.sum(self.accu_track_entr.val_bins)/self.sample_total
            # self.fast_accu_tot = np.sum(self.accu_track_fast.val_bins)/self.sample_total
        else:
            self._test_single_exit()
        #accuracy of each exit over FULL data set
        self.full_exit_accu = self.accu_track_totl.get_accu(return_list=True)
        #TODO save test stats along with link to saved model
