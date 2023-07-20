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
from time import perf_counter
from tqdm import tqdm
import matplotlib.pyplot as plt

class Comparison:
    def __init__(
            self,
            name: str, 
            compare_func: Callable[[torch.Tensor, float, bool],torch.Tensor], 
            exit_track: Tracker, 
            accu_track: AccuTracker,
            exit_thresholds: list[float],
            test = False
        ):
        self.name = name
        self.compare_func = compare_func
        self.exit_track = exit_track
        self.accu_track = accu_track
        self.exit_thresholds = exit_thresholds
        
        self.total_time = 0
        
        self.test = test
        if self.test:
            self.raw_softmax = None
            self.true_result = None


    def eval(self, batched_results: torch.Tensor, batched_correct_results: torch.Tensor):
        # FAILED BECAUSE CONDITIONAL OPERATIONS ARE NOT SUPPORTED
        # batched results is batched along the second dimension [num_exits, batch_size, num_classes]
        # batched_compare = functorch.vmap(self.compare_func, in_dims=1) # [E, B, C] -> [B, 2] instead of [E,C] -> [2]
        
        # exit_location = batched_compare(batched_results)
        
        # call self.compare to obtained tensor of size [B,2] with the exit pos and layer for each batch
        
                # loop over all exits 
        
        # batched_compare_func = functorch.vmap(self.compare_func, in_dims=(0,None), out_dims=(0, None)) # ([B,C], float) -> [B, 1] instead of ([C], float) -> [1]


        num_batches = batched_results.size(dim=1)
        num_exits = batched_results.size(dim=0)
        num_classes = batched_results.size(dim=2)
        
        hasExited = torch.zeros([num_batches], dtype=torch.bool)

        exit_counter = [0] * num_exits

        start = perf_counter()
        
        if self.test:
            sft_accum = None
        
        for exit_b in range(num_exits): 
            index = torch.Tensor([exit_b]).to(batched_results.device).type(torch.int32)
            result_layer = torch.index_select(batched_results, 0, index) # select exit from [E, B, C] to get [1, B, C]

            result_layer = result_layer.reshape(num_batches,num_classes)# reshape from [B,1,C] to [B,C]
            
            
            if self.test:
                exit_result, raw_softmax = self.compare_func(result_layer, self.exit_thresholds[exit_b], test=True)
                
                # stack to separate exit dimension
                sft_accum = torch.stack((sft_accum, raw_softmax),dim=0) if sft_accum is not None else raw_softmax
                
            else:
                exit_result = self.compare_func(result_layer, self.exit_thresholds[exit_b])
                
            # do this better by indexing into the arrays with the exit_result booleans 
            # result_layer[exti_result] return a ternsor with only the layers which would've exited
            for batch, isExit in enumerate(exit_result):
                if isExit and not hasExited[batch]:
                    self.exit_track.add_val(1, exit_b)
                    self.accu_track.update_correct(result_layer[batch], batched_correct_results[batch], bin_index=exit_b)
                    hasExited[batch] = 1
                    exit_counter[exit_b] += 1
        stop = perf_counter()
        
        if self.test:
            # concatenate along the batch dimension
            self.raw_softmax = torch.cat((self.raw_softmax, sft_accum), dim=1) if self.raw_softmax is not None else sft_accum
            
            self.true_result = torch.cat((self.true_result, batched_correct_results)) if self.true_result is not None else batched_correct_results
        
        self.total_time += stop - start
    
    def print_tracker_info(self, num_samples):
        print("---", self.name, "---")
        exit_perc = self.exit_track.get_avg(return_list=True)
        accu_perc = self.accu_track.get_accu(return_list=True)
        print("Exit percentages:", exit_perc)
        print("Accuracy:", accu_perc)
        print(f"Total Accuracy: {np.dot(exit_perc, accu_perc):4f}")
        print("Total time:", self.total_time, "s", "Avg time:", self.total_time/num_samples, "s")

        
    def get_comp_info(self):
        return_val = {}
        
        return_val['name'] = self.name
        return_val['exit_percs'] = self.exit_track.get_avg(return_list=True)
        return_val['accu_percs'] = self.accu_track.get_avg(return_list=True)
        return_val['exit_threshs'] = self.exit_thresholds
        return_val['combined_accuracy'] = np.dot(self.exit_track.get_avg(return_list=True), self.accu_track.get_avg(return_list=True))
        
        if self.test:
            return_val['raw_softmax'] = self.raw_softmax.tolist()
            
        return return_val

class Tester:
    def __init__(self,model,test_dl,loss_f=nn.CrossEntropyLoss(),exits=2,
            top1acc_thresholds=[],entropy_thresholds=[],comp_funcs=None,device=None, save_raw = False):
        self.model=model
        self.test_dl=test_dl
        self.loss_f=loss_f
        self.exits=exits
        self.sample_total = len(test_dl)
        self.top1acc_thresholds = top1acc_thresholds
        self.entropy_thresholds = entropy_thresholds
        
        self.comp_funcs = comp_funcs
        
        self.save_raw = save_raw
        
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

            self.comparators = [
                Comparison(
                    "Entropy",
                    self._entropy_comparison,
                    Tracker(test_dl.batch_size,exits,self.sample_total),
                    AccuTracker(1,exits),
                    self.entropy_thresholds,
                    self.save_raw
                ),            
                Comparison(
                    "Softmax",
                    self._softmax_comparison,
                    Tracker(test_dl.batch_size,exits,self.sample_total),
                    AccuTracker(1,exits),
                    self.top1acc_thresholds,
                    self.save_raw
                ),            
                Comparison(
                    "Trunc Base-2 Softmax",
                    self._fast_softmax_comparison,
                    Tracker(test_dl.batch_size,exits,self.sample_total),
                    AccuTracker(1,exits),
                    self.top1acc_thresholds,
                    self.save_raw
                ),
                Comparison(
                    "Non-Trunc Base-2 Softmax",
                    self._base2_softmax_comparison,
                    Tracker(test_dl.batch_size,exits,self.sample_total),
                    AccuTracker(1,exits),
                    self.top1acc_thresholds,
                    self.save_raw
                ),
                Comparison(
                    "Base-2 Sub-Softmax",
                    self._base2_sub_softmax_comparison,
                    Tracker(test_dl.batch_size,exits,self.sample_total),
                    AccuTracker(1,exits),
                    self.top1acc_thresholds,
                    self.save_raw
                ),
                ]

        #total exit accuracy over the test data
        self.accu_track_totl = AccuTracker(test_dl.batch_size,exits,self.sample_total)


    def _entropy_comparison(self, layer: torch.Tensor, thresh: float) -> bool:
        softmax = nn.functional.softmax(layer,dim=-1)
        entr = -torch.sum(torch.nan_to_num(softmax * torch.log(softmax)), dim=-1)
           
        return entr < thresh
        
    def _softmax_comparison(self, layer: torch.Tensor, thresh: float, test=False) -> torch.Tensor:
        softmax = nn.functional.softmax(layer,dim=-1)
        # breakpoint()
        
        sftmx_max = torch.max(softmax, dim=-1).values
        
        if test:
            return sftmx_max > thresh, softmax
        else:
            return sftmx_max > thresh

    def _fast_softmax_comparison(self, layer: torch.Tensor, thresh: float, test=False) -> bool:
        
        softmax = hw_sim.base2_softmax_torch(layer)
        # softmax = hw_sim.subMax_softmax(exit)            
        sftmx_max = torch.max(softmax, dim=-1).values           
        
        if test:
            return sftmx_max > thresh, softmax
        else:
            return sftmx_max > thresh
    
    def _base2_softmax_comparison(self, layer: torch.Tensor, thresh: float, test=False) -> bool:
        
        softmax = hw_sim.nonTrunc_base2_softmax_torch(layer)
        # softmax = hw_sim.subMax_softmax(exit)            
        sftmx_max = torch.max(softmax, dim=-1).values           
        
        if test:
            return sftmx_max > thresh, softmax
        else:
            return sftmx_max > thresh
    
    def _base2_sub_softmax_comparison(self, layer: torch.Tensor, thresh: float, test=False) -> bool:
        
        exp, sums = hw_sim.base2_subMax_softmax_fixed(layer)
        # softmax = hw_sim.subMax_softmax(exit)            
        # sftmx_max = torch.max(softmax, dim=-1).values       
        
        max_exp = np.max(exp,-1)
        
        threshes = (sums * thresh).flatten()
        
        # breakpoint()
        if test:
            return torch.Tensor(max_exp > threshes), torch.Tensor(exp.get_val())/torch.Tensor(sums.get_val()),  
        else:
            return torch.Tensor(max_exp > threshes)
        
    def _test_multi_exit(self):
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            with tqdm(total=self.sample_total*self.test_dl.batch_size, unit="samples") as pbar:
                for xb,yb in self.test_dl:
                    xb,yb = xb.to(self.device),yb.to(self.device)
                    res = self.model(xb) # implicitly calls forward and returns array of arrays of the final layer for each exit (techically list of tensors for each exit)
                    # res has dimension [num_exits, batch_size, num_classes]
                    # breakpoint()
                    self.accu_track_totl.update_correct_list(res,yb)
                    
                    if self.comp_funcs is not None:
                        for comp in self.comp_funcs:
                            self.comparators[comp].eval(res,yb)
                    else:
                        for comp in self.comparators:
                            comp.eval(res, yb)
                    
                    pbar.update(self.test_dl.batch_size)

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
            for xb,yb in self.test_dl:
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

    def get_stats(self):
        
        return_val = {}
        
        return_val['comps'] = []
        
        if self.comp_funcs is not None:
            for comp in self.comp_funcs:
                return_val['comps'].append(self.comparators[comp].get_comp_info())
        else:
            for comp in self.comparators:
                return_val['comps'].append(comp.get_comp_info())

        return_val['num_exits'] = self.exits
        return_val['num_samples'] = self.sample_total * self.test_dl.batch_size
        return_val['batch_size'] = self.test_dl.batch_size
        return_val['accu_per_exit'] = self.accu_track_totl.get_accu(return_list=True)
        
        if self.save_raw:
            return_val['true_indices'] = self.comparators[self.comp_funcs[0]].true_result.tolist() # grab from any of the comparators that were used
        
        return return_val

    def test(self):
        print(f"Test of length {self.sample_total} starting")
        if self.exits > 1:
            self._test_multi_exit()
            print("### TEST FINISHED ###")
                        
                        
            if self.comp_funcs is not None:
                for comp in self.comp_funcs:
                    self.comparators[comp].print_tracker_info(self.sample_total)
            else:
                for comp in self.comparators:
                    comp.print_tracker_info(self.sample_total)    
            print("########")
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
            print("### TEST FINISHED ###")
            
        #accuracy of each exit over FULL data set
        print("Total Accuracy:", self.accu_track_totl.get_accu(return_list=True))
        #TODO save test stats along with link to saved model
