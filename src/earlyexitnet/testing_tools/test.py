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
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms

# general imports
import os
import numpy as np
from datetime import datetime as dt
from typing import Callable
from time import perf_counter
from tqdm import tqdm
import matplotlib.pyplot as plt

class Tester:
    def __init__(
        self,
        model,
        test_dl,
        loss_f=nn.CrossEntropyLoss(),
        exits=2,
        top1acc_thresholds=[],
        entropy_thresholds=[],
        conf_funcs=None,
        device=None,
        save_raw=False,
    ):
        self.model = model
        self.test_dl = test_dl
        self.loss_f = loss_f
        self.exits = exits
        self.sample_total = len(test_dl)

        # TODO treat in the same way as the func selection
        #self.top1acc_thresholds = top1acc_thresholds
        #self.entropy_thresholds = entropy_thresholds
        self.thrs = [
                entropy_thresholds,
                top1acc_thresholds,
                top1acc_thresholds,
                top1acc_thresholds,
                ]
        #TODO
        #,top1acc_thresholds,top1acc_thresholds,top1acc_thresholds

        # list of AVAILABLE conf threshold methods
        self.conf_list = [
                self._thr_entropy,
                self._thr_max_softmax,
                self._thr_max_softmax_fast,
                self._thr_max_softmax_fast_noTrunc,
            ]

        # select functions chosen during cli
        if conf_funcs is None:
            # FIXME need default thresholds for other metrics
            # they function like the softmax on so should be similar
            self.conf_funcs = [(f, t) for f,t in zip(self.conf_list, self.thrs)]
        else:
            self.conf_funcs = [(self.conf_list[func], self.thrs[func]) \
                    for func in conf_funcs]

        self.save_raw = save_raw
        if self.save_raw:
            self.true_result = None
            self.raw_layer = None

        # set up the device
        if device is None or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = device

        # everybody needs a batch size
        self.batch_size = test_dl.batch_size
        self.tracker_dict = {}
        # set up conf bin and general stats
        self.stats_dict = {
            'conf_metrics' : {},
            'num_exits' : None,
            'num_samples' : None,
            'batch_size' : None,
            'accu_per_exit' : None
        }
        if exits > 1:
            # set up stat trackers
            for (func,thr) in self.conf_funcs:
                self.tracker_dict[str(func.__name__)] = {
                    'exit' : Tracker(self.batch_size, exits, self.sample_total),
                    'accu' : AccuTracker(self.batch_size, exits),
                }
                # set up return dict for all the confidence metric stats
                self.stats_dict['conf_metrics'][str(func.__name__)] = {
                    "exit_pc" :  None,
                    "accu_pc" : None,
                    "exit_threshs" : None,
                    "combined_accuracy" : None,
                }
                # add dict space for raw softmax
                if self.save_raw:
                    self.stats_dict['conf_metrics'][str(func.__name__)]["raw_softmax"] = None

            #        "Entropy",
            #        self._thr_entropy,
            #        "Softmax",
            #        self._thr_max_softmax,
            ##        "Trunc Base-2 Softmax",
            ##        self._fast_softmax_comparison,
            ##        "Non-Trunc Base-2 Softmax",
            ##        self._base2_softmax_comparison,
            ##        "Base-2 Sub-Softmax",
            ##        self._base2_sub_softmax_comparison,

        # total exit accuracy over the test data
        self.accu_track_totl = AccuTracker(test_dl.batch_size, exits, self.sample_total)

    def _thr_entropy(self, exit_results, thr):
        ### NOTE DEFINING entropy less than threshold
        sftmax = nn.functional.softmax(exit_results, dim=-1)
        entr = sftmax.log().mul(sftmax).nan_to_num().sum(dim=-1).mul(-1)
        exit_mask = entr.lt(thr)
        return exit_mask

    def _thr_max_softmax(self, exit_results, thr):
        ### NOTE DEFINING TOP1 of SOFTMAX DECISION
        sftmax = nn.functional.softmax(exit_results, dim=-1)
        # getting maximum values from softmax op
        sftmx_max = torch.max(sftmax, dim=-1).values
        # comparing to threshold to get boolean tensor mask for exit
        exit_mask = sftmx_max.gt(thr)
        return exit_mask

    def _thr_max_softmax_fast(self, exit_results, thr):
        # max softmax function but with base 2 usage
        # 2 to power of truncated logit values
        pow2 = torch.pow(2, torch.trunc(exit_results))
        # normalise exp wrt sum of exp
        pow2_max = pow2.div(pow2.sum(dim=-1).unsqueeze(1))
        # get max for each result in batch
        sftmx_max = torch.max(pow2_max, dim=-1).values
        # comparing to threshold to get boolean tensor mask for exit
        exit_mask = sftmx_max.gt(thr)
        return exit_mask

    def _thr_max_softmax_fast_noTrunc(self, exit_results, thr):
        # max softmax function but with base 2 usage
        # 2 to power of logit values
        pow2 = torch.pow(2, exit_results)
        # normalise exp wrt sum of exp
        pow2_max = pow2.div(pow2.sum(dim=-1).unsqueeze(1))
        # get max for each result in batch
        sftmx_max = torch.max(pow2_max, dim=-1).values
        # comparing to threshold to get boolean tensor mask for exit
        exit_mask = sftmx_max.gt(thr)
        return exit_mask

    def _thr_max_softmax_fast_noTrunc(self, exit_results, thr):
        # TODO implement sub version
        raise NotImplementedError("sub non trunc version not implemented.")

    def _thr_compare_(self, exit_track, accu_track,
            results, gnd_trth, thrs, thr_func):
        # generate all false mask
        prev_mask = torch.tensor([False] * self.batch_size,
                dtype=torch.bool, device=self.device
        )
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
                exit[exit_mask],
                gnd_trth[exit_mask],
                accum_count=exit_num,
                bin_index=i,
            )
            # update exit mask
            prev_mask = exit_mask

    #### lr versions of comparison stuff
    #def _entropy_comparison(
    #    self, layer: torch.Tensor, thresh: float, test=False
    #) -> bool:
    #    softmax = nn.functional.softmax(layer, dim=-1)
    #    entr = -torch.sum(torch.nan_to_num(softmax * torch.log(softmax)), dim=-1)

    #    if test:
    #        return entr < thresh, entr
    #    else:
    #        return entr < thresh

    #def _softmax_comparison(
    #    self, layer: torch.Tensor, thresh: float, test=False
    #) -> torch.Tensor:
    #    softmax = nn.functional.softmax(layer, dim=-1)
    #    # breakpoint()

    #    sftmx_max = torch.max(softmax, dim=-1).values

    #    if test:
    #        return sftmx_max > thresh, softmax
    #    else:
    #        return sftmx_max > thresh

    #def _fast_softmax_comparison(
    #    self, layer: torch.Tensor, thresh: float, test=False
    #) -> bool:
    #    softmax = hw_sim.base2_softmax_torch(layer)
    #    # softmax = hw_sim.subMax_softmax(exit)
    #    sftmx_max = torch.max(softmax, dim=-1).values

    #    if test:
    #        return sftmx_max > thresh, softmax
    #    else:
    #        return sftmx_max > thresh

    #def _base2_softmax_comparison(
    #    self, layer: torch.Tensor, thresh: float, test=False
    #) -> bool:
    #    softmax = hw_sim.nonTrunc_base2_softmax_torch(layer)
    #    # softmax = hw_sim.subMax_softmax(exit)
    #    sftmx_max = torch.max(softmax, dim=-1).values

    #    if test:
    #        return sftmx_max > thresh, softmax
    #    else:
    #        return sftmx_max > thresh

    def _base2_sub_softmax_comparison(
        self, layer: torch.Tensor, thresh: float, test=False
    ) -> bool:
        exp, sums = hw_sim.base2_subMax_softmax_fixed(layer)
        # softmax = hw_sim.subMax_softmax(exit)
        # sftmx_max = torch.max(softmax, dim=-1).values

        max_exp = np.max(exp, -1)

        threshes = (sums * thresh).flatten()

        # breakpoint()
        if test:
            return (
                torch.BoolTensor(max_exp > threshes, type=bool),
                torch.Tensor(exp.get_val()) / torch.Tensor(sums.get_val()),
            )
        else:
            return torch.BoolTensor(max_exp > threshes)

    def _test_multi_exit(self):
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            with tqdm(
                total=self.sample_total * self.batch_size,
                unit="samples",
            ) as pbar:
                for xb, yb in self.test_dl:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    res = self.model(xb)
                    # implicitly calls forward and returns array of arrays of the
                    # final layer for each exit (techically list of tensors for each exit)
                    # res has dimension [num_exits, batch_size, num_classes]

                    # accuracy of exits over everything
                    self.accu_track_totl.update_correct(res,yb)

                    for (conf,thrs) in self.conf_funcs:
                        self._thr_compare_(
                                self.tracker_dict[str(conf.__name__)]['exit'],
                                self.tracker_dict[str(conf.__name__)]['accu'],
                                res, yb,
                                thrs,
                                conf
                            )

                    # still not completely sure what this does
                    if self.save_raw:
                        self.true_result = (
                            torch.cat((self.true_result, yb))
                            if self.true_result is not None
                            else yb
                        )
                        self.raw_layer = (
                            torch.cat((self.raw_layer, res), dim=1)
                            if self.raw_layer is not None
                            else res
                        )

                    # Progress bar update based in batch size
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
        # for debugging the test set up
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

    def print_tracker_info(self, name_str):
        # prints info on a per conf basis so need to provide it with refs
        print("---", name_str, "---")
        exit_pc = self.tracker_dict[name_str]['exit'].get_avg(return_list=True)
        accu_pc = self.tracker_dict[name_str]['accu'].get_accu(return_list=True)
        print("Exit percentages:", exit_pc)
        print("Accuracy:", accu_pc)
        print(f"Total Accuracy: {np.dot(exit_pc, accu_pc):4f}")

        # TODO fix timing things - low priority because I know it's faster
        #print(
        #    "Total time:",
        #    self.total_time,
        #    "s",
        #    "Avg time:",
        #    self.total_time / self.sample_total,
        #    "s",
        #)

    def get_stats(self):
        for (func,thrs) in self.conf_funcs:
            name_str = str(func.__name__)
            ex_avg = self.tracker_dict[name_str]['exit'].get_avg(return_list=True)
            ac_avg = self.tracker_dict[name_str]['accu'].get_avg(return_list=True)

            self.stats_dict['conf_metrics'][name_str]["exit_pc"] = ex_avg
            self.stats_dict['conf_metrics'][name_str]["accu_pc"] = ac_avg
            self.stats_dict['conf_metrics'][name_str]["exit_threshs"] = thrs
            self.stats_dict['conf_metrics'][name_str]["combined_accuracy"] = np.dot(ex_avg, ac_avg)

            # TODO currently broken due to move from other class
            if self.save_raw:
                self.stats_dict['conf_metrics'][name_str]["raw_softmax"] = self.raw_softmax.tolist()

        self.stats_dict["num_exits"] = self.exits
        self.stats_dict["num_samples"] = self.sample_total * self.test_dl.batch_size
        self.stats_dict["batch_size"] = self.test_dl.batch_size
        self.stats_dict["accu_per_exit"] = self.accu_track_totl.get_accu(return_list=True)

        if self.save_raw:
            self.stats_dict["true_indices"] = self.true_result.tolist()
            # grab from any of the comparators that were used
            self.stats_dict["raw_layer"] = self.raw_layer.tolist()

        return self.stats_dict

    def test(self):
        print(f"Test of length {self.sample_total} starting")
        if self.exits > 1:
            self._test_multi_exit()
            print("### TEST FINISHED ###")
            for (func,_) in self.conf_funcs:
                self.print_tracker_info(str(func.__name__))
            print("########")
        else:
            self._test_single_exit()
            print("### TEST FINISHED ###")

        # accuracy of each exit over FULL data set
        print("Total Accuracy:", self.accu_track_totl.get_accu(return_list=True))
