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


class Comparison:
    def __init__(
        self,
        name: str,
        compare_func: Callable[[torch.Tensor, float, bool], torch.Tensor],
        exit_track: Tracker,
        accu_track: AccuTracker,
        exit_thresholds: list[float],
        test=False,
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

    def eval(
        self,
        batched_results: torch.Tensor,
        batched_correct_results: torch.Tensor,
    ):
        # extract various information about the size of the incoming vector
        num_batches = batched_results.size(dim=1)
        num_exits = batched_results.size(dim=0)
        num_classes = batched_results.size(dim=2)

        # tensor to keep track of which values in the batch have exited already
        hasExited = torch.zeros([num_batches], dtype=torch.bool)

        start = perf_counter()

        if self.test:
            sft_accum = None

        for exit_b in range(num_exits):
            result_layer = batched_results[exit_b]

            # if saving the vector values keep track of them
            if self.test:
                exit_result, raw_softmax = self.compare_func(
                    result_layer, self.exit_thresholds[exit_b], test=True
                )

                # stack to separate exit dimension
                sft_accum = (
                    torch.stack((sft_accum, raw_softmax), dim=0)
                    if sft_accum is not None
                    else raw_softmax
                )

                # compute mask of values which have exited now and haven't exited before
                mask = torch.logical_and(exit_result, torch.logical_not(hasExited))
                # compute how many values exited
                exit_size = mask.sum().item()
                exited_vec = result_layer[mask]
                truth_vec = batched_correct_results[mask]
                # update which values have exited
                hasExited = torch.logical_or(hasExited, mask)
            else:
                # don't care about results of the computations for those values which have exited already
                # perform operation only on values not yet exited
                not_exit_mask = torch.logical_not(hasExited)
                exit_result = self.compare_func(
                    result_layer[not_exit_mask],
                    self.exit_thresholds[exit_b],
                )
                exit_size = exit_result.sum().item()
                # tensor containing the values that have exited
                # first index those values which hadn't already exited
                # then index those values which now have exited (the second mask doesn't make sense without the first one)
                exited_vec = (result_layer[not_exit_mask])[exit_result]
                truth_vec = (batched_correct_results[not_exit_mask])[exit_result]
                # update which values have exited
                # use the same mask as before to update only those values for which the computation was performed
                hasExited[not_exit_mask] = exit_result

            # compute how many values are correctly identified
            correct = (exited_vec.argmax(dim=-1) == truth_vec).sum().item()
            self.exit_track.add_val(exit_size, bin_index=exit_b)
            self.accu_track.add_val(correct, accum_count=exit_size, bin_index=exit_b)

        stop = perf_counter()

        if self.test:
            # concatenate along the batch dimension
            self.raw_softmax = (
                torch.cat((self.raw_softmax, sft_accum), dim=1)
                if self.raw_softmax is not None
                else sft_accum
            )

        self.total_time += stop - start

    def print_tracker_info(self, num_samples):
        print("---", self.name, "---")
        exit_perc = self.exit_track.get_avg(return_list=True)
        accu_perc = self.accu_track.get_accu(return_list=True)
        print("Exit percentages:", exit_perc)
        print("Accuracy:", accu_perc)
        print(f"Total Accuracy: {np.dot(exit_perc, accu_perc):4f}")
        print(
            "Total time:",
            self.total_time,
            "s",
            "Avg time:",
            self.total_time / num_samples,
            "s",
        )

    def get_comp_info(self):
        return_val = {}

        return_val["name"] = self.name
        return_val["exit_percs"] = self.exit_track.get_avg(return_list=True)
        return_val["accu_percs"] = self.accu_track.get_avg(return_list=True)
        return_val["exit_threshs"] = self.exit_thresholds
        return_val["combined_accuracy"] = np.dot(
            self.exit_track.get_avg(return_list=True),
            self.accu_track.get_avg(return_list=True),
        )

        if self.test:
            return_val["raw_softmax"] = self.raw_softmax.tolist()

        return return_val


class Tester:
    def __init__(
        self,
        model,
        test_dl,
        loss_f=nn.CrossEntropyLoss(),
        exits=2,
        top1acc_thresholds=[],
        entropy_thresholds=[],
        comp_funcs=None,
        device=None,
        save_raw=False,
    ):
        self.model = model
        self.test_dl = test_dl
        self.loss_f = loss_f
        self.exits = exits
        self.sample_total = len(test_dl)
        self.top1acc_thresholds = top1acc_thresholds
        self.entropy_thresholds = entropy_thresholds

        self.comp_funcs = comp_funcs

        self.save_raw = save_raw

        if self.save_raw:
            self.true_result = None
            self.raw_layer = None

        if device is None or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = device

        self.batch_size = test_dl.batch_size
        if exits > 1:
            # set up stat trackers
            # samples exited
            self.exit_track_top1 = Tracker(self.batch_size, exits, self.sample_total)
            self.exit_track_entr = Tracker(self.batch_size, exits, self.sample_total)
            # individual accuracy over samples exited
            self.accu_track_top1 = AccuTracker(self.batch_size, exits)
            self.accu_track_entr = AccuTracker(self.batch_size, exits)

            self.comparators = [
                Comparison(
                    "Entropy",
                    self._entropy_comparison,
                    Tracker(test_dl.batch_size, exits, self.sample_total),
                    AccuTracker(1, exits),
                    self.entropy_thresholds,
                    self.save_raw,
                ),
                Comparison(
                    "Softmax",
                    self._softmax_comparison,
                    Tracker(test_dl.batch_size, exits, self.sample_total),
                    AccuTracker(1, exits),
                    self.top1acc_thresholds,
                    self.save_raw,
                ),
                Comparison(
                    "Trunc Base-2 Softmax",
                    self._fast_softmax_comparison,
                    Tracker(test_dl.batch_size, exits, self.sample_total),
                    AccuTracker(1, exits),
                    self.top1acc_thresholds,
                    self.save_raw,
                ),
                Comparison(
                    "Non-Trunc Base-2 Softmax",
                    self._base2_softmax_comparison,
                    Tracker(test_dl.batch_size, exits, self.sample_total),
                    AccuTracker(1, exits),
                    self.top1acc_thresholds,
                    self.save_raw,
                ),
                Comparison(
                    "Base-2 Sub-Softmax",
                    self._base2_sub_softmax_comparison,
                    Tracker(test_dl.batch_size, exits, self.sample_total),
                    AccuTracker(1, exits),
                    self.top1acc_thresholds,
                    self.save_raw,
                ),
            ]

        # total exit accuracy over the test data
        self.accu_track_totl = AccuTracker(test_dl.batch_size, exits, self.sample_total)

        self.top1_pc = None  # % exit for top1 confidence
        self.entr_pc = None  # % exit for entropy confidence
        self.top1_accu = None  # accuracy of exit over exited samples
        self.entr_accu = None  # accuracy of exit over exited samples
        self.full_exit_accu = None  # accuracy of the exits over all samples
        self.top1_accu_tot = None  # total accuracy of network given exit strat
        self.entr_accu_tot = None  # total accuracy of network given exit strat

    def _thr_max_softmax(self, exit_results, thr):
        ### NOTE DEFINING TOP1 of SOFTMAX DECISION
        sftmax = nn.functional.softmax(exit_results, dim=-1)
        # getting maximum values from softmax op
        sftmx_max = torch.max(sftmax, dim=-1).values
        # comparing to threshold to get boolean tensor mask for exit
        exit_mask = sftmx_max.gt(thr)
        return exit_mask

    def _thr_entropy(self, exit_results, thr):
        ### NOTE DEFINING entropy less than threshold
        sftmax = nn.functional.softmax(exit_results, dim=-1)
        entr = sftmax.log().mul(sftmax).nan_to_num().sum(dim=-1).mul(-1)
        exit_mask = entr.lt(thr)
        return exit_mask

    def _thr_compare_(self, exit_track, accu_track, results, gnd_trth, thrs, thr_func):
        # generate all false mask
        prev_mask = torch.tensor(
            [False] * self.batch_size, dtype=torch.bool, device=self.device
        )
        for i, (exit, thr) in enumerate(zip(results, thrs)):
            # call function to generate mask
            exit_mask = thr_func(exit, thr)
            # mask out values that previously exited
            exit_mask = exit_mask.logical_and(prev_mask.logical_not())
            # get number that are exiting here
            exit_num = exit_mask.sum()
            # updated the number exiting
            exit_track.add_val(exit_num, bin_index=i)
            # update accuracy, along with number exiting here
            accu_track.update_correct(
                exit[exit_mask],
                gnd_trth[exit_mask],
                accum_count=exit_num,
                bin_index=i,
            )
            # update exit mask
            prev_mask = exit_mask

    def _entropy_comparison(
        self, layer: torch.Tensor, thresh: float, test=False
    ) -> bool:
        softmax = nn.functional.softmax(layer, dim=-1)
        entr = -torch.sum(torch.nan_to_num(softmax * torch.log(softmax)), dim=-1)

        if test:
            return entr < thresh, entr
        else:
            return entr < thresh

    def _softmax_comparison(
        self, layer: torch.Tensor, thresh: float, test=False
    ) -> torch.Tensor:
        softmax = nn.functional.softmax(layer, dim=-1)
        # breakpoint()

        sftmx_max = torch.max(softmax, dim=-1).values

        if test:
            return sftmx_max > thresh, softmax
        else:
            return sftmx_max > thresh

    def _fast_softmax_comparison(
        self, layer: torch.Tensor, thresh: float, test=False
    ) -> bool:
        softmax = hw_sim.base2_softmax_torch(layer)
        # softmax = hw_sim.subMax_softmax(exit)
        sftmx_max = torch.max(softmax, dim=-1).values

        if test:
            return sftmx_max > thresh, softmax
        else:
            return sftmx_max > thresh

    def _base2_softmax_comparison(
        self, layer: torch.Tensor, thresh: float, test=False
    ) -> bool:
        softmax = hw_sim.nonTrunc_base2_softmax_torch(layer)
        # softmax = hw_sim.subMax_softmax(exit)
        sftmx_max = torch.max(softmax, dim=-1).values

        if test:
            return sftmx_max > thresh, softmax
        else:
            return sftmx_max > thresh

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
                total=self.sample_total * self.test_dl.batch_size,
                unit="samples",
            ) as pbar:
                for xb, yb in self.test_dl:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    res = self.model(
                        xb
                    )  # implicitly calls forward and returns array of arrays of the final layer for each exit (techically list of tensors for each exit)
                    # res has dimension [num_exits, batch_size, num_classes]

                    # # accuracy of exits over everything
                    # self.accu_track_totl.update_correct(res,yb)
                    # # maximum value of softmax (top1) GREATER than thr
                    # self._thr_compare_(self.exit_track_top1,
                    #     self.accu_track_top1,
                    #     res, yb, self.top1acc_thresholds,
                    #     self._thr_max_softmax)
                    # # entropy of softmax is LOWER than threshold
                    # self._thr_compare_(self.exit_track_entr,
                    #     self.accu_track_entr,
                    #     res, yb, self.entropy_thresholds,
                    #     self._thr_entropy)

                    self.accu_track_totl.update_correct_list(res, yb)

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

                    if self.comp_funcs is not None:
                        for comp in self.comp_funcs:
                            self.comparators[comp].eval(res, yb)
                    else:
                        for comp in self.comparators:
                            comp.eval(res, yb)

                    pbar.update(self.test_dl.batch_size)

    def _test_single_exit(self):
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            for xb, yb in self.test_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                res = self.model(xb)
                self.accu_track_totl.update_correct(res, yb)

    def debug_values(self):
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            for xb, yb in self.test_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                res = self.model(xb)
                for i, exit in enumerate(res):
                    # print("raw exit {}: {}".format(i, exit))
                    softmax = nn.functional.softmax(exit, dim=-1)
                    # print("softmax exit {}: {}".format(i, softmax))
                    sftmx_max = torch.max(softmax)
                    print("exit {} max softmax: {}".format(i, sftmx_max))
                    entr = -torch.sum(torch.nan_to_num(softmax * torch.log(softmax)))
                    print("exit {} entropy: {}".format(i, entr))
                    # print("exit CE loss: {}".format(loss_f(exit,yb)))

    def get_stats(self):
        return_val = {}

        return_val["comps"] = []

        if self.comp_funcs is not None:
            for comp in self.comp_funcs:
                return_val["comps"].append(self.comparators[comp].get_comp_info())
        else:
            for comp in self.comparators:
                return_val["comps"].append(comp.get_comp_info())

        return_val["num_exits"] = self.exits
        return_val["num_samples"] = self.sample_total * self.test_dl.batch_size
        return_val["batch_size"] = self.test_dl.batch_size
        return_val["accu_per_exit"] = self.accu_track_totl.get_accu(return_list=True)

        if self.save_raw:
            return_val[
                "true_indices"
            ] = (
                self.true_result.tolist()
            )  # grab from any of the comparators that were used
            return_val["raw_layer"] = self.raw_layer.tolist()

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
        else:
            self._test_single_exit()
            print("### TEST FINISHED ###")

        # accuracy of each exit over FULL data set
        print("Total Accuracy:", self.accu_track_totl.get_accu(return_list=True))
        # TODO save test stats along with link to saved model
