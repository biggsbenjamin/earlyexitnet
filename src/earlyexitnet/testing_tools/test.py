"""
Class for testing early exit and normal CNNs.
Includes requires accuracy and loss trackers from tools
"""

# pytorch imports
import torch
from torch import nn#, optim

# general imports
import numpy as np
#from datetime import datetime as dt
#from typing import Callable
#from time import perf_counter
from tqdm import tqdm

# import custom funcions to simulate hardware
from earlyexitnet.testing_tools import hw_sim

# importing trackers for loss + accuracy, and generic tracker for exit distribution
from earlyexitnet.tools import Tracker, AccuTracker#LossTracker

class Tester:
    def __init__(
        self,
        model,
        test_dl,
        loss_f=nn.CrossEntropyLoss(),
        exits=2,
        top1acc_thresholds=[],
        fast_thresholds=None,
        fast_noTrunc_thresholds=None,
        fast_sub_thresholds=None,
        fast_sub_bitAcc_thresholds=None,
        fast_base4_threholds=None,
        fast_base4_sub_threholds=None,
        entropy_thresholds=[],
        conf_funcs=None,
        device=None,
        save_raw=False, # NOTE save_raw is expensive
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
            fast_thresholds if fast_thresholds is not None else top1acc_thresholds,
            fast_noTrunc_thresholds if fast_noTrunc_thresholds is not None else top1acc_thresholds,
            fast_sub_thresholds if fast_sub_thresholds is not None else top1acc_thresholds,
            fast_sub_bitAcc_thresholds if fast_sub_bitAcc_thresholds is not None else top1acc_thresholds,
            #fast_base4_threholds if fast_base4_threholds is not None else top1acc_thresholds,
            #fast_base4_sub_threholds if fast_base4_sub_threholds is not None else fast_sub_thresholds,
            ]

        # list of AVAILABLE conf threshold methods
        self.conf_list = [
            self._thr_entropy,
            self._thr_max_softmax,
            self._thr_max_softmax_fast,
            self._thr_max_softmax_fast_noTrunc,
            self._thr_max_softmax_fast_sub,
            self._thr_max_softmax_fast_sub_bitAcc,
            #self._thr_max_softmax_fast_base4,
            #self._thr_max_softmax_fast_base4_sub
        ]

        # select functions chosen during cli
        if conf_funcs is None:
            # FIXME need default thresholds for other metrics
            # they function like the softmax on so should be similar
            #self.conf_funcs = [(f, t) for f,t in zip(self.conf_list, self.thrs)]
            self.conf_funcs = list(zip(self.conf_list, self.thrs))
        else:
            self.conf_funcs = [(self.conf_list[func], self.thrs[func]) \
                    for func in conf_funcs]

        self.save_raw = save_raw
        if self.save_raw:
            print("Saving raw values, test will take longer.")
            self.true_result = None
            self.raw_layer = None
            self.raw_softmax = {}

        # set up the device
        if device is None or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = device
        print(f"Running on device: {self.device}")

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
        if self.save_raw:
            # ground truth
            self.stats_dict["true_indices"] = None
            # raw network output
            self.stats_dict["raw_layer"] = None
        # total exit accuracy over the test data
        self.accu_track_totl = AccuTracker(test_dl.batch_size, exits, self.sample_total)
        if exits > 1:
            # set up stat trackers
            for (func, _) in self.conf_funcs:
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
                    self.raw_softmax[str(func.__name__)] = None
                    self.stats_dict['conf_metrics'][str(func.__name__)]["raw_softmax"]=\
                        None

    def _thr_entropy(self, exit_results, thr):
        ### NOTE DEFINING entropy less than threshold
        sftmax = nn.functional.softmax(exit_results, dim=-1)
        entr = sftmax.log().mul(sftmax).nan_to_num().sum(dim=-1).mul(-1)
        exit_mask = entr.lt(thr)
        if not self.save_raw:
            return exit_mask
        return exit_mask, entr

    def _thr_max_softmax(self, exit_results, thr):
        ### NOTE DEFINING TOP1 of SOFTMAX DECISION
        sftmax = nn.functional.softmax(exit_results, dim=-1)
        # getting maximum values from softmax op
        sftmx_max = torch.max(sftmax, dim=-1).values
        # comparing to threshold to get boolean tensor mask for exit
        exit_mask = sftmx_max.gt(thr)
        if not self.save_raw:
            return exit_mask
        return exit_mask, sftmax

    def _thr_max_softmax_fast(self, exit_results, thr):
        # max softmax function but with base 2 usage
        # 2 to power of truncated logit values - integer component
        pow2 = torch.pow(2, torch.trunc(exit_results))
        # normalise exp wrt sum of exp
        pow2_max = pow2.div(pow2.sum(dim=-1).unsqueeze(1))
        # get max for each result in batch
        sftmx_max = torch.max(pow2_max, dim=-1).values
        # comparing to threshold to get boolean tensor mask for exit
        exit_mask = sftmx_max.gt(thr)
        if not self.save_raw:
            return exit_mask
        return exit_mask, pow2_max

    def _thr_max_softmax_fast_noTrunc(self, exit_results, thr):
        # max softmax function but with base 2 usage and no truncation
        # 2 to power of logit values
        pow2 = torch.pow(2, exit_results)
        # normalise exp wrt sum of exp
        pow2_max = pow2.div(pow2.sum(dim=-1).unsqueeze(1))
        # get max for each result in batch
        sftmx_max = torch.max(pow2_max, dim=-1).values
        # comparing to threshold to get boolean tensor mask for exit
        exit_mask = sftmx_max.gt(thr)
        if not self.save_raw:
            return exit_mask
        return exit_mask, pow2_max

    def _thr_max_softmax_fast_sub(self, exit_results, thr):
        # TODO make these variables:
        # Input fixed datatype
        #IN_FRAC_W = 8
        #IN_INT_W = 8
        #IN_TOT_W = IN_INT_W + IN_FRAC_W
        # Internal accumulation datatype
        #ACCUM_INT_W = 1
        ACCUM_FRAC_W = 28
        #ACCUM_TOT_W = ACCUM_INT_W + ACCUM_FRAC_W

        # find max value - not affected by precision (in a meaningful way)
        batch_max = torch.max(exit_results, dim=-1).values
        # subtract max from each value
        # take integer component of results
        # negate (on hw this is NOT and +1)
        subs = -torch.trunc(exit_results - batch_max.unsqueeze(1))
        # IF negate > num of frac bits
        nzero_mask = subs.lt(ACCUM_FRAC_W) # true value is not shifted
        shifters = torch.zeros(subs.shape, device=self.device)
            # THEN
                # result is too small to care, 2^val = 0
            # ELSE:
                # take 1.0...0 bitacc val and shift right by negation result
        # NOTE can't do bitwise shift for floats, do division instead lol
        shifters[nzero_mask] = 1.0
        shifters[nzero_mask] = shifters[nzero_mask].div(torch.pow(2, subs[nzero_mask]))
        # sum the values
        # exit IF 1 > sum * thr
        exit_mask = shifters.sum(dim=-1).mul(thr).le(1)
        if not self.save_raw:
            return exit_mask
        return exit_mask, shifters

    def _thr_max_softmax_fast_sub_bitAcc(self, exit_results, thr):
        # supposedly hls, bit accurate method for max sftmax using:
        # base2, truncated, with sub
        # getting the fxp object for the results
        eToz_arr, sum_eToz_arr = hw_sim.base2_subMax_softmax_fixed(exit_results)

        # do we want to stay bitAcc here?
        # whats left?
        # have each of the values in eToz_arr,
        # have the accums in the sum var
        # don't actually need each of the vals, just the sums
        # because of the max subtraction
        # mul is done in bitAcc fashion so that's fine to change prec
        mm_res = np.less_equal(np.multiply(sum_eToz_arr, thr), 1)
        # boolean array, convert back to torch
        exit_mask = torch.from_numpy(mm_res.get_val()).to(self.device).gt(0).flatten()
        if not self.save_raw:
            return exit_mask
        # eToz_arr is still in fxp format...
        return exit_mask, torch.from_numpy(eToz_arr.get_val())

    def _thr_max_softmax_fast_base4(self, exit_results, thr):
        # max softmax function but with base 4 usage
        # 4 to power of truncated logit values - integer component
        pow4 = torch.pow(4, torch.trunc(exit_results))
        # normalise exp wrt sum of exp
        pow4_max = pow4.div(pow4.sum(dim=-1).unsqueeze(1))
        # get max for each result in batch
        sftmx_max = torch.max(pow4_max, dim=-1).values
        # comparing to threshold to get boolean tensor mask for exit
        exit_mask = sftmx_max.gt(thr)
        if not self.save_raw:
            return exit_mask
        return exit_mask, pow4_max

    def _thr_max_softmax_fast_base4_sub(self, exit_results, thr):
        # TODO make these variables:
        # Input fixed datatype
        #IN_FRAC_W = 8
        #IN_INT_W = 8
        #IN_TOT_W = IN_INT_W + IN_FRAC_W
        # Internal accumulation datatype
        #ACCUM_INT_W = 1
        ACCUM_FRAC_W = 28
        #ACCUM_TOT_W = ACCUM_INT_W + ACCUM_FRAC_W

        # find max value - not affected by precision (in a meaningful way)
        batch_max = torch.max(exit_results, dim=-1).values
        # subtract max from each value
        # take integer component of results
        # negate (on hw this is NOT and +1)
        subs = -torch.trunc(exit_results - batch_max.unsqueeze(1))
        # IF negate > num of frac bits
        nzero_mask = subs.lt(ACCUM_FRAC_W) # true value is not shifted
        shifters = torch.zeros(subs.shape, device=self.device)
            # THEN
                # result is too small to care, 2^val = 0
            # ELSE:
                # take 1.0...0 bitacc val and shift right by negation result
        # NOTE can't do bitwise shift for floats, do division instead lol
        shifters[nzero_mask] = 1.0
        shifters[nzero_mask] = shifters[nzero_mask].div(torch.pow(4, subs[nzero_mask]))
        # sum the values
        # exit IF 1 > sum * thr
        exit_mask = shifters.sum(dim=-1).mul(thr).le(1)
        if not self.save_raw:
            return exit_mask
        return exit_mask, shifters

    def _thr_compare_(self, exit_track, accu_track,
            results, gnd_trth, thrs, thr_func):
        # generate all false mask
        prev_mask = torch.tensor([False] * self.batch_size,
                dtype=torch.bool, device=self.device
        )
        raw_norm = None
        for i,(exit_res,thr) in enumerate(zip(results,thrs)):
            # call function to generate mask
            if not self.save_raw:
                exit_mask = thr_func(exit_res,thr)
            else:
                exit_mask, nxt_raw_norm = thr_func(exit_res,thr)
                raw_norm = (
                    torch.stack((raw_norm, nxt_raw_norm))
                    if raw_norm is not None
                    else nxt_raw_norm
                )
            # mask out values that previously exited
            exit_mask = exit_mask.logical_and(prev_mask.logical_not())
            # get number that are exiting here
            exit_num = exit_mask.sum()
            # updated the number exiting
            exit_track.add_val(exit_num,bin_index=i)
            # update accuracy, along with number exiting here
            accu_track.update_correct(
                exit_res[exit_mask],
                gnd_trth[exit_mask],
                accum_count=exit_num,
                bin_index=i,
            )
            # update exit mask
            prev_mask = exit_mask
        return raw_norm

    # NOTE keeping commented func for rebase with data analysis
    #def _base2_sub_softmax_comparison(
    #    self, layer: torch.Tensor, thresh: float, test=False
    #) -> bool:
    #    exp, sums = hw_sim.base2_subMax_softmax_fixed(layer)
    #    # softmax = hw_sim.subMax_softmax(exit)
    #    # sftmx_max = torch.max(softmax, dim=-1).values

    #    max_exp = np.max(exp, -1)

    #    threshes = (sums * thresh).flatten()

    #    # breakpoint()
    #    if test:
    #        return (
    #            torch.BoolTensor(max_exp > threshes, type=bool),
    #            torch.Tensor(exp.get_val()) / torch.Tensor(sums.get_val()),
    #        )
    #    else:
    #        return torch.BoolTensor(max_exp > threshes)

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
                    # implicitly calls forward and returns array of arrays of the
                    # final layer for each exit (techically list of tensors for each exit)
                    # res has dimension [num_exits, batch_size, num_classes]
                    res = self.model(xb)

                    # accuracy of exits over everything
                    self.accu_track_totl.update_correct(res,yb)

                    # Compute confidence values for each metric
                    for (conf,thrs) in self.conf_funcs:
                        # trackers are updated inplace
                        raw_norm = self._thr_compare_(
                                self.tracker_dict[str(conf.__name__)]['exit'],
                                self.tracker_dict[str(conf.__name__)]['accu'],
                                res, yb,
                                thrs,
                                conf
                            )
                        if self.save_raw:
                            # need to save output B4 threshold
                            self.raw_softmax[str(conf.__name__)] = (
                                torch.concatenate(
                                (self.raw_softmax[str(conf.__name__)], raw_norm), dim=1)
                                if self.raw_softmax[str(conf.__name__)] is not None
                                else raw_norm
                            )

                    # NOTE still not completely sure what this does
                    # leaving in for rebase with data_analysis
                    if self.save_raw:
                        # incrementally create a batch sized tensor of ground truth
                        self.true_result = (
                            torch.cat((self.true_result, yb))
                            if self.true_result is not None
                            else yb
                        )
                        # simlarly, concat along the 2nd dim the 10 wide vector of
                        # the raw model output (no norm).
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
        """
        for debugging the test set up.
        Prints the top1 softmax and entropy values as inputs are run through
        the model.
        """
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            for xb,yb in self.test_dl:
                xb,yb = xb.to(self.device),yb.to(self.device)
                res = self.model(xb)
                for i, exit_res in enumerate(res):
                    #print("raw exit {}: {}".format(i, exit))
                    softmax = nn.functional.softmax(exit_res,dim=-1)
                    #print("softmax exit {}: {}".format(i, softmax))
                    sftmx_max = torch.max(softmax)
                    print(f"exit {i} max softmax: {sftmx_max}")
                    entr = -torch.sum(torch.nan_to_num(softmax * torch.log(softmax)))
                    print(f"exit {i} entropy: {entr}")
                    #print("exit CE loss: {}".format(loss_f(exit,yb)))

    def print_tracker_info(self, name_str):
        """
        Prints info on a per confidence metric basis so need to provide it with refs
        """
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
        """
        Get the confidence metric statistics for those used during the test.
        If raw data is required, this is provided when save_raw is enabled.
        """
        for (func,thrs) in self.conf_funcs:
            name_str = str(func.__name__)
            ex_avg = self.tracker_dict[name_str]['exit'].get_avg(return_list=True)
            ac_avg = self.tracker_dict[name_str]['accu'].get_avg(return_list=True)

            self.stats_dict['conf_metrics'][name_str]["exit_pc"] = ex_avg
            self.stats_dict['conf_metrics'][name_str]["accu_pc"] = ac_avg
            self.stats_dict['conf_metrics'][name_str]["exit_threshs"] = thrs
            self.stats_dict['conf_metrics'][name_str]["combined_accuracy"] = np.dot(
                ex_avg, ac_avg)

            if self.save_raw:
                # save the results of the softmax with different bases
                # (also the raw entropy value)
                self.stats_dict['conf_metrics'][name_str]["raw_softmax"] = \
                    self.raw_softmax[name_str].tolist()

        self.stats_dict["num_exits"] = self.exits
        self.stats_dict["num_samples"] = self.sample_total * self.test_dl.batch_size
        self.stats_dict["batch_size"] = self.test_dl.batch_size
        self.stats_dict["accu_per_exit"] = self.accu_track_totl.get_accu(return_list=True)

        if self.save_raw:
            # get the ground truth values
            self.stats_dict["true_indices"] = self.true_result.tolist()
            # grab from any of the comparators that were used
            self.stats_dict["raw_layer"] = self.raw_layer.tolist()
        return self.stats_dict

    def test(self):
        """
        Run the test!
        """
        print(f"Test of length {self.sample_total} starting:")
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
