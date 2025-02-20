"""
Python scripts to access graph gen stuff to pull graph data to be used
in pgfplots tikz latex graphs.
"""

import os
import json
#import csv
from datetime import datetime as dt
import numpy as np
import pandas as pd
#import torch
from sklearn import metrics
from earlyexitnet.data_analysis.graphs import fit_kernel

def adj_density(target_tot, curr_tot, arr_in, xbins):
    """
    instead of changing the total values, return a bar chart set
    x and y
    """
    #to_add = target_tot-curr_tot
    scale = target_tot/curr_tot
    #create a list per bin
    #adj_out=None
    densities = []
    for idx, b_int in enumerate(xbins[:-1]):
        bin_mask = np.logical_and((arr_in >= b_int), (arr_in < xbins[idx+1]))
        bin_vals = arr_in[bin_mask]
        bin_len = bin_mask.sum()
        #adj = (bin_len/curr_tot) * scale
        adj = bin_len/(target_tot * abs(xbins[idx+1]-b_int))
        densities.append(adj)

        #new_bin_len = round(bin_len*scale)
        # number of new vals to add to adjust density
        #diff = new_bin_len - bin_len
        #if diff > 0:
        #    adj_bin_vals = np.append(
        #        bin_vals, np.full(shape=diff, fill_value=bin_vals[0]))
        #    # append the adj list
        #    if adj_out is None:
        #        adj_out = np.copy(adj_bin_vals)
        #    else:
        #        adj_out = np.append(adj_out, adj_bin_vals)
        #else:
        #    if adj_out is None:
        #        adj_out = np.copy(bin_vals)
        #    else:
        #         adj_out = np.append(adj_out, bin_vals)
        ## reduce the values by the diff so that we don't go over density
        #to_add = to_add - diff
        #if to_add <= 0:
        #   continue
    den_arr = np.array(densities)
    #den_sum = den_arr.sum()
    return den_arr# / den_sum

def graph_dat():
    """
    Gen dat data
    """
    # load the json file with the data in it
    json_file = '~/earlyexitnet/model-outputs/b_lenet_cifar/cifar10_raw_train_output_2025-01-23_122425.json'
    fp = os.path.expanduser(json_file)
    with open(fp) as jdat:
        data = json.load(jdat)
    # pull raw layer info out and construct data
    title_name = data["model"] + " " + data["dataset"]
    truth_idxs = data["true_indices"] # ground truth classifications
    raw_layer = np.array(data["raw_layer"]) # raw final layer outputs per ex

    #exit_num = data["num_exits"]
    #class_num = np.array(raw_layer).shape[-1]
    # get datetime
    ts = dt.now().strftime("%Y%m%d_%H%M%S")
    # [num_exits, num_samples, num_classes]
    model_pred_idx = np.argmax(raw_layer, -1)

    # pull the raw_softmax info out
    #pt_softmax_values = data["conf_metrics"]["_thr_max_softmax"]["raw_softmax"]

    # get per exit correct/incorrect
        # generate bool mask for each exit based on if prediction
        # matches ground truth
    ee_correct_mask = model_pred_idx[0] == truth_idxs
    fe_correct_mask = model_pred_idx[1] == truth_idxs

    # get difficulty rating - might be interesting to plot the othrs
    # construct weighting system where values that are identified as
    # correct earlier are given more weight
    # for model with 2 exits:
    # 0 means it was always misclassified
    # 1 means it was correctly identified at the final exit
    # 2 means it was identified correctly at first then misclassified (overthinking)
    # 3 means it was identified correctly both times

    # NOTE not sure how helpful it is to group d0, d1 etc.
    # what about grouping into - dont exit and DO exit
    # DO exit: d0(wasted compute) + d2(overthinking!) + d3(wasted compute)
    # DONT exit: d1

    # only misclassifications in the first exit
    #chosen_diffies = [0, 1]
    # construct d=0 mask
    d0_mask = np.logical_and(
        np.logical_not(ee_correct_mask),
        np.logical_not(fe_correct_mask)
    )
    d0_total = d0_mask.sum()
    print(f"d0 total: {d0_total}")
    d1_mask = np.logical_and(
        np.logical_not(ee_correct_mask),
        fe_correct_mask
    )
    d0d1_mask = np.logical_or(d0_mask, d1_mask)
    print(f"d0d1 tot: {d0d1_mask.sum()}")

    DONT_mask = d1_mask
    print(f"DONT total: {DONT_mask.sum()}")
    DO_mask = np.logical_not(d1_mask)
    print(f"DO total: {DO_mask.sum()}")

    # determine x axis thresholds, how many bins?
    raw_layer_maxs = np.max(raw_layer, axis=-1)
    print(f"raw_layer_maxs {raw_layer_maxs.shape}")
    maxmax_val = raw_layer_maxs.max()
    minmax_val = raw_layer_maxs.min()
    print(f"Raw layer Maximums max:{maxmax_val}, min:{minmax_val}")
    bins = 100
    rw_lyr_x = np.linspace(minmax_val, maxmax_val, bins)

    # gnerating right wrong list
    rw_mask = model_pred_idx == truth_idxs
    print(f"rw_mask {rw_mask.shape}")

    # getting these results for both exits I suppose
    # NOTE these are lists of arrays of different sizes
    raw_lyr_mx_d0   = [raw_layer_maxs[0][d0_mask],
                       raw_layer_maxs[1][d0_mask]
                       ]

    adj0 = adj_density(d0d1_mask.sum(), d0_mask.sum(), raw_lyr_mx_d0[0], rw_lyr_x)
    adj1 = adj_density(d0d1_mask.sum(), d0_mask.sum(), raw_lyr_mx_d0[1], rw_lyr_x)
    adj0 = np.stack((rw_lyr_x[:-1], adj0), axis=-1)
    adj1 = np.stack((rw_lyr_x[:-1], adj1), axis=-1)
    raw_lyr_d0_densityadj = [adj0, adj1]
    print("density adjs", raw_lyr_d0_densityadj[0].shape, raw_lyr_d0_densityadj[1].shape)
    print(f"first row: {adj0[0]} {adj1[0]}")
    print(f"d0 sum ee {raw_lyr_d0_densityadj[0].sum(axis=0)}")
    print(f"d0 sum fe {raw_lyr_d0_densityadj[1].sum(axis=0)}")

    raw_lyr_mx_d0d1 = [raw_layer_maxs[0][d0d1_mask],
                       raw_layer_maxs[1][d0d1_mask]
                       ]
    raw_lyr_mx_DO   = [raw_layer_maxs[0][DO_mask],
                       raw_layer_maxs[1][DO_mask]
                       ]
    raw_lyr_mx_DN   = [raw_layer_maxs[0][DONT_mask],
                       raw_layer_maxs[1][DONT_mask]
                       ]
    raw_lyr_mx_corr = [raw_layer_maxs[0][rw_mask[0]],
                       raw_layer_maxs[1][rw_mask[1]]
                       ]
    raw_lyr_mx_inco = [raw_layer_maxs[0][np.logical_not(rw_mask[0])],
                       raw_layer_maxs[1][np.logical_not(rw_mask[1])]
                       ]

    # NOTE some sanity checks:
    print(raw_lyr_mx_corr[0].shape + raw_lyr_mx_inco[0].shape)
    print(raw_lyr_mx_corr[1].shape + raw_lyr_mx_inco[1].shape)
    print(raw_lyr_mx_DO[0].shape, raw_lyr_mx_DN[0].shape)
    print(raw_lyr_mx_DO[1].shape, raw_lyr_mx_DN[1].shape)

    # where am I saving it?
    home_dir = os.path.expanduser('~')
    save_dir = os.path.join(home_dir,
                f'earlyexitnet/outputs/{data["model"]}/pgfdata/{ts}/')
    # make sure it exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #S HIST raw layer dist per exit - correct,incorrect, kernels, d0,d1
    # x ax just vals
    save_path = os.path.join(save_dir, "rawVals_histbins.csv")
    raw_df_xvals = pd.DataFrame(rw_lyr_x)
    raw_df_xvals.to_csv(save_path, index=False)
    hist_nm     = ["correct", "incorrect", "Do_Exit", "DNT_Ext", "d0DenAdj", "d0d1"]
    hist_dat    = [raw_lyr_mx_corr, raw_lyr_mx_inco, raw_lyr_mx_DO, raw_lyr_mx_DN,
                   raw_lyr_d0_densityadj, raw_lyr_mx_d0d1]

    for exi, ex in enumerate(["ee", "fe"]):
        for nm, dat in zip(hist_nm, hist_dat):
            raw_df = pd.DataFrame(dat[exi])
            save_path = os.path.join(save_dir, f"rawVals_{nm}_{ex}.csv")
            if nm == "d0DenAdj":
                raw_df.to_csv(save_path, index=False, header=["x","y"])
            else:
                raw_df.to_csv(save_path, index=False)

    for conf in data['conf_metrics'].keys():
        if 'base4' in conf:
            # skipping these, not enough time
            continue
        elif ('sub' in conf) and ('bitAcc' not in conf):
            continue
        name = conf[5:]
        print(f"Confidence function: {name}")


        # FIXME make this general for more than 2 exits
        raw_norm_ee = np.array(data["conf_metrics"][conf]["raw_softmax"][0])
        # set up the differences between the metrics...

        # this sets up an x axis limited to between 0,1
        # makes sense for the softmax variants...
        # BUT NOT for entropy
        bins=50
        histbins = np.linspace(0, 1, bins)
        auroc_thresholds = np.linspace(0, 1, 200)
        gt_inequality_bool = True
        norm_proc_norm = None
        norm_histbins = None

        if "entropy" in conf.lower():
            print(f"Found entropy! {name}")
            # raw_norm values are the calculated entropy values of
            # the logits, passed through softmax
            # NOTE these apply to auroc
            proc_norm = raw_norm_ee
            histbins = np.linspace(0, max(raw_norm_ee), bins)
            # auroc has higher res
            auroc_thresholds = np.linspace(max(raw_norm_ee), 0, 200)
            auprc_thresholds = np.linspace(0, 1, 200)
            # move between 0 and 1 so that the plots
            # to be compared to the softmax-based metrics
            # these should apply plots r/w and diffi
            scaled_entr = np.divide(raw_norm_ee, raw_norm_ee.max())
            #scaled_entr = 1/proc_norm
            norm_proc_norm = np.add(np.multiply(scaled_entr, -1), 1)
            #norm_proc_norm = scaled_entr # SHOULD just have the ineq flipped
            norm_histbins = np.linspace(0, 1, bins)
            gt_inequality_bool = False
        elif "sub" in conf.lower():
            print(f"Found a sub! {name}")
            # the max value will always be 1 by design...
            # metric: 1 > thr * sum
            proc_norm = np.sum(raw_norm_ee, axis=-1)
            proc_norm = 1 / proc_norm
        else:
            # for standard softmax
            proc_norm = np.max(raw_norm_ee, -1)

        # generate the tp/fp/tn/fn rates

        # [threshold, true pos rate, false pos rate]
        roc = np.zeros((len(auroc_thresholds), 3,))
        num_total = len(proc_norm)
        pr_pc_arr = np.zeros((len(auroc_thresholds), 4,))

        # go through each threshold and determine exit state
        for i, thr in enumerate(auroc_thresholds):
            if gt_inequality_bool: # greater than threshold
                ee_true_mask = proc_norm > thr
            else:
                ee_true_mask = proc_norm < thr
            """
            msb ee true?
            midsb ee correct?
            lsb fe correct?
            # last cat you save compute on what will be wrong anyway
            tp = cat_dict['cat110'].shape[0] + \
                cat_dict['cat111'].shape[0] + \
                cat_dict['cat100'].shape[0]

            fp = cat_dict['cat101'].shape[0]
            tn = cat_dict['cat001'].shape[0]

            # last cat you save compute on what will be wrong anyway
            fn = cat_dict['cat011'].shape[0] + \
                cat_dict['cat010'].shape[0] + \
                cat_dict['cat000'].shape[0]
            """

            true_pos_cnt = np.logical_and(
                ee_true_mask,
                np.logical_not(
                    np.logical_and(
                        np.logical_not(ee_correct_mask),
                        fe_correct_mask)
                )
            ).sum()
            fp_cnt = np.logical_and(
                ee_true_mask,
                np.logical_and(
                    np.logical_not(ee_correct_mask),
                    fe_correct_mask)
            ).sum()
            tn_cnt = np.logical_and(
                np.logical_not(ee_true_mask),
                np.logical_and(
                    np.logical_not(ee_correct_mask),
                    fe_correct_mask)
            ).sum()
            fn_cnt = np.logical_and(
                np.logical_not(ee_true_mask),
                np.logical_not(
                    np.logical_and(
                        np.logical_not(ee_correct_mask),
                        fe_correct_mask)
                )
            ).sum()

            #True +ve rate (recall):         TP/(TP + FN)
            #Specificity:                    TN/(TN + FP)
            #False +ve rate (1-specificity): FP/(TN + FP)

            # true positive rate
            tpr = true_pos_cnt/(true_pos_cnt + fn_cnt)

            # false positive rate
            fpr = fp_cnt/(tn_cnt + fp_cnt)

            # update the roc array with the calc values
            roc[i][0] = thr
            roc[i][1] = tpr
            roc[i][2] = fpr


            exit_pc = ee_true_mask.sum()/num_total

            if true_pos_cnt + fp_cnt > 0:
                prec = true_pos_cnt/(true_pos_cnt+fp_cnt)
            else:
                prec = 1

            # [recall, precision, thr, exit %]
            pr_pc_arr[i][0] = tpr # NOTE recall and tpr the same
            pr_pc_arr[i][1] = prec

            if "entropy" in conf.lower():
                # re eval mask with norm vals
                ee_true_mask = norm_proc_norm > auprc_thresholds[i]
                # get exit perc
                exit_pc = ee_true_mask.sum()/num_total
                # set thr to norm thr val
                pr_pc_arr[i][2] = auprc_thresholds[i]
            else:
                pr_pc_arr[i][2] = thr
            pr_pc_arr[i][3] = exit_pc

            #print(f"tp{true_pos_cnt} fp{fp_cnt} tn{tn_cnt} fn{fn_cnt}")
        # might not even need to run the graph funcs?
        # definitely need the auroc value - check where this comes from

        # work out what format the hist data needs to be in for dist plots
        # work out how to pull kernel plots (just to have)

        #S SCAT AUROC plot - tpr vs fpr
            # should have value in legend
        auroc_val = metrics.auc(roc[:,2], roc[:,1])
        auc_str = "0" + str(round(auroc_val*1000))
        print(f"{name} AUC:{auc_str}")
            #convert roc to csv
        auroc_df = pd.DataFrame(roc)
        save_path = os.path.join(save_dir, f"{name}_aurocVals_{auc_str}.csv")
        auroc_col_names = ["ThrshldVal", "TruPosRate", "FalPosRate"]
        if "entropy" in conf.lower():
            auroc_col_names[0] = "NormThrVal"
        auroc_df.to_csv(save_path, index=False, header=auroc_col_names)

        #S SCAT per ex recall(tpr), accu, prec, fpr, exit %
        # NOTE changed my mind - doing a PR plot
        auprc_val = metrics.auc(pr_pc_arr[:,0], pr_pc_arr[:,1])
        aupr_str = "0" + str(round(auprc_val*1000))
        print(f"{name} AUPRC:{aupr_str}")
            #convert prc to csv
        auprc_df = pd.DataFrame(pr_pc_arr)
        save_path = os.path.join(save_dir, f"{name}_auprcVals_{aupr_str}.csv")
        auprc_col_names = ["Recall","Precision", "Threshold", "ExitPcnt"]
        if "entropy" in conf.lower():
            auprc_col_names[2] = "NormThreshold"
        auprc_df.to_csv(save_path, index=False, header=auprc_col_names)

        if "entropy" in conf.lower():
            # NOTE switching the function of proc norm for ENTROPY
            proc_norm = norm_proc_norm
            histbins = norm_histbins
        #S HIST per conf dist per ex - correct,incorrect, kernels, d0,d1
            # x ax thr values (norm for entropy)
        # NOTE these are lists of arrays of different sizes
        proc_norm_d0 = proc_norm[d0_mask]

        adj0 = adj_density(d0d1_mask.sum(), d0_mask.sum(), proc_norm_d0, histbins)
        proc_norm_d0_densityadj = np.stack((histbins[:-1], adj0), axis=-1)
        print("density adjs", proc_norm_d0_densityadj.shape)
        print(f"d0 sum ee {proc_norm_d0_densityadj[0].sum(axis=0)}")

        proc_norm_d0d1 = proc_norm[d0d1_mask]
        proc_norm_DO   = proc_norm[DO_mask]
        proc_norm_DN   = proc_norm[DONT_mask]
        proc_norm_corr = proc_norm[rw_mask[0]]
        proc_norm_inco = proc_norm[np.logical_not(rw_mask[0])]

        # NOTE some sanity checks:
        print(proc_norm_corr.shape + proc_norm_inco.shape)
        print(proc_norm_corr.shape + proc_norm_inco.shape)
        print(proc_norm_DO.shape, proc_norm_DN.shape)
        print(proc_norm_DO.shape, proc_norm_DN.shape)

        # where am I saving it?
        home_dir = os.path.expanduser('~')
        save_dir = os.path.join(home_dir,
                    f'earlyexitnet/outputs/{data["model"]}/pgfdata/{ts}/')
        # make sure it exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        #S HIST raw layer dist per exit - correct,incorrect, kernels, d0,d1
        hist_nm  = ["correct", "incorrect", "d0DenAdj", "d0d1"]#, "Do_Exit", "DNT_Ext"]
        hist_dat = [proc_norm_corr, proc_norm_inco, proc_norm_d0_densityadj, proc_norm_d0d1]
        #proc_norm_DO, proc_norm_DN]

        for nm, dat in zip(hist_nm, hist_dat):
            conf_df = pd.DataFrame(dat)
            # TODO changes this to name rather than conf
            save_path = os.path.join(save_dir, f"metric_{conf}_{nm}.csv")
            if nm == "d0DenAdj":
                conf_df.to_csv(save_path, index=False, header=["x","y"])
            else:
                conf_df.to_csv(save_path, index=False)

    return


if __name__ == "__main__":
    graph_dat()
