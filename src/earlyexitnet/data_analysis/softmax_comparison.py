"""
Comparing behaviour of softmax functions
"""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from earlyexitnet.data_analysis.graphs import plot_right_wrong, plot_difficulties,\
plot_auroc, plot_false_positives, group_by_class, fit_kernel, make_axes, group_by_1D

def DOCTOR_softmax_from_softmax(softmax: np.ndarray):
    """
    formula from DOCTOR paper
    """
    g = np.square(softmax).sum(-1)

    return g

def DOCTOR_softmax_from_raw(raw_layer: np.ndarray):
    """
    formula from DOCTOR paper but with raw logits as input
    """
    # exp = np.exp(raw_layer)
    exp = np.power(2, raw_layer)

    square_exp = np.power(exp, 2)

    square_sum = np.power(exp.sum(-1), 2)

    g = square_exp.sum(-1) / square_sum

    return g

def raw_distance(raw_layer: np.ndarray):
    """
    Calculate difference between the max val and the average of the
    remaining values
    """
    max_val = raw_layer.max(-1)
    num_classes = raw_layer.shape[-1]
    avg_vals = (raw_layer.sum(-1) - max_val) / (num_classes - 1)
    dist = max_val - avg_vals
    return dist

def raw_distance_daghero(raw_layer: np.ndarray):
    """
    Calculate the distance between the max val and the
    second highest value.
    """
    sorted_vals = np.sort(raw_layer, axis=-1)

    max_val = sorted_vals[:, :, -1]
    second_max = sorted_vals[:, :, -2]

    dist = max_val - second_max
    return dist

def main(json_file, funcs=None, plot_classes=False):
    # json_file = "./rawSoftmax_b_lenet_se_singleThresh_2023-07-19_153525.json"
    # json_file = "./b_lenet_cifar_singleThresh_2023-07-20_172520.json"

    # open json file containing the test values
    # NOTE a test should have been run with save_raw=True
    # to produce the raw_softmax values etc.
    with open(json_file) as json_data:
        data = json.load(json_data)

    # FIXME manually added in the jupyter code
    # move this to Tester class
    title_name = data["model"] + " " + data["dataset"]
    true_vals = data["true_indices"] # ground truth classifications
    raw_layer = data["raw_layer"] # raw final layer outputs for each exit

    # get the raw softmax from the thr_max_softmax confidence metric
    pt_softmax_values = data["conf_metrics"]["_thr_max_softmax"]["raw_softmax"]

    # if pt_softmax_values is not None:
    doctor_sft = DOCTOR_softmax_from_softmax(np.array(pt_softmax_values))
    # add the DOCTOR stats for the softmax values to the metrics dictionary
    data["conf_metrics"]["_thr_"+"DOCTOR_distance_softmax"] = {"raw_softmax":doctor_sft}

    doctor_raw = DOCTOR_softmax_from_raw(np.array(raw_layer))
    # add the DOCTOR stats for the raw layer values to the metrics dictionary
    data["conf_metrics"]["_thr_"+"DOCTOR_distance_raw"] = {"raw_softmax":doctor_raw}

    # custom_func = raw_distance(np.array(raw_layer))
    # data['conf_metrics'].append({
    #   'name':"doctor distance between max and avg",
    #   'raw_softmax':custom_func
    # })

    # generate the doctor curve for distance between top 2 values
    custom_func = raw_distance_daghero(np.array(pt_softmax_values))
    data["conf_metrics"]["_thr_"+"DOCTOR_dist_top2_softmax"] = {"raw_softmax":custom_func}

    # get some constants from the json
    num_exits = data["num_exits"]
    num_classes = np.array(raw_layer).shape[-1]

    # establish colours for correct/incorrect plots
    correct_col = "green"
    wrong_col = "red"

    model_prediction = np.argmax(raw_layer, -1) # [num_exits, num_samples, num_classes]

    # generate bool mask for each exit based on if prediction matches ground truth
    correctness = model_prediction == true_vals

    # construct weighting system where values that are identified as
    # correct earlier are given more weight
    # for model with 2 exits:
    # 0 means it was always misclassified
    # 1 means it was correctly identified at the final exit
    # 2 means it was identified correctly at first then misclassified (overthinking)
    # 3 means it was identified correctly both times
    difficulty = None
    for i, exit_layer in enumerate(correctness):
        weight = 2 ** (num_exits - i - 1)
        exit_layer = exit_layer * weight
        difficulty = exit_layer if difficulty is None else difficulty + exit_layer

    difficulties = [0, 1]  # only misclassifications in the first exit

    # plot the distribution the maximum values of each class
    max_vals = np.max(raw_layer, -1)
    # these are the maxima of the logit values, raw final layer values of chosen class

    fig1, axis1 = plt.subplots(nrows=num_exits, sharex=True)
    fig1.suptitle(f"{title_name} Raw value distribution")

    for e_idx, e_exit in enumerate(max_vals):
        ax = axis1[e_idx]

        # set up x axis histogram bins
        max_val = max(e_exit)
        min_val = min(e_exit)
        bins = 100
        x = np.linspace(min_val, max_val, bins)

        # see graphs.py for more info
        plot_right_wrong(ax, e_exit, correctness[e_idx], x)

        # see graphs.py for more info
        plot_difficulties(
            ax, difficulty, e_exit, x, density=True, difficulties=difficulties
        )

        ax.set_title(f"exit {e_idx}")
        ax.legend(loc="upper left")
    fig1.set_size_inches(20,10)
    fig1.tight_layout()

    # grouping the raw final layer values by their index (aka their class)
    grouped_by_class = group_by_class(raw_layer, correctness)

    # Do a plot of the raw final layer values on a per-class basis
    # overlay with the correct/incorrect kernels for comparison
    # TODO see if we can share the x limits between ex0,1 for better comparison
    fig2, axis2 = plt.subplots(nrows=num_exits, sharex=True)
    fig2.suptitle(f"{title_name} Per class final layer distribution")
    for e, e_exit in enumerate(grouped_by_class):
        ax = axis2[e]

        max_val = max(max_vals[e])
        min_val = min(max_vals[e])

        x = np.linspace(min_val, max_val, 100)

        for class_num, vals in enumerate(e_exit):
            label = f"C{class_num}"
            ax.plot(
                x,
                fit_kernel(vals[:, 0], x, bandwidth=(max_val - min_val) / 30),
                label=label,
                alpha=0.7,
            )
            # ax.hist(vals[:,0], density=True,histtype='step',label=label, bins=20)

        correct = max_vals[e][correctness[e]]
        # do the same right wrong plot on top of the per-class versions
        ax.plot(
            x,
            fit_kernel(correct, x, bandwidth=(max_val - min_val) / 30),
            color=correct_col,
            ls="dashed",
            label="correct avg",
        )
        # apply correctness mask to only return the raw vals that correspond
        # to an incorrect classification
        wrong = max_vals[e][np.invert(correctness[e])]
        ax.plot(
            x,
            fit_kernel(wrong, x, bandwidth=(max_val - min_val) / 30),
            color=wrong_col,
            ls="dashdot",
            label="incorrect avg",
        )

        ax.set_title(f"exit {e}")
        ax.legend(loc="upper left")
    # fig2 is the layer dist per class
    fig2.set_size_inches(20,10)
    fig2.tight_layout()

    # FIXME not the best way to handle it but making lint errors shush
    #fps_cost_ax = None
    auroc_ax = None
    axis = None
    if not plot_classes:
        #fig5, fps_ax = plt.subplots(1, 1)
        #fig5.suptitle("Combined FPS plot, bit messy")
        #fps_cost_ax = fps_ax.twinx()
        fig9, auroc_ax = plt.subplots()
        fig9.suptitle("AUROC")

    print("Running analysis on different confidence functions")
    # ANALYSE VARIOUS SOFTMAX FUNCTIONS
    for idx, function in enumerate(data["conf_metrics"].keys()):
        name = function[5:]
        print(f"Confidence function {idx}: {name}")
        # NOTE temporarily removing conf met selection
        #if funcs is None or idx in funcs: # None defaults to do them all
        sftmx = data["conf_metrics"][function]["raw_softmax"]

        subt = f"[{title_name}] ({name})"
        if not plot_classes:
            fig, axis = plt.subplots(
                ncols=2, nrows=num_exits - 1, squeeze=False, layout="constrained"
            )
            fig.suptitle(f"{subt}")
            fig.set_size_inches(20, 8)

        # don't perform this analysis on final exit as there is no decision to be made
        for exit_num, raw_norm in enumerate(sftmx[:-1]):
            raw_norm = np.array(raw_norm)
            # this sets up an x axis limited to between 0,1
            # makes sense for the softmax variants...
            # BUT NOT for entropy or the subtract version
            logbins = np.linspace(0, 1, 100)
            auroc_thresholds = None
            gt_inequality_bool = None
            if "entropy" in function.lower():
                # I think this old code actually does the entropy calc
                # BUT my version already does it
                #raw_norm = (1 / np.log(num_classes)) * raw_norm
                #raw_norm = 1 - raw_norm
                proc_norm = raw_norm
                logbins = np.linspace(0, max(raw_norm), 100)
                auroc_thresholds = np.linspace(max(raw_norm), 0, 200)
                gt_inequality_bool = False
            elif "sub" in function.lower():
                print(f"Found a sub! {name}")
                # the max value will always be 1 by design...
                # need to use the sum instead I think?
                # since the threshold is 1 > thr * sum
                proc_norm = np.sum(raw_norm, axis=-1)
                logbins = np.linspace(0, max(proc_norm), 100)
                auroc_thresholds = np.linspace(max(proc_norm), 0, 200)
                gt_inequality_bool = False
            elif "doctor" in function.lower():
                print(f"Found the Doctor! {name}")
                proc_norm = raw_norm
                # automatically enlarge x axis if doing (1-g)/g instead of only (1-g)
                if max(raw_norm) > 1 or max(raw_norm) < 0:
                    logbins = np.linspace(min(raw_norm), max(raw_norm), 100)
            else:
                proc_norm = np.max(raw_norm, -1)

            ### TODO separate this out bcos long ting ###
            if plot_classes:
                fig1, _ = make_axes(num_classes, layout="constrained")
                fig2, _ = make_axes(num_classes, layout="constrained")
                fig3, ax3 = plt.subplots()
                fig1.suptitle(f"{subt} Exit {exit_num} Distribution")
                fig2.suptitle(f"{subt} Exit {exit_num} Cost")
                fig3.suptitle(f"{subt} Exit {exit_num} Combined")
                sft_grouped = group_by_1D(
                    np.stack(
                        (
                            # consider only those values which are correctly
                            # classified by last layer
                            np.array(raw_layer[exit_num])[correctness[-1]].argmax(
                                -1
                            ),
                            proc_norm[correctness[-1]],
                            correctness[exit_num][correctness[-1]],
                        ),
                        -1,
                    )
                )
                if gt_inequality_bool is None:
                    # the inequality direction for the top1 style conf metrics
                    gt_inequality_bool = True

                for i, cl in enumerate(sft_grouped):
                    plot_right_wrong(
                        fig1.axes[i],
                        cl[:, 0],
                        np.array(cl[:, 1], dtype=bool),
                        logbins,
                    )

                    # plot per class separately
                    plot_false_positives(
                        logbins,
                        cl[:, 0],
                        np.array(cl[:, 1], dtype=bool),
                        gt_thr=gt_inequality_bool,
                        ax=fig2.axes[i],
                        normalised=True,
                        cax=fig2.axes[i].twinx(),
                    )
                    # combine all per class ones
                    plot_false_positives(
                        logbins,
                        cl[:, 0],
                        np.array(cl[:, 1], dtype=bool),
                        gt_thr=gt_inequality_bool,
                        normalised=True,
                        cax=ax3,
                        label_prefix=f"{i}",
                    )

                    fig1.axes[i].set_title(f"{i}")
                    fig2.axes[i].set_title(f"{i}")
            ### TODO separate this out bcos long ting ###
            else:
                if auroc_thresholds is None:
                    # auroc thresholds for the top1 style conf metrics
                    auroc_thresholds = np.linspace(0, 1, 200)
                if gt_inequality_bool is None:
                    # the inequality direction for the top1 style conf metrics
                    gt_inequality_bool = True
                plot_auroc(
                    auroc_ax,
                    auroc_thresholds,
                    proc_norm,
                    correctness[exit_num],
                    gt_thr=gt_inequality_bool,
                    prefix=f"{name} ",
                )
                auroc_ax.legend()
                # axis for the split plots for each confidence metric (lhs)
                ax = axis[exit_num][0]
                plot_right_wrong(ax, proc_norm, correctness[exit_num], logbins)
                plot_difficulties(
                    ax,
                    difficulty,
                    proc_norm,
                    logbins,
                    density=True,
                    difficulties=difficulties,
                )

                # axis for plotting the confusing fps curve (rhs)
                fp_ax = axis[exit_num][1]
                fp_cost_ax = fp_ax.twinx()

                # keep only those values that are correct at the next exit
                # could be changed to keep only values that are
                # correct at the final exit
                #relative_correctness = correctness[exit_num][
                #    correctness[exit_num + 1]
                #]
                # NOTE relative_correctness not used?

                plot_false_positives(
                    logbins,
                    proc_norm,
                    correctness[exit_num], # just if the MODEL predicts the correct classif
                    gt_thr=gt_inequality_bool,
                    ax=fp_ax,
                    normalised=True,
                    cax=fp_cost_ax,
                )

                # NOTE overly complicated plot...
                #plot_false_positives(
                #    logbins,
                #    proc_norm,
                #    correctness[exit_num],
                #    ax=fps_ax,
                #    label_prefix=name + " ",
                #    cax=fps_cost_ax,
                #)

                ax.set_title(f"exit {exit_num}")
                ax.legend()
    # fig5 idk what it is really
    #fig5.set_size_inches(20,8)
    #fig5.tight_layout()
    # fig9 is the AUROC plot - TODO still checking if its the one I want
    fig9.set_size_inches(12,12)
    fig9.tight_layout()
    # fig is the threshold dist and fps columned plots
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Early Exit Data Analyzer")
    parser.add_argument(
        "-f",
        "--filename",
        required=True,
        help="Path to the data to be analyzed and plotted",
    )
    parser.add_argument(
        "-fc",
        "--functions",
        required=False,
        nargs="+",
        type=int,
        help="Index into the functions to be used in analysis",
    )

    parser.add_argument(
        "-cl",
        "--per_class",
        required=False,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Run the analysis on a per class basis. Warning, will produce many graphs",
    )

    args = parser.parse_args()

    print(f"Analysis on: {args.filename}")
    main(args.filename, funcs=args.functions, plot_classes=args.per_class)
