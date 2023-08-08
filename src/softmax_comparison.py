import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mstats
import sys
import argparse
from helper.graphs import *

from matplotlib.pyplot import cm


def DOCTOR_softmax_from_softmax(softmax: np.array):
    # formula from DOCTOR paper
    g = np.square(softmax).sum(-1)

    return g


def DOCTOR_softmax_from_raw(raw_layer: np.array):
    # exp = np.exp(raw_layer)
    exp = np.power(2, raw_layer)

    square_exp = np.power(exp, 2)

    square_sum = np.power(exp.sum(-1), 2)

    g = square_exp.sum(-1) / square_sum

    return g


def raw_distance(raw_layer: np.array):
    max_val = raw_layer.max(-1)
    max_ind = raw_layer.argmax(-1)

    num_classes = raw_layer.shape[-1]
    # breakpoint()
    avg_vals = (raw_layer.sum(-1) - max_val) / (num_classes - 1)
    dist = max_val - avg_vals
    # breakpoint()
    return dist


def raw_distance_daghero(raw_layer: np.array):
    sorted_vals = np.sort(raw_layer, axis=-1)

    max_val = sorted_vals[:, :, -1]
    second_max = sorted_vals[:, :, -2]

    dist = max_val - second_max
    # breakpoint()
    return dist


def plot_false_positives(
    x,
    confidence_layer: np.array,
    correctness: np.array,
    label_prefix="",
    normalised=True,
    ax=None,
    cax=None,
):
    exit_perc = []
    fp_rate = []

    total_num = len(confidence_layer) if normalised else 1
    for thr in x:
        exiting_mask = confidence_layer > thr
        num_exiting = np.invert(exiting_mask).sum()
        # keep track of how many samples exited at this thresh level
        exit_perc.append(num_exiting / total_num)

        # pick out the values that would exit and see how many are wrong
        fp_num = np.invert(correctness[exiting_mask]).sum()
        fp_rate.append(fp_num / total_num)

    if ax is not None:
        line1 = ax.plot(x, exit_perc, label=label_prefix + "E%", ls="dashed")
        ax.plot(x, fp_rate, label=label_prefix + "FP", color=line1[0].get_color())
        ax.legend(fontsize="small")

    if cax is not None:
        cost = np.array(fp_rate) + 0.1 * np.array(exit_perc)
        cax.plot(
            x,
            cost,
            label=label_prefix + "C",
            ls="dashdot",
            color=line1[0].get_color() if ax else None,
        )
        cax.legend(fontsize="small")


def main(json_file, funcs=None, plot_classes=False):
    # json_file = "./rawSoftmax_b_lenet_se_singleThresh_2023-07-19_153525.json"
    # json_file = "./b_lenet_cifar_singleThresh_2023-07-20_172520.json"

    with open(json_file) as json_data:
        data = json.load(json_data)

    title_name = data["model"] + " " + data["dataset"]

    true_vals = data["test_vals"]["true_indices"]
    raw_layer = data["test_vals"]["raw_layer"]

    # doctor = DOCTOR_softmax_from_raw(np.array(raw_layer))

    softmax_values = None
    for func in data["test_vals"]["comps"]:
        if func["name"] == "Softmax":
            softmax_values = func["raw_softmax"]

    # if softmax_values is not None:
    doctor = DOCTOR_softmax_from_softmax(np.array(softmax_values))
    data["test_vals"]["comps"].append({"name": "doctor sfmtx", "raw_softmax": doctor})

    doctor = DOCTOR_softmax_from_raw(np.array(raw_layer))
    data["test_vals"]["comps"].append({"name": "doctor raw", "raw_softmax": doctor})

    # custom_func = raw_distance(np.array(raw_layer))
    # data['test_vals']['comps'].append({
    #   'name':"doctor distance between max and avg",
    #   'raw_softmax':custom_func
    # })

    custom_func = raw_distance_daghero(np.array(softmax_values))
    data["test_vals"]["comps"].append(
        {"name": "doctor distance between top 2", "raw_softmax": custom_func}
    )

    num_exits = data["test_vals"]["num_exits"]
    num_classes = np.array(raw_layer).shape[-1]
    num_compares = len(data["test_vals"]["comps"])

    correct_col = "blue"
    wrong_col = "red"

    # num_exits x num_samples x num_classes
    model_prediction = np.argmax(raw_layer, -1)

    # discern between values that are wrong and right on single exit
    correctness = model_prediction == true_vals

    # construct weighting system where values that are identified as correct earlier are given more weight
    # for model with 2 exits:
    # 0 means it was always misclassified, 1 means it was correctly identified at the final exit
    # 2 means it was identified correactly at first then misclassified (overthinking)
    # 3 means it was identified correctly both times
    difficulty = None
    for i, exit_layer in enumerate(correctness):
        weight = 2 ** (num_exits - i - 1)
        exit_layer = exit_layer * weight
        difficulty = exit_layer if difficulty is None else difficulty + exit_layer

    difficulties = [0, 1]  # only misclassifications in the first exit

    # plot the distribution the maximum values of each class
    max_vals = np.max(raw_layer, -1)

    fig1, axis1 = plt.subplots(nrows=num_exits)

    fig1.suptitle(f"{title_name} Raw value distribution")

    for e, e_exit in enumerate(max_vals):
        ax = axis1[e]

        max_val = max(e_exit)
        min_val = min(e_exit)
        bins = 100

        x = np.linspace(min_val, max_val, bins)

        plot_right_wrong(ax, e_exit, correctness[e], x)

        plot_difficulties(
            ax, difficulty, e_exit, x, density=True, difficulties=difficulties
        )

        ax.set_title(f"exit {e}")
        ax.legend(loc="upper left")

    grouped_by_class = group_by_class(raw_layer, correctness)

    fig2, axis2 = plt.subplots(nrows=num_exits)
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
        ax.plot(
            x,
            fit_kernel(correct, x, bandwidth=(max_val - min_val) / 30),
            color=correct_col,
            ls="dashed",
            label="correct avg",
        )

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

    if not plot_classes:
        fig5, fps_ax = plt.subplots(1, 1)
        fps_cost_ax = fps_ax.twinx()

    print("Running analysis on different confidence functions")
    # ANALISE VARIOUS SOFTMAX FUNCTIONS
    for row, function in enumerate(data["test_vals"]["comps"]):
        name = function["name"]
        print(f"{row} {name}")
        if funcs is None or row in funcs:
            sftmx = function["raw_softmax"]

            if not plot_classes:
                fig, axis = plt.subplots(ncols=2, nrows=num_exits - 1, squeeze=False)
                fig.suptitle(f"{title_name} {row}:{name}")
                fig.set_size_inches(14, 6)

            # don't perform this analysis on the last exit as there is no decision to be made
            for exit_num, softmax in enumerate(sftmx[:-1]):
                softmax = np.array(softmax)
                logbins = np.linspace(0, 1, 100)
                if name == "Entropy":
                    softmax = (1 / np.log(num_classes)) * softmax
                    softmax = 1 - softmax
                elif "doctor" in name:
                    softmax = softmax
                    # automatically enlarge the x axis if doing (1-g)/g instead of only (1-g)
                    if max(softmax) > 1 or max(softmax) < 0:
                        logbins = np.linspace(min(softmax), max(softmax), 100)
                else:
                    softmax = np.max(softmax, -1)

                if plot_classes:
                    fig1, axs = make_axes(num_classes, layout="constrained")
                    fig2, axs = make_axes(num_classes, layout="constrained")
                    fig3, ax3 = plt.subplots()
                    subt = f"[{title_name}]({name} Exit {exit_num})"
                    fig1.suptitle(f"{subt} Distribution")
                    fig2.suptitle(f"{subt} Cost")
                    fig3.suptitle(f"{subt} Combined")
                    sft_grouped = group_by_1D(
                        np.stack(
                            (
                                # consider only those values which are correctly classified by last layer
                                np.array(raw_layer[exit_num])[correctness[-1]].argmax(
                                    -1
                                ),
                                softmax[correctness[-1]],
                                correctness[exit_num][correctness[-1]],
                            ),
                            -1,
                        )
                    )

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
                            ax=fig2.axes[i],
                            normalised=True,
                            cax=fig2.axes[i].twinx(),
                        )
                        # combine all per class ones
                        plot_false_positives(
                            logbins,
                            cl[:, 0],
                            np.array(cl[:, 1], dtype=bool),
                            normalised=True,
                            cax=ax3,
                            label_prefix=f"{i}",
                        )

                        fig1.axes[i].set_title(f"{i}")
                        fig2.axes[i].set_title(f"{i}")

                else:
                    ax = axis[exit_num][0]
                    plot_right_wrong(ax, softmax, correctness[exit_num], logbins)
                    plot_difficulties(
                        ax,
                        difficulty,
                        softmax,
                        logbins,
                        density=True,
                        difficulties=difficulties,
                    )

                    # axis for false positive plotting
                    fp_ax = axis[exit_num][1]
                    fp_cost_ax = fp_ax.twinx()
                    # keep only those values that are correct at the next exit
                    # could be changed to keep only values that are correct at the final exit
                    relative_correctness = correctness[exit_num][
                        correctness[exit_num + 1]
                    ]
                    plot_false_positives(
                        logbins,
                        softmax[correctness[exit_num + 1]],
                        relative_correctness,
                        ax=fp_ax,
                        normalised=True,
                        cax=fp_cost_ax,
                    )

                    plot_false_positives(
                        logbins,
                        softmax[correctness[exit_num + 1]],
                        relative_correctness,
                        ax=fps_ax,
                        label_prefix=name + " ",
                        cax=fps_cost_ax,
                    )

                    ax.set_title(f"exit {exit_num}")
                    ax.legend()
    # plt.tight_layout()
    plt.show()


# fig.set_size_inches(6 * num_exits, 4 * num_compares)

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
        help="Run the analysis on a per class basis. Warning, will produce many graphs",
    )

    args = parser.parse_args()

    print(f"Analysis on: {args.filename}")
    main(args.filename, funcs=args.functions, plot_classes=args.per_class)
