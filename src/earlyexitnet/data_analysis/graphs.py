"""
Helper functions for graphing softmax-based (and other) confidence metrics.

Making things pretty with kernels to fit histogram distributions.
Plotting the \"difficulty\" of a given input by how well the early and late
classifiers do. Plotting right and wrong using the power of mOrAlS...
These plots just capture the various values the processed (or raw final layer)
values disaggregated between correct and incorrect classifications.

"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mstats
from sklearn.neighbors import KernelDensity
from sklearn import metrics
from matplotlib.pyplot import cm

def fit_kernel(data, x_vals, kernel="gaussian", bandwidth=None):
    """
    Get a density (y coords) estimation for a bunch of points (x coords)
    using the kernel shape (gaussian), sized by the bandwidth param.
    Acts to smooth out the histogram plot with a nice looking density func (probability?).
    """
    # NOTE bw must be 0 < bw < inf
    if bandwidth is None or bandwidth == 0.0:
        bandwidth = (max(data) - min(data)) / 30
        if bandwidth == 0.0:
            print(f"WARNING: bandwidth param is {bandwidth} so using 0.001 instead.")
            print(f"max:{max(data)} min:{min(data)}")
            bandwidth = 0.001 #FIXME just guessing at a good value

    model = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    model.fit(data.reshape(len(data), 1))
    probs = model.score_samples(x_vals.reshape(len(x_vals), 1))
    return np.exp(probs)


def plot_difficulties(ax, difficulty, layer, bins, difficulties=None, density=False):
    """
    `difficulty` is a measure of how much the network struggles with classification.

    Construct weighting system where values that are identified as
    correct earlier are given more weight (BB NOTE: not sure...)
    for model with 2 exits:
    0 means it was always misclassified
    1 means it was correctly identified at the final exit
    2 means it was identified correctly at first then misclassified (overthinking)
    3 means it was identified correctly both times

    I think a high degree of overlap between these density functions indicates that
    the classification model has trouble separating misclassifs at e0&e1 and
    misclassifs only @ e0.
    Basically, it's an uphill battle for any confidence metric from the raw vals alone.
    """
    vals = []
    labels = []

    num_vals = max(difficulty) + 1 if difficulties is None else len(difficulties)

    colours = cm.viridis(np.linspace(0, 1, num_vals))

    for i in range(max(difficulty) + 1) if difficulties is None else difficulties:
        index = difficulty == i
        extracted_val = layer[index]

        max_val = max(extracted_val)
        min_val = min(extracted_val)

        vals.append(extracted_val)
        labels.append(f"d{i} {index.sum()}")
        ax.plot(
            bins,
            # fit kernel just makes a prettier graph
            fit_kernel(extracted_val, bins, bandwidth=(max_val - min_val) / 30),
            color=colours[i],
        )

    ax.hist(
        vals,
        bins=bins,
        density=density,
        histtype="barstacked",
        label=labels,
        alpha=0.4,
        color=colours,
    )


def plot_hist_kernel(ax, vals, xax=None, col=None, label=None, hist=True, ls="--"):
    if xax is None:
        xax = np.linspace(min(vals), max(vals), 100)
    if len(vals) > 0:
        ax.plot(xax, fit_kernel(vals, xax), color=col, label=label, ls=ls)
        if hist:
            ax.hist(vals, bins=xax, histtype="step", density=True, color=col)


def plot_right_wrong(
    ax,
    values,
    correctness,
    xax=None,
    right_col="green",
    wrong_col="red",
    quants=[0.25, 0.5, 0.75],
):
    """
    Discriminate between the data values that are associated with a correct classification
    and an incorrect (wrong) classification.
    Plot the histogram and prettified kernel for both sets of values.

    I think a high degree of overlap between these density functions indicates that
    the classification model has trouble separating right from wrong lol.

    Basically, it's an uphill battle for any confidence metric from the raw vals alone.
    """
    if xax is None:
        xax = np.linspace(min(values), max(values), 100)
    correct_vals = values[correctness]
    wrong_vals = values[np.invert(correctness)]

    plot_hist_kernel(
        ax, correct_vals, xax, right_col, f"correct {correct_vals.shape[0]}"
    )
    plot_hist_kernel(ax, wrong_vals, xax, wrong_col, f"incorrect {wrong_vals.shape[0]}")

    ax.set_xlabel("threshold value")
    ax.set_ylabel("density")

    if quants is not None:
        quantiles_w = mstats.mquantiles(wrong_vals, prob=quants)
        quantiles_c = mstats.mquantiles(correct_vals, prob=quants)

        for i, (qw, qc) in enumerate(zip(quantiles_w, quantiles_c)):
            ax.axvline(qw, 0, color="red", alpha=quants[i], ls="--",
                       label=f"{quants[i]*100:.0f}%: {qw:.02f}")
            ax.axvline(qc, 0, color="green", alpha=quants[i], ls="--",
                       label=f"{quants[i]*100:.0f}%: {qc:.02f}")


def group_by_class(vals: np.ndarray, correctness=None, classes=None, class_vals=None):
    """
    take an array and return a list of lists with values separated by class.
    By default the class is the argmax, and the values considered are the
    correctness and the max_value of the array
    """
    if class_vals is None:
        class_vals = np.max(vals, -1)

    if classes is None:
        classes = np.argmax(vals, -1)

    if correctness is None:
        comb = np.stack((classes, class_vals), -1)
    else:
        comb = np.stack((classes, class_vals, correctness), -1)

    return group_by(comb)


def group_by(comb):
    """uses the first index of the innermost array as the label by which to group.
    Expects a 2D array where the grouping is performed on the inner most dimension
    """
    sorted_raw = np.asarray([c[c[:, 0].argsort()] for c in comb])
    return [
        np.split(s[:, 1:], np.unique(s[:, 0], return_index=True)[1][1:])
        for s in sorted_raw
    ]


def group_by_1D(comb):
    """uses the first index of the innermost array as the label by which to group.
    Expects a 1D array where the grouping is performed on the inner most dimension
    """
    sort = comb[comb[:, 0].argsort()]
    return np.split(sort[:, 1:], (np.unique(sort[:, 0], return_index=True)[1][1:] - 1))


def make_axes(num_classes: int, **kwargs):
    sqrt = math.sqrt(num_classes)

    # check if the number of classes is NOT a square number I guess?
    if (sqrt - math.floor(sqrt)) > 0:
        return plt.subplots(
            nrows=math.floor(sqrt) + 1, ncols=math.floor(sqrt), **kwargs
        )
    else:
        return plt.subplots(nrows=int(sqrt), ncols=int(sqrt), **kwargs)


def plot_false_positives(
    x,
    confidence_layer: np.ndarray,
    correctness: np.ndarray,
    gt_thr: bool, # dictates the direction of the threshold inequality
    label_prefix="",
    normalised=True,
    ax=None,
    cax=None,
):
    exit_perc = []
    fp_rate = []

    total_num = len(confidence_layer) if normalised else 1
    for thr in x:
        if gt_thr:
            exiting_mask = confidence_layer > thr
        else:
            exiting_mask = confidence_layer < thr
        num_exiting = np.invert(exiting_mask).sum()
        # keep track of how many samples exited at this thresh level
        exit_perc.append(num_exiting / total_num)

        # pick out the values that would exit and see how many are wrong
        fp_num = np.invert(correctness[exiting_mask]).sum()
        fp_rate.append(fp_num / total_num)

    if ax is not None:
        line1 = ax.plot(x, exit_perc, label=label_prefix + "Exit %", ls="dashed")
        ax.plot(x, fp_rate, label=label_prefix + "FP", color=line1[0].get_color())
        ax.set_xlabel("Threshold Value")
        ax.legend(loc='center left', fontsize="small")

    if cax is not None:
        # some fairly arbitrary looking "cost" metric??
        # false positive rate + scaled percentage of exits
        # NOTE would make sense if this could be linked to some real HW cost for
        # the difference between early exiting and not
        cost = np.array(fp_rate) + 0.1 * np.array(exit_perc)
        cax.plot(
            x,
            cost,
            label=label_prefix + "Cost:FPR + 0.1(exit%)",
            ls="dashdot",
            color=line1[0].get_color() if ax else None,
        )
        cax.legend(loc='center right', fontsize="small")


def plot_auroc(
    ax: plt.Axes,
    threshes: np.ndarray,
    vals: np.ndarray,
    correct: np.ndarray,
    gt_thr: bool, # dictates the direction of the threshold inequality
    prefix="",
    **kwargs,
):
    """
    What is AUROC?
    https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5

    Area Under the Curve (AUC)
    Receiver Operating Charateristics (ROC)
    Area Under the Receiver Operating Characteristics (AUROC)

    True +ve rate (recall):         TP/(TP + FN)
    Specificity:                    TN/(TN + FP)
    False +ve rate (1-specificity): FP/(TN + FP)

    AUROC plot:
        y axis : T+ve rate
        x axis : T-ve rate

    Perfect model has AUC = 1, perfectly separate T-ves and T+ves
    Realistic good model AUC >= 0.7, able to separate T-ves and T+ves
    Useless model AUC = 0.5, can't separate T-ves and T+ves

    Other useful metrics:
    Precision:  TP/(TP + FP)
    """
    # get the number of incorrect classifications
    num_false = np.logical_not(correct).sum()
    # get the number of correct classifications
    num_true = correct.sum()
    roc = np.zeros((len(threshes), 3,))

    for i, thr in enumerate(threshes):
        if gt_thr: # greater than threshold
            estimate = vals > thr
        else:
            estimate = vals < thr

        # correctly identified positive values (both true)
        num_correct = np.logical_and(estimate, correct).sum()
        # true positive rate
        tpr = num_correct / num_true

        # false positive (true in estimate, false in correct)
        num_false_positive = np.logical_and(estimate, np.logical_not(correct)).sum()
        # false positive rate
        fpr = num_false_positive / num_false

        roc[i][0] = thr
        roc[i][1] = tpr
        roc[i][2] = fpr

    auroc = metrics.auc(roc[:, 2], roc[:, 1])
    # fpr vs tpr
    ax.plot(
        roc[:, 2],
        roc[:, 1],
        label=f"{prefix}AUROC:{auroc:0.4f}",
        **kwargs,
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
