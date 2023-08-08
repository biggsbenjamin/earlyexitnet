import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mstats
from sklearn.neighbors import KernelDensity
from sklearn import metrics
from matplotlib.pyplot import cm
import math


def fit_kernel(data, x_vals, kernel="gaussian", bandwidth=None):
    if bandwidth is None:
        bandwidth = (max(data) - min(data)) / 30

    model = KernelDensity(bandwidth=bandwidth, kernel=kernel)

    model.fit(data.reshape(len(data), 1))
    probs = model.score_samples(x_vals.reshape(len(x_vals), 1))
    return np.exp(probs)


def plot_difficulties(ax, difficulty, layer, bins, difficulties=None, density=False):
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
    right_col="blue",
    wrong_col="red",
    quants=[0.2, 0.5, 0.8],
):
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
            # ax.axvline(qw, 0, color='orange', ls='--', label=f"qw {quants[i]*100:.0f}%: {qw:.02f}")
            # ax.axvline(qc, 0, color='green', ls='--', label=f"qc {quants[i]*100:.0f}%: {qc:.02f}")
            ax.axvline(qw, 0, color="orange", alpha=quants[i], ls="--")
            ax.axvline(qc, 0, color="green", alpha=quants[i], ls="--")


def group_by_class(vals: np.array, correctness=None, classes=None, class_vals=None):
    """take an array and return a list of lists with values separated by class.
    By default the class is the argmax, and the values considered are the correctness and the max_value of the array
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

    if (sqrt - math.floor(sqrt)) > 0:
        return plt.subplots(
            nrows=math.floor(sqrt) + 1, ncols=math.floor(sqrt), **kwargs
        )
    else:
        return plt.subplots(nrows=sqrt, ncols=sqrt, **kwargs)


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


def plot_auroc(
    ax: plt.Axes,
    threshes: np.array,
    vals: np.array,
    correct: np.array,
    prefix="",
    **kwargs,
):
    num_false = np.logical_not(correct).sum()
    num_true = correct.sum()
    roc = []

    for thr in threshes:
        estimate = vals > thr

        # correctly identified positive values (both true)
        num_correct = np.logical_and(estimate, correct).sum()
        # true positive rate
        tpr = num_correct / num_true

        # false positive (true in estimate, false in correct)
        num_false_positive = np.logical_and(estimate, np.logical_not(correct)).sum()
        # false positive rate
        fpr = num_false_positive / num_false

        roc.append([thr, tpr, fpr])
    roc = np.array(roc)

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
