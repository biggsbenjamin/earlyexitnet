import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mstats
from sklearn.neighbors import KernelDensity

from matplotlib.pyplot import cm


def fit_kernel(data, x_vals, kernel='gaussian', bandwidth=None):
  
  if bandwidth is None:
    bandwidth = (max(data) - min(data))/30
  
  model = KernelDensity(bandwidth=bandwidth, kernel=kernel)
  
  model.fit(data.reshape(len(data), 1))
  probs = model.score_samples(x_vals.reshape(len(x_vals), 1))
  return np.exp(probs)

def plot_difficulties(ax, difficulty, layer, bins,difficulties=None, density=False):
  vals = []
  labels = []
  
  num_vals = max(difficulty) + 1 if difficulties is None else len(difficulties)
  
  colours = cm.viridis(np.linspace(0, 1, num_vals))
  
  for i in (range(max(difficulty) + 1) if difficulties is None else difficulties):
    index = difficulty == i
    extracted_val = layer[index]
    
    max_val = max(extracted_val)
    min_val = min(extracted_val)
    
    vals.append(extracted_val)
    labels.append(f"d{i} {index.sum()}")
    ax.plot(bins, fit_kernel(extracted_val, bins, bandwidth=(max_val - min_val)/30), color=colours[i])

    
  ax.hist(vals, bins=bins, density=density, histtype='barstacked', label=labels, alpha=0.4, color = colours)

def plot_hist_kernel(ax, xax, vals, col=None, label=None):
  ax.plot(xax, fit_kernel(vals, xax), color=col, ls='--')
  ax.hist(vals, bins=xax,histtype='step',label=label,density=True, color=col)


def plot_right_wrong(ax, values, correctness, xax=None, right_col='blue', wrong_col='red',quants = [0.2, 0.5, 0.8]):
    if xax is None:
      xax = np.linspace(min(values), max(values), 100)
    correct_vals = values[correctness]
    wrong_vals = values[np.invert(correctness)]

    plot_hist_kernel(ax, xax, correct_vals, right_col, f'correct {correct_vals.shape[0]}')
    plot_hist_kernel(ax, xax, wrong_vals, wrong_col, f'incorrect {wrong_vals.shape[0]}')
    
    ax.set_xlabel('threshold value')      
    ax.set_ylabel('density')
    
    if quants is not None:
      quantiles_w = mstats.mquantiles(wrong_vals, prob=quants)
      quantiles_c = mstats.mquantiles(correct_vals, prob=quants)
        
      for i, (qw, qc) in enumerate(zip(quantiles_w, quantiles_c)):
        # ax.axvline(qw, 0, color='orange', ls='--', label=f"qw {quants[i]*100:.0f}%: {qw:.02f}")
        # ax.axvline(qc, 0, color='green', ls='--', label=f"qc {quants[i]*100:.0f}%: {qc:.02f}")
        ax.axvline(qw, 0, color='orange',alpha=quants[i], ls='--')
        ax.axvline(qc, 0, color='green',alpha=quants[i], ls='--')
        
        
# take an array and return a list of lists with values separated by class
# by default the class is the argmax, and the values considered are the correctness and the max_value of the array
def group_by_class(vals: np.array, correctness=None, classes=None, class_vals=None):
  if class_vals is None:
    class_vals = np.max(vals, -1)
    
  if classes is None:
    classes = np.argmax(vals, -1)  
  
  if correctness is None:
    comb = np.stack((classes, class_vals), -1)
  else:    
    comb = np.stack((classes, class_vals, correctness), -1)
  
  return group_by(comb)

# uses the first index of the innermost array as the label by which to group
def group_by(comb):
  sorted_raw = np.asarray([c[c[:,0].argsort()] for c in comb])
  return [np.split(s[:,1:], np.unique(s[:,0], return_index=True)[1][1:]) for s in sorted_raw]