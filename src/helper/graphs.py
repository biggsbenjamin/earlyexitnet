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
