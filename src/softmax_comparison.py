import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mstats
from sklearn.neighbors import KernelDensity
import sys

def fit_kernel(data, x_vals, kernel='gaussian', bandwidth=1):
  model = KernelDensity(bandwidth=bandwidth, kernel=kernel)
  
  model.fit(data.reshape(len(data), 1))
  probs = model.score_samples(x_vals.reshape(len(x_vals), 1))
  return np.exp(probs)

def main(json_file):

  # json_file = "./rawSoftmax_b_lenet_se_singleThresh_2023-07-19_153525.json"
  # json_file = "./b_lenet_cifar_singleThresh_2023-07-20_172520.json"

  with open(json_file) as json_data:
      data = json.load(json_data)

  title_name = data['model'] + " " + data['dataset']
  
  true_vals = data['test_vals']['true_indices']
  raw_layer = data['test_vals']['raw_layer']

  num_exits = data['test_vals']['num_exits']

  num_compares = len(data['test_vals']['comps'])

  correct_col = 'blue'
  wrong_col = 'red'

  # num_exits x num_samples x num_classes
  correctness = np.argmax(raw_layer, -1 ) == true_vals

  # plot the distribution the maximum values of each class
  max_vals = np.max(raw_layer, -1)
  argmax_vals = np.argmax(raw_layer, -1)
  
  
  comb = np.stack((argmax_vals, max_vals, correctness), -1)
  
  fig1, axis1 = plt.subplots(nrows=num_exits)
  
  fig1.suptitle(f"{title_name} Raw value distribution")
  
  for e, e_exit in enumerate(max_vals):
    ax = axis1[e]
    
    max_val = max(e_exit)
    min_val = min(e_exit)
    bins = 40
    
    x = np.linspace(min_val, max_val, 100)
    
    correct = e_exit[correctness[e]]    
    ax.plot(x, fit_kernel(correct, x, bandwidth=(max_val - min_val)/30), color=correct_col)
    ax.hist(correct, bins=bins, density=True, histtype='step', label="correct", color=correct_col)
    
    wrong = e_exit[np.invert(correctness[e])]    
    ax.plot(x, fit_kernel(wrong, x, bandwidth=(max_val - min_val)/30), color=wrong_col)
    ax.hist(wrong, bins=bins, density=True, histtype='step', label="wrong", color=wrong_col)
    
    ax.set_title(f"exit {e}")
    ax.legend(loc='upper left')
  
  sorted_raw = np.asarray([c[c[:,0].argsort()] for c in comb])
  
  grouped_by_class = [np.split(s[:,1:], np.unique(s[:,0], return_index=True)[1][1:]) for s in sorted_raw]
  
  fig2, axis2 = plt.subplots(nrows=num_exits)
  fig2.suptitle(f"{title_name} Per class final layer distribution")
  for e, e_exit in enumerate(grouped_by_class):
    ax = axis2[e]
    
    max_val = max(max_vals[e])
    min_val = min(max_vals[e])
    
    x = np.linspace(min_val, max_val, 100)
    
    for class_num, vals in enumerate(e_exit):
      label = f'C{class_num}'      
      ax.plot(x, fit_kernel(vals[:,0], x, bandwidth=(max_val - min_val)/30),label=label, alpha=0.7)
      # ax.hist(vals[:,0], density=True,histtype='step',label=label, bins=20)
    
    correct = max_vals[e][correctness[e]]    
    ax.plot(x, fit_kernel(correct, x, bandwidth=(max_val - min_val)/30), color=correct_col, ls="dashed", label="correct avg")
    
    wrong = max_vals[e][np.invert(correctness[e])]    
    ax.plot(x, fit_kernel(wrong, x, bandwidth=(max_val - min_val)/30), color=wrong_col, ls="dashdot", label="incorrect avg")
    
    ax.set_title(f"exit {e}")
    ax.legend(loc='upper left')



  # ANALISE VARIOUS SOFTMAX FUNCTIONS
  for row, function in enumerate(data['test_vals']['comps']):
    
    sftmx = function['raw_softmax']
    name = function['name']
    
    fig, axis = plt.subplots(nrows=num_exits)
    
    fig.suptitle(f"{title_name} {name}")
    
    for exit_num, softmax in enumerate(sftmx):
      
      ax = axis[exit_num]
      softmax = np.array(softmax)
      # separate the maximum values for the correct and incorrect
      
      if name != "Entropy":
        softmax = np.max(softmax, -1)
      else:
        softmax = (1 / np.log(10)) * softmax
      
      # breakpoint()
      
      correct_vals = softmax[correctness[exit_num]]
      wrong_vals = softmax[np.invert(correctness)[exit_num]]
      
      
      
      # logbins = np.logspace(np.log10(0.1),np.log10(1),100)
      logbins =  np.linspace(0, 1, 100)
      
      model_c = KernelDensity(bandwidth=0.02, kernel='gaussian')
      model_w = KernelDensity(bandwidth=0.02, kernel='gaussian')
    
      model_c.fit(correct_vals.reshape(len(correct_vals), 1))
      model_w.fit(wrong_vals.reshape(len(wrong_vals), 1))
      
      probs_c = model_c.score_samples(logbins.reshape(len(logbins), 1))
      probs_c = np.exp(probs_c)
      probs_w = model_w.score_samples(logbins.reshape(len(logbins), 1))
      probs_w = np.exp(probs_w)
      
      ax.plot(logbins, probs_c, color=correct_col, ls='--')
      ax.plot(logbins, probs_w, color=wrong_col, ls='--')
      
      # if name == "Base-2 Sub-Softmax":
      #   logbins = np.logspace(np.log10(0.1), np.log10(1), 64)
      #   # logbins = np.linspace(0,1,32)
        
      quants = [0.25, 0.5, 0.75]
      quantiles = mstats.mquantiles(wrong_vals, prob=quants)
      for i, q in enumerate(quantiles):
        ax.axvline(q, 0, color='orange', ls='--', label=f"q {quants[i]*100:.0f}%: {q:.02f}")
      
      ax.hist(correct_vals, bins=logbins,histtype='step',label=f'correct {correct_vals.shape[0]}',density=True, color=correct_col)
      ax.hist(wrong_vals, bins=logbins, histtype='step', label=f'incorrect {wrong_vals.shape[0]}',density=True, color=wrong_col)
      
      # plot hists side by side
      # ax.hist([correct_vals, wrong_vals], bins=logbins, label=[f'correct {correct_vals.shape[0]}', f'incorrect {wrong_vals.shape[0]}'])

      # if name != "Base-2 Sub-Softmax":
      # ax.set_xscale('log')
      # ax.set_yscale('log')
      ax.set_xlabel('threshold value')
      
      ax.set_ylabel('count')
      ax.set_title(f"exit {exit_num}")
      ax.legend(loc='upper left')
      
    fig.set_size_inches(20,15)
  plt.show()

# fig.set_size_inches(6 * num_exits, 4 * num_compares)

if __name__ == "__main__":
  print(f"Analysis on: {sys.argv[1]}")
  main(sys.argv[1])