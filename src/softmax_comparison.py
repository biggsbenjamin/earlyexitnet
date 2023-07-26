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

# def DOCTOR_softmax_from_raw(raw_layer: np.array()):
  
  
def DOCTOR_softmax_from_softmax(softmax: np.array):
  # formula from DOCTOR paper
  g = np.square(softmax).sum(-1)
  
  return (1 - g)

def DOCTOR_softmax_from_raw(raw_layer: np.array):
  
  # exp = np.exp(raw_layer)
  exp = np.power(2, raw_layer)
  
  square_exp = np.power(exp, 2)
  
  square_sum = np.power(exp.sum(-1), 2)
  
  g = (square_exp.sum(-1) / square_sum)
  
  return (1 - g)

def plot_difficulties(ax, difficulty, layer, bins,difficulties=None, density=False):
  vals = []
  labels = []
  for i in (range(max(difficulty) + 1) if difficulties is None else difficulties):
    vals.append(layer[difficulty == i])
    labels.append(f"d{i} {(difficulty == i).sum()}")
    
  ax.hist(vals, bins=bins, density=density, histtype='barstacked', label=labels, alpha=0.4)

def main(json_file):

  # json_file = "./rawSoftmax_b_lenet_se_singleThresh_2023-07-19_153525.json"
  # json_file = "./b_lenet_cifar_singleThresh_2023-07-20_172520.json"

  with open(json_file) as json_data:
      data = json.load(json_data)

  title_name = data['model'] + " " + data['dataset']
  
  true_vals = data['test_vals']['true_indices']
  raw_layer = data['test_vals']['raw_layer']


  # doctor = DOCTOR_softmax_from_raw(np.array(raw_layer))
  
  softmax_values = None
  for func in data['test_vals']['comps']:
    if func['name'] == "Softmax":
      softmax_values = func['raw_softmax']
  
  
  # if softmax_values is not None:
  doctor = DOCTOR_softmax_from_softmax(np.array(softmax_values))
  data['test_vals']['comps'].append({
    'name':"doctor sfmtx",
    'raw_softmax':doctor
  })
  
  doctor = DOCTOR_softmax_from_raw(np.array(raw_layer))
  data['test_vals']['comps'].append({
    'name':"doctor raw",
    'raw_softmax':doctor
  })
  
  num_exits = data['test_vals']['num_exits']

  num_compares = len(data['test_vals']['comps'])

  correct_col = 'blue'
  wrong_col = 'red'

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
  
  difficulties = [0,1] # only misclassifications in the first exit

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
    
    plot_difficulties(ax, difficulty, e_exit, bins, density=True, difficulties=difficulties)
    
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
      
      logbins =  np.linspace(0, 1, 100)
      
      if name == "Entropy":
        softmax = (1 / np.log(10)) * softmax
      elif 'doctor' in name:
        softmax = softmax
        # automatically enlarge the x axis if doing (1-g)/g instead of only (1-g)
        if max(softmax) > 1:
          logbins = np.linspace(0, max(softmax), 100)
      else:
        softmax = np.max(softmax, -1)
      
      
      correct_vals = softmax[correctness[exit_num]]
      wrong_vals = softmax[np.invert(correctness)[exit_num]]
      
      
      
      # logbins = np.logspace(np.log10(0.1),np.log10(1),100)
      
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
        
      quants = [0.2, 0.5, 0.8]
      quantiles_w = mstats.mquantiles(wrong_vals, prob=quants)
      quantiles_c = mstats.mquantiles(correct_vals, prob=quants)
      
      for i, (qw, qc) in enumerate(zip(quantiles_w, quantiles_c)):
        # ax.axvline(qw, 0, color='orange', ls='--', label=f"qw {quants[i]*100:.0f}%: {qw:.02f}")
        # ax.axvline(qc, 0, color='green', ls='--', label=f"qc {quants[i]*100:.0f}%: {qc:.02f}")
        ax.axvline(qw, 0, color='orange',alpha=quants[i], ls='--')
        ax.axvline(qc, 0, color='green',alpha=quants[i], ls='--')
      
      ax.hist(correct_vals, bins=logbins,histtype='step',label=f'correct {correct_vals.shape[0]}',density=True, color=correct_col)
      ax.hist(wrong_vals, bins=logbins, histtype='step', label=f'incorrect {wrong_vals.shape[0]}',density=True, color=wrong_col)
      
      plot_difficulties(ax, difficulty, softmax, logbins, density=True, difficulties=difficulties)

      
      # plot hists side by side
      # ax.hist([correct_vals, wrong_vals], bins=logbins, label=[f'correct {correct_vals.shape[0]}', f'incorrect {wrong_vals.shape[0]}'])

      # if name != "Base-2 Sub-Softmax":
      # ax.set_xscale('log')
      # ax.set_yscale('log')
      ax.set_xlabel('threshold value')
      
      ax.set_ylabel('density')
      ax.set_title(f"exit {exit_num}")
      ax.legend()
      
    # fig.set_size_inches(20,15)
  plt.show()

# fig.set_size_inches(6 * num_exits, 4 * num_compares)

if __name__ == "__main__":
  print(f"Analysis on: {sys.argv[1]}")
  main(sys.argv[1])