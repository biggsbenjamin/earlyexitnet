import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mstats
import sys

from helper.graphs import *

from matplotlib.pyplot import cm



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
  
  return g

def raw_distance(raw_layer: np.array):
  
  max_val = raw_layer.max(-1)
  max_ind = raw_layer.argmax(-1)
  
  num_classes = raw_layer.shape[-1]
  # breakpoint()
  avg_vals = (raw_layer.sum(-1) - max_val) / (num_classes - 1)
  dist = (max_val - avg_vals)
  # breakpoint()
  return dist

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
  
  custom_func = raw_distance(np.array(raw_layer))
  data['test_vals']['comps'].append({
    'name':"doctor distance",
    'raw_softmax':custom_func
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
  
  fig1, axis1 = plt.subplots(nrows=num_exits)
  
  fig1.suptitle(f"{title_name} Raw value distribution")
  
  for e, e_exit in enumerate(max_vals):
    ax = axis1[e]
    
    max_val = max(e_exit)
    min_val = min(e_exit)
    bins = 100
    
    x = np.linspace(min_val, max_val, bins)
    
    plot_right_wrong(ax, e_exit, correctness[e], x)
    
    plot_difficulties(ax, difficulty, e_exit, x, density=True, difficulties=difficulties)
    
    ax.set_title(f"exit {e}")
    ax.legend(loc='upper left')
  
  grouped_by_class = group_by_class(raw_layer, correctness)
  
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
        if max(softmax) > 1 or max(softmax) < 0:
          logbins = np.linspace(min(softmax), max(softmax), 100)
      else:
        softmax = np.max(softmax, -1)
      
      plot_right_wrong(ax, softmax, correctness[exit_num], logbins)
      
      plot_difficulties(ax, difficulty, softmax, logbins, density=True, difficulties=difficulties)

      
      ax.set_title(f"exit {exit_num}")
      ax.legend()
      
    # fig.set_size_inches(20,15)
  plt.show()

# fig.set_size_inches(6 * num_exits, 4 * num_compares)

if __name__ == "__main__":
  print(f"Analysis on: {sys.argv[1]}")
  main(sys.argv[1])