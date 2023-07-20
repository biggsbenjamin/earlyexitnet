import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mstats

json_file = "./rawSoftmax_b_lenet_se_singleThresh_2023-07-19_153525.json"

with open(json_file) as json_data:
    data = json.load(json_data)


true_vals = data['test_vals']['true_indices']

num_exits = data['test_vals']['num_exits']

num_compares = len(data['test_vals']['comps'])

# num_exits x num_samples x num_classes


# print(axis)

for row, function in enumerate(data['test_vals']['comps']):
  
  sftmx = function['raw_softmax']
  name = function['name']
  fig, axis = plt.subplots(nrows=num_exits, sharey=True)
  
  fig.suptitle(name)
  
  for exit_num, softmax in enumerate(sftmx):
    
    ax = axis[exit_num]
    
    # discern between correctly classified values and not  
    correctness = np.argmax(softmax, -1) == true_vals

    # separate the maximum values for the correct and incorrect
    softmax = np.max(softmax, -1)
    correct_vals = softmax[correctness]
    wrong_vals = softmax[np.invert(correctness)]
    
    logbins = np.logspace(np.log10(0.1),np.log10(1),200)
    
    if name == "Base-2 Sub-Softmax":
      logbins = np.logspace(np.log10(0.1), np.log10(1), 32)
       
    quants = [0.25, 0.5, 0.75]
    quantiles = mstats.mquantiles(wrong_vals, prob=quants)
    for i, q in enumerate(quantiles):
      ax.axvline(q, 0, color='b', ls='--', label=f"q {quants[i]*100:.0f}%: {q:.02f}")
    
    ax.hist(correct_vals, bins=logbins, stacked=True,label=f'correct {correct_vals.shape[0]}')
    ax.hist(wrong_vals, bins=logbins, stacked=True, label=f'incorrect {wrong_vals.shape[0]}')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('threshold value')
    ax.set_ylabel('count')
    ax.set_title(f"exit {exit_num}")
    ax.legend()

# fig.set_size_inches(6 * num_exits, 4 * num_compares)
plt.show()