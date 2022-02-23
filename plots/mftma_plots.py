#%% imports
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

# import data
load_path = os.path.join('..', 'results', 'mftma', 'cifar')
normalize = ["lrnb", "lrns", "gn", "ln", "nn", "lrnc", "in", "bn"]
normalize = ['bn', 'nn']

# form dict for metric
norms = dict()
for n in normalize:
    metric_file = f'metrics_{n}.pkl'
    load_file = os.path.join(load_path, metric_file)
    
    metrics = dict()
    with open(load_file, 'rb') as f:
        out = pickle.load(f)
        metrics['capacities'] = out[0]
        metrics['radii'] = out[1]
        metrics['dimensions'] = out[2]
        metrics['correlations'] = out[3]

        print(metrics['capacities'])

        for idx, m 
            metrics
            
    norms[n] = metrics



# %%
