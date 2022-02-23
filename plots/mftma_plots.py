#%% imports
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

# import data
load_path = os.path.join('..', 'results', 'mftma', 'cifar')
normalize = ["lrnb", "lrns", "gn", "ln", "nn", "lrnc", "in", "bn"]
# normalize = ['bn']

# form dict for metric
norms = {}
for n in normalize:
    metric_file = f'metrics_{n}.pkl'
    load_file = os.path.join(load_path, metric_file)
    
    metrics = {}
    with open(load_file, 'rb') as f:
        out = pickle.load(f)
        metrics['capacities'] = out[0]
        metrics['radii'] = out[1]
        metrics['dimensions'] = out[2]
        metrics['correlations'] = out[3]

    norms[n] = metrics

print(list(metrics['capacities'].keys()))
print(norms['bn']['capacities'].keys())

# %% plot

fig, ax = plt.subplots(4, 1, figsize=(6, 15), sharex=True)
for n in normalize:
    ax[0].plot(list(norms[n]['capacities'].keys()), list(norms[n]['capacities'].values()), label=n)
    ax[0].set(title='capacities')
    ax[1].plot(list(norms[n]['radii'].keys()), list(norms[n]['radii'].values()), label=n)
    ax[1].set(title='radii')
    ax[2].plot(list(norms[n]['dimensions'].keys()), list(norms[n]['dimensions'].values()), label=n)
    ax[2].set(title='dimensions')
    ax[3].plot(list(norms[n]['correlations'].keys()), list(norms[n]['correlations'].values()), label=n)
    ax[3].set(title='correlations')
ax[3].tick_params(axis='x', rotation=90)


# %%
