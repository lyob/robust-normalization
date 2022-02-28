#%% imports
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

# import data
load_path = os.path.join('..', 'results', 'mftma', 'cifar', 'eps-0_v1')
normalize = ["lrnb", "lrns", "gn", "ln", "nn", "lrnc", "in", "bn"]
# normalize = ['bn']

# form dict for metric
norms = {}
for n in normalize:
    metric_file = f'metrics_{n}.pkl'
    load_file = os.path.join(load_path, metric_file)
    
    metrics = {}
    with open(load_file, 'rb') as f:
        metrics = pickle.load(f)
        # print(metrics)

    norms[n] = metrics

print(norms['bn']['capacities'].keys())

# %% plot

fig, ax = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
for n in normalize:
    l1 = ax[0,0].plot(list(norms[n]['capacities'].keys()), list(norms[n]['capacities'].values()), label=n)
    ax[0,0].set(title='capacities')
    ax[1,0].plot(list(norms[n]['radii'].keys()), list(norms[n]['radii'].values()), label=n)
    ax[1,0].set(title='radii', ylim=([0, 15]))
    ax[0,1].plot(list(norms[n]['dimensions'].keys()), list(norms[n]['dimensions'].values()), label=n)
    ax[0,1].set(title='dimensions')
    ax[1,1].plot(list(norms[n]['correlations'].keys()), list(norms[n]['correlations'].values()), label=n)
    ax[1,1].set(title='correlations')
fig.legend([l1], labels=normalize, loc=2, bbox_to_anchor=(0.55, 0.78, 0, 0))
ax[1,0].tick_params(axis='x', rotation=90)
ax[1,1].tick_params(axis='x', rotation=90)
fig.suptitle('ResNet, Cifar-10, eps=2/255')
fig.tight_layout()

# %%
