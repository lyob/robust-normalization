#%% imports
import os
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


#%% import data
results_dir = os.path.join('..', 'code', 'analysis', 'exemplar-manifolds', 'results')
dataset_name = 'MNIST'
model_name = 'ConvNet'
iter_val = 1
random = False
seed = 0

files = [y for x in os.walk(results_dir) for y in glob(os.path.join(x[0], f'*{dataset_name}*'))]
print(files)
df = pd.concat([pd.read_csv(f) for f in files])
print(df['norm_method'])


# %% define measures to print
measures = ['mean_cap', 'dim', 'rad', 'center_corr', 'EVD90', 'PR']
manifold_type = 'exemplar'
eps_val = '6.0'
eps = float(eps_val)/255


#%% plot the data of different norms for a single epsilon value, or a single norm at different epsilon values
def plot_layerwise(df, measures, manifold_type, eps=1/255, norm_method=None):
    # eps or norm_method -- one must have a value
    assert (eps and norm_method) == None, 'One out of `eps` or `norm_method` must be None.'
    assert eps != None or norm_method != None, "Specify the value of either `eps` or `norm_method`."
    
    fig, axes = plt.subplots(nrows=len(measures), ncols=1, figsize=(12,4*len(measures)))

    if eps!=None and norm_method==None:
        for ax, measure in zip(axes, measures):
            # filter the df for data of interest
            data = df[
                (df['eps'].apply(lambda x : np.isclose(x, eps)))&
                (df['manifold_type']==manifold_type)
            ]
                        
            # average over seeds / layers / models
            data = data.groupby(['model', 'layer', 'seed', 'norm_method']).mean().sort_values(by=['layer'])

            ax = sns.lineplot(
                x='layer',
                y=measure,
                hue='norm_method',
                ax=ax,
                ci='sd',
                data=data,
            )
            sns.despine()
        axes[0].set(title=f'eps = {int(eps*255)}/255 = {eps:2f}')
        plt.show()

    elif norm_method!=None and eps==None:
        for ax, measure in zip(axes, measures):
            # filter the df for data of interest
            data = df[
                (df['norm_method']==norm_method)&
                (df['manifold_type']==manifold_type)
            ]

            data = data.groupby(['model', 'layer', 'seed', 'eps']).mean().sort_values(by=['layer'])

            ax = sns.lineplot(
                x='layer',
                y=measure,
                hue='eps',
                ax=ax,
                ci='sd',
                data=data,
            )
            sns.despine()
        axes[0].set(title=f'norm method = {norm_method}')
        plt.show()

plot_layerwise(df, measures, eps=eps, norm_method=None, manifold_type=manifold_type)


# %% plot the data of a single norm at different epsilon values
plot_layerwise(df, measures, norm_method='bn', eps=None, manifold_type=manifold_type)


# %%
