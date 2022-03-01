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
eps_val = '0.03137254901960784'
iter_val = 1
random = False
seed = 0
file_name = f'model_{dataset_name}_{model_name}-manifold_exemplar_eps_{eps_val}-iter_{iter_val}-random_{random}-seed_{seed}'
# 'model_MNIST_ConvNet-manifold_exemplar-eps_0.03137254901960784-iter_1-random_False-seed_0'

files = [y for x in os.walk(results_dir) for y in glob(os.path.join(x[0], f'*{dataset_name}*'))]
df = pd.concat([pd.read_csv(f) for f in files])
print(df.head(3))


# %% define measures to print
measures = ['mean_cap', 'dim', 'rad', 'center_corr', 'EVD90', 'PR']
manifold_type = 'exemplar'
eps = 8/255


#%% plot the data
def plot_layerwise(df, measures, eps, manifold_type):
    fig, axes = plt.subplots(nrows=len(measures), ncols=1, figsize=(12,4*len(measures)))

    for ax, measure in zip(axes, measures):
        # filter the df for data of interest
        data = df[
            (df['eps'].apply(lambda x : np.isclose(x, eps)))&
            (df['manifold_type']==manifold_type)
        ]

        # average over seeds / layers / models
        data = data.groupby(['model', 'layer', 'seed']).mean().sort_values(by=['layer'])

        ax = sns.lineplot(
            x='layer',
            y=measure,
            hue='norm_method',
            ax=ax,
            ci='sd',
            data=data,
        )
        sns.despine()
    plt.show()

plot_layerwise(df, measures, eps=eps, manifold_type=manifold_type)
# %%
