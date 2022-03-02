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
eps_val = '2.0'
iter_val = 1
random = False
seed = 0
norm_method = 'bn'

files = [y for x in os.walk(results_dir) for y in glob(os.path.join(x[0], f'*{dataset_name}*'))]
df = pd.concat([pd.read_csv(f) for f in files])
print(df['norm_method'])


# %% define measures to print
measures = ['mean_cap', 'dim', 'rad', 'center_corr', 'EVD90', 'PR']
manifold_type = 'exemplar'
eps = float(eps_val)/255


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
        data = data.groupby(['model', 'layer', 'seed', 'norm_method']).mean().sort_values(by=['layer'])
        print(data)

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
