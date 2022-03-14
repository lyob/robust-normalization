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
## unused parameters
# model_name = 'ConvNet'
# iter_val = 1
# random = False
# seed = 0

files = [y for x in os.walk(results_dir) for y in glob(os.path.join(x[0], f'*{dataset_name}*'))]
# print(files)
df = pd.concat([pd.read_csv(f) for f in files])
print(df['adv_accuracy'])


# %% define measures to print
measures = ['mean_cap', 'dim', 'rad', 'center_corr', 'EVD90', 'PR']
manifold_type = 'exemplar'
# eps_val = '6.0'
# eps = float(eps_val)/255
eps = 0.07

#%% plot the data of different norms for a single epsilon value, or a single norm at different epsilon values
def plot_layerwise(df, measures, manifold_type, mode, eps=None, norm_method=None):
    fig, axes = plt.subplots(nrows=len(measures), ncols=1, figsize=(12,4*len(measures)))

    if mode=='eps':
        assert eps!=None, 'Mode is `eps`, specify the value of `eps`.'
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
        # axes[0].set(title=f'eps = {int(eps*255)}/255 = {eps:2f}')
        axes[0].set(title=f'eps = {eps:.2g}')
        plt.show()

    elif mode=='norm_method':
        assert norm_method!=None, 'Mode is `norm_method, specify the value of `norm_method`.'
        for ax, measure in zip(axes, measures):
            # filter the df for data of interest
            data = df[
                (df['norm_method']==norm_method)&
                (df['manifold_type']==manifold_type)&
                (df['eps'].apply(lambda x : x in [0.05, 0.1, 0.15, 0.2] or np.isclose(x, 1/255)))
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
            handles, labels = ax.get_legend_handles_labels()
            new_labels = [f'{float(label):.2g}' for label in labels]
            ax.legend(handles, new_labels)
        axes[0].set(title=f'norm method = {norm_method}')
        plt.show()

#%%
eps = 6/255
plot_layerwise(df, measures, manifold_type, mode='eps', eps=eps)

# %% plot the data of a single norm at different epsilon values
plot_layerwise(df, measures, manifold_type, mode='norm_method', norm_method='nn')


# %% plot capacity against attack strength for many norm methods
def plot_epwise(df, measures, manifold_type, layer):
    fig, axes = plt.subplots(nrows=len(measures), ncols=1, figsize=(12,4*len(measures)))

    for ax, measure in zip(axes, measures):
        # filter the df for data of interest -- here we only want to consider the last layer.
        data = df[
            (df['manifold_type']==manifold_type)&
            (df['layer']==layer)&
            (df['eps']!=0)
        ]
                    
        # average over seeds / layers / models
        data = data.groupby(['model', 'seed', 'norm_method', 'eps']).mean().sort_values(by=['eps'])

        ax = sns.lineplot(
            x='eps',
            y=measure,
            hue='norm_method',
            ax=ax,
            ci='sd',
            data=data,
        )
        sns.despine()
    # axes[0].set(title=f'eps = {int(eps*255)}/255 = {eps:2f}')
    axes[0].set(title=f'layer = {layer}')
    plt.show()

# layer options: ['0.pixels', '1.conv1', '2.norm', '3.relu', '4.conv2', '5.norm', '6.relu', '7.linear']
plot_epwise(df, measures = ['mean_cap', 'dim', 'rad', 'center_corr'], manifold_type=manifold_type, layer='1.conv1')

# %% plot the normalized capacity (adv / clean) for each layer as we increase the attack strength
def plot_normalized_capacity(df, layers, manifold_type):
    fig, axes = plt.subplots(nrows=1, ncols=len(layers), figsize=(5*len(layers), 5))

    for ax, layer in zip(axes, layers):

        # filter the df for data of interest -- here we only want to consider the last layer.
        data = df[
            (df['manifold_type']==manifold_type)&
            (df['layer']==layer)&
            (df['eps'].apply(lambda x: x not in [0.2, 0.15]))
        ]

        a = data.loc[:, ['mean_cap', 'norm_method', 'seed', 'eps']]
        for nm in a['norm_method'].unique():
            for s in a['seed'].unique():
                for eps in a['eps'].unique():
                    adv_cond = (a['eps']==eps) & (a['norm_method']==nm) & (a['seed']==s)
                    clean_cond = (a['eps']==0) & (a['norm_method']==nm) & (a['seed']==s)
                    a.loc[adv_cond, 'mean_cap'] /= a.loc[clean_cond, 'mean_cap']
        # print(a)
                    
        # average over seeds / layers / models
        a = a.groupby(['seed', 'norm_method', 'eps']).mean().sort_values(by=['eps'])
        # print(a)

        ax = sns.lineplot(
            x='eps',
            y='mean_cap',
            hue='norm_method',
            ax=ax,
            ci='sd',
            data=a
        )
        sns.despine()
        ax.set(title=f'layer={layer}')
    # axes[0].set(title=f'eps = {int(eps*255)}/255 = {eps:2f}')
    # axes[0].set(title=f'normalized capacity (adv manif. cap / clean manif. cap)')
    plt.show()

# ['0.pixels', '1.conv1', '2.norm', '3.relu', '4.conv2', '5.norm', '6.relu', '7.linear']
plot_normalized_capacity(df, ['0.pixels', '1.conv1', '2.norm', '3.relu'], manifold_type)
plot_normalized_capacity(df, ['4.conv2', '5.norm', '6.relu', '7.linear'], manifold_type)

# %% plot accuracy vs capacity for the final layer

def move_legend(ax, new_loc, **kws):
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=title, **kws)
    
def acc_vs_cap(df, manifold_type):
    # parameters
    sort_by = 'eps' # 'eps' or 'norm_method'
    norm_method = False # any of the norm methods, or False
    adv_vs_clean = 'adv_accuracy' # 'adv_accuracy' or 'clean_accuracy'

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

    # filter the df for data of interest -- here we only want to consider the last layer
    data = df[
        (df['manifold_type']==manifold_type)&
        (df['layer']=='7.linear')&
        # (df['eps']!=0)
        (df['eps'].apply(lambda x : x in [0.05, 0.1, 0.15, 0.2] or np.isclose(x, 1/255)))
        # &(df['seed']==1)
    ]
    if norm_method:
        data = data[data['norm_method']==norm_method]

    data = data.groupby(['seed', 'norm_method', 'mean_cap', adv_vs_clean, 'eps']).mean().sort_values(by=['mean_cap'])

    # switch the hue between eps and norm method
    ax = sns.scatterplot(
        x='mean_cap',
        y=adv_vs_clean,

        hue=sort_by,

        ax=ax,
        data=data
    )
    sns.despine()

    ax.set(title=f'accuracy vs capacity', xlabel="capacity", ylabel="accuracy")
    # ax.set(title=f'accuracy vs capacity, norm method = {norm_method}', xlabel="capacity", ylabel="accuracy")
    # ax.set(title=f'accuracy vs capacity (clean accuracy)', xlabel="capacity", ylabel="accuracy")

    handles, labels = ax.get_legend_handles_labels()
    if labels[0][0]=="0":
        new_labels = [f'{float(label):.2g}' for label in labels]
        ax.legend(handles, new_labels, title='eps', loc='lower right')
    else:
        ax.legend(handles, labels, title='norm method', loc='lower right')
    # move_legend(ax, "lower right")

    plt.show()

acc_vs_cap(df, manifold_type)


# %%

# %%
