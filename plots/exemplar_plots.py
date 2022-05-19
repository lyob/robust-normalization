#%% imports
%reload_ext autoreload
%autoreload 2
import os
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from art.utils import load_mnist


#%% import data
model_name = 'lenet'
dataset_name = 'mnist'

results_dir = os.path.join('..', 'code', 'analysis', 'exemplar-manifolds', 'results', model_name)
## unused parameters
# iter_val = 1
manifold_type = 'exemplar'
norm_method = 'nn'
eps = 0.1
iter = 1
random = False
seed = 0
img_idx = [0,1,2,3]
img_idx = list(range(50))
num_manifolds = len(img_idx)
plot_img_idx = 1

# select the csv files
NT=2000
seeded=False

# read many files
files = [y for x in os.walk(results_dir) for y in glob(os.path.join(x[0], f'*seeded={seeded}-seed_analysis*-NT={NT}*'))]
df = pd.concat([pd.read_csv(f) for f in files])

# read one file
# if len(img_idx) < 10:
#     file_name = f'model={model_name}-manifold={manifold_type}-norm={norm_method}-eps={eps}-iter={iter}-random={random}-seed={seed}-num_manifolds={num_manifolds}-img_idx={img_idx}-NT={NT}-seeded={seeded}.csv'
# else:
#     file_name = f'model={model_name}-manifold={manifold_type}-norm={norm_method}-eps={eps}-iter={iter}-random={random}-seed={seed}-num_manifolds={num_manifolds}-range={len(img_idx)}-NT={NT}-seeded={seeded}.csv'
# df = pd.read_csv(os.path.join(results_dir, file_name))
# print(df['label'])
    


# %% define measures to print
if type(img_idx) == list:
    measures = ['mean_cap', 'cap', 'dim', 'rad', 'width', 'EVD90', 'PR']
else:
    measures = ['mean_cap', 'dim', 'rad', 'center_corr', 'EVD90', 'PR']
# eps_val = '6.0'
# eps = float(eps_val)/255
# eps = 0.07
eps = 0.1

(_, _), (_, y_test), _, _ = load_mnist()
ylabel = np.argmax(y_test[plot_img_idx])


#%% plot the data of different norms for a single epsilon value, or a single norm at different epsilon values
def plot_layerwise(df, measures, manifold_type, mode, eps=None, norm_method=None, title=None):
    fig, axes = plt.subplots(nrows=len(measures), ncols=1, figsize=(8,2*len(measures)), sharex='all')

    if mode=='eps':
        assert eps!=None, 'Mode is `eps`, specify the value of `eps`.'
        for ax, measure in zip(axes, measures):
            # filter the df for data of interest
            data = df[
                (df['eps'].apply(lambda x : np.isclose(x, eps)))
                &(df['manifold_type']==manifold_type)
                &(df['label']==float(ylabel))
                # &(df['layer']!='0.pixels')
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
        if title==None:
            axes[0].set(title=f'eps = {eps:.2g}')
        else:
            axes[0].set(title=f'eps = {eps:.2g}, {title}')
        fig.tight_layout()
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
        if title==None:
            axes[0].set(title=f'norm method = {norm_method}')
        else:
            axes[0].set(title=f'norm method = {norm_method}, {title}')
            
        plt.show()

#%%
eps = 0.1
if len(img_idx) < 10:
    plot_layerwise(df, measures, manifold_type, mode='eps', eps=eps, title=f'num_manifolds = {num_manifolds} {img_idx}, img_idx = {plot_img_idx}')
else:
    plot_layerwise(df, measures, manifold_type, mode='eps', eps=eps, title=f'num_manifolds = {num_manifolds} (range: 50), img_idx = {plot_img_idx}')

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
