#%%
import os
import sys
import pickle
from glob import glob

import pandas as pd
import seaborn as sns
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# code_root = os.path.abspath('..')
code_root = os.path.abspath('/Users/blyo/Documents/research/chung-lab/robust-normalization/code/analysis')
os.chdir(code_root)
print('the root directory is', os.path.abspath('.'))


#%% import data from mftma and plot for one image
# datafile parameters
seeded = False
NT = 2000
M = 50

ma_data_dir = os.path.join('.', 'exemplar-manifolds', 'results', 'lenet')
ma_files = [y for x in os.walk(ma_data_dir) for y in glob(os.path.join(x[0], f'*seeded={seeded}-seed_analysis*-NT={NT}*'))]
ma_df = pd.concat([pd.read_csv(f) for f in ma_files])
ma_df = ma_df.rename(columns={'img_idx': 'image index'})

# data selection parameters
img_idx = 1
measures = ['normalized width']
norm_method = 'nn'
# and these ones just in case:
eps = 0.1
manifold_type = 'exemplar'
label = 7

# select the appropriate data
data = ma_df[
    (ma_df['eps'].apply(lambda x : np.isclose(x, eps)))
    &(ma_df['manifold_type']==manifold_type)    
    &(ma_df['norm_method']==norm_method)
]

# find width of layer 0 (input layer) and put it into a separate column
for i in range(M):
    width_of_layer0_for_one_img = data[(data['image index']==float(i))&(data['layer']=='0.pixels')]['width'].mean()
    # normalize each layer width by the width of layer 0
    data.loc[data['image index']==float(i), 'normalized width'] = data[(data['image index']==float(i))]['width'].div(width_of_layer0_for_one_img)
    
# now plot what the width looks like for one image
data_1 = data[data['image index']==img_idx]
# width_of_layer0 = data[data['layer']=='0.pixels']['width'].mean()
# remove layer 0 from dataframe (so we don't plot it)
data_1 = data_1[data_1['layer']!='0.pixels']
# get the names of the layers
layer_names = data_1['layer'].unique()
# join the columns you want and get the average
data_1 = data_1.groupby(['layer', 'model', 'seed', 'norm_method', 'image index']).mean().sort_values(by=['layer'])
# data['normalized width'] = data['width'].div(width_of_layer0)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax = sns.lineplot(
    x = 'layer',
    y = measures[0],
    hue = 'image index',
    ax = ax,
    ci = 'sd',
    data = data_1
)
sns.despine()
fig.tight_layout()


#%% import data from FIM and plot for one image
# fim datafile parameters
fim_load_seed = 1
norm_method = 'nn'

fim_data_dir = os.path.join('.', 'fisher-info', 'saved-metrics', 'convnet4')
fim_filename = f'per-img-metrics-seed={fim_load_seed}-norm_method={norm_method}-num_images={M}.pkl'

file = open(os.path.join(fim_data_dir, fim_filename), 'rb')
fim_data = pickle.load(file)
(metrics, eigvals, eigvecs) = fim_data
metric_labels = list(metrics[0]['1.conv1'].keys())

# data selection parameters
# metric_labels = ['logdet', 'sum', 'max', 'pr', 'npr']


fig, ax = plt.subplots(3, 2, figsize=(10, 6))
for idx, m in enumerate(metric_labels):
    # layer l
    per_metric_data = [metrics[img_idx][l][m] for l in layer_names]
    ax[idx//2, idx%2].plot(layer_names, per_metric_data)
    ax[idx//2, idx%2].set(title=f'{m}')
fig.suptitle(f'FIM metrics for image {img_idx}')
fig.tight_layout()


# put data into dataframe
image_range = range(M)

fim_df = []
for layer_name in layer_names:
    m_logdet = []
    m_sum = []
    m_max = []
    m_pr = []
    m_npr = []
    for i in image_range:
        m_logdet.append(metrics[i][layer_name]['logdet'])
        m_max.append(metrics[i][layer_name]['max'])
        m_sum.append(metrics[i][layer_name]['sum'])
        m_pr.append(metrics[i][layer_name]['pr'])
        m_npr.append(metrics[i][layer_name]['npr'])

    df = pd.DataFrame(
        columns=[*metric_labels, 'image index'], 
        data = np.array([
            m_logdet,
            m_sum,
            m_max,
            m_pr,
            m_npr,
            image_range
        ]).T
    )
    df['layer'] = np.repeat([layer_name], len(image_range), axis=0)
    
    fim_df.append(df)
fim_df = pd.concat(fim_df, ignore_index=True)
# fim_df = fim_df['image index'].astype(int)


# %% merge the two dataframes using their common columns

both_df = pd.merge(data, fim_df, on=['layer', 'image index'])

#%% compare the two metrics for a couple of images, 1 layer, 1 FIM metric

def plot_relationship(both_df, selected_layer, selected_metric, image_range):
    # condition the data on a single layer and a single metric 
    data_conditioned = both_df[
        (both_df['layer']==selected_layer)
        &(both_df['eps'].apply(lambda x : np.isclose(x, eps)))
        &(both_df['manifold_type']==manifold_type)
        &(both_df['norm_method']==norm_method)
        &(both_df['image index'].apply(lambda x : x in image_range))
    ]

    data_conditioned_means = data_conditioned.groupby([selected_metric]).mean()
    data_conditioned_stdev = data_conditioned.groupby([selected_metric]).std()['normalized width'].agg(list)

    # palette = sns.color_palette('crest', 50, as_cmap=False)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax = sns.scatterplot(
        x = selected_metric,
        y = 'normalized width',
        hue = 'image index',
        # palette = palette,
        # ci = 'sd',
        data = data_conditioned_means
    )

    # show stdev for each collection of runs (for one image, one metric)
    points = np.array(ax.collections[0].get_offsets())
    # print(points.T)
    x_coords = points.T[0]
    y_coords = points.T[1]
        
    ax.errorbar(
        x = x_coords,
        y = y_coords,
        yerr = data_conditioned_stdev,
        fmt=' ',
        # ecolor = palette,
        # ecolor = cmap
    )
    fig.suptitle(f'layer: {selected_layer}, FIM metric: {selected_metric}')
    fig.tight_layout()
    
    # save the figure
    
    

# selected intersection
# selected_layer = layer_names[3]
# print(selected_layer)
# selected_metric = 'logdet'  # logdet, sum, max, pr, npr 
# selected_metric = metric_labels[0]
image_range = range(50)
save_dir = os.path.join('.')

for selected_layer in layer_names:
    for selected_metric in metric_labels:
        plot_relationship(both_df, selected_layer, selected_metric, image_range)

# %%
