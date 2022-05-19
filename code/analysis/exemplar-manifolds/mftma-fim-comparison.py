#%%
import os
import sys
import pickle
from glob import glob

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

# code_root = os.path.abspath('..')
code_root = os.path.abspath('/Users/blyo/Documents/research/chung-lab/robust-normalization/code/analysis')
os.chdir(code_root)
print('the root directory is', os.path.abspath('.'))


#%% import data from FIM
# fim load parameters
fim_load_seed = 1
norm_method = 'nn'

fim_data_dir = os.path.join('.', 'fisher-info', 'saved-metrics', 'convnet4')
fim_filename = f'per-img-metrics-seed={fim_load_seed}-norm_method={norm_method}.pkl'

file = open(os.path.join(fim_data_dir, fim_filename), 'rb')
fim_data = pickle.load(file)
(metrics, eigvals, eigvecs) = fim_data


#%% import data from mftma
# mftma load parameters
seeded = False
NT = 2000

ma_data_dir = os.path.join('.', 'exemplar-manifolds', 'results', 'lenet')
ma_files = [y for x in os.walk(ma_data_dir) for y in glob(os.path.join(x[0], f'*seeded={seeded}-seed_analysis*-NT={NT}*'))]
ma_df = pd.concat([pd.read_csv(f) for f in ma_files])

# %% calculate the layerwise ratio of widths
# parameters
eps = 0.1
measures = ['width']
manifold_type = 'exemplar'
label = 7
img_idx = 0
norm_method = 'nn'

data = ma_df[
    (ma_df['eps'].apply(lambda x : np.isclose(x, eps)))
    &(ma_df['manifold_type']==manifold_type)
    &(ma_df['label']==float(label))
    &(ma_df['Unnamed: 0']==img_idx)
    &(ma_df['norm_method']==norm_method)
]

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

# %%
