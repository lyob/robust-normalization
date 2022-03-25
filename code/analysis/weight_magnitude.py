#%% load modules
import numpy as np
# import pickle
import torch
import torch.nn as nn

import os
import sys
sys.path.insert(0,'..')
os.chdir('..')

from mnist_layer_norm import Net
from vonenet.vonenet import VOneNet

#%% parameters
# model_name = 'standard'
model_name = 'vone_convnet-layer1_norm'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1
norm_method = 'nn'
lr = 0.01
wd = 0.0005

simple_channels = 16
complex_channels = 16
ksize = 5

os.chdir('../results')
print(os.path.abspath('.'))

#%% load data
if model_name == 'standard':
    save_folder = os.path.join('mnist_regularize')

    conv_1 = nn.Conv2d(in_channels=1, out_channels=simple_channels+complex_channels, kernel_size=ksize, stride=2, padding=ksize//2)
    model = Net(conv_1, simple_channels + complex_channels, normalize=norm_method)
if model_name == 'vone_convnet-layer1_norm':
    save_folder = os.path.join('vone_frontend')
    
    model = VOneNet(simple_channels=simple_channels, complex_channels=complex_channels, norm_method=norm_method)

save_path = os.path.join(save_folder, 'trained_models', model_name)
save_name = os.path.join(save_path, f'{model_name}-lr_{str(lr)}-wd_{str(wd)}-seed_{str(seed)}-normalize_{norm_method}.pth')

model.load_state_dict(torch.load(save_name, map_location=device))
model.eval()


#%% look at weight magnitudes

print(model)
print(model.vone_block)
print(model.model)
print(model.model.conv_2.weight.shape)

# %%
