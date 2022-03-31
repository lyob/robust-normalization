#%% load modules
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import os
import sys
sys.path.insert(0,'..')
os.chdir('..')

from mnist_layer_norm import Net
from vonenet.vonenet import VOneNet

os.chdir('../results')
print(os.path.abspath('.'))

#%% parameters
# model_name = 'standard'
model_name = 'convnet'
frontend = 'vone_filterbank'  # learned_conv or vone_filterbank
frontend = 'learned_conv'  # learned_conv or vone_filterbank
norm_position = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1
norm_method = 'in'
lr = 0.01
wd = 0.0005

simple_channels = 16
complex_channels = 16
ksize = 5

#%% load data
if model_name == 'convnet':
    save_folder = os.path.join(model_name)

    if frontend=='learned_conv':
        conv_1 = nn.Conv2d(in_channels=1, out_channels=simple_channels+complex_channels, kernel_size=ksize, stride=2, padding=ksize//2)
        model = Net(conv_1, simple_channels + complex_channels, normalize=norm_method, norm_position=norm_position)
    elif frontend=='vone_filterbank':
        model = VOneNet(simple_channels=simple_channels, complex_channels=complex_channels, norm_method=norm_method, norm_position=norm_position)

model_folder_name = f'{frontend}_frontend-norm_{norm_position}'
save_path = os.path.join(save_folder, 'trained_models', model_folder_name)
save_name = os.path.join(save_path, f'{model_name}-lr_{str(lr)}-wd_{str(wd)}-seed_{str(seed)}-normalize_{norm_method}.pth')
model.load_state_dict(torch.load(save_name, map_location=device))
model.eval()


#%% extract model frontend (layer 1) filters
if frontend == 'learned_conv':
    filters = model.conv_1
    print('first filter layer:', filters)
    weights = filters.weight

elif frontend == 'vone_filterbank':
    filters = model.vone_block.simple_conv_q0
    print('first filter layer:', filters)
    weights = filters.weight

print('weight tensor shape:', weights.shape)
f_min, f_max = weights.min(), weights.max()

# normalize the weight values to 0-1 so we can visualize them
weights = (weights - f_min) / (f_max - f_min)
weights = weights.detach().numpy()


#%% visualize filters
# weights.shape = [output_ch, input_ch, filter_axis_1, filter_axis_2]

n_filters = 32  # out of 32
n_input_ch = weights.shape[1]

if n_input_ch == 1 and n_filters == 32:
    fig, axes = plt.subplots(4,8, figsize=(10, 7))

    for i in range(n_filters):
        # get the filter
        f = weights[i, :,:,:]

        ax = axes[i//8, i%8]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(f[0,:,:], cmap='gray')
    
else: 
    fig, axes = plt.subplots(n_filters, n_input_ch, figsize=(3,10))
    
    for i in range(n_filters):
        # get the filter
        f = weights[i, :,:,:]
        # plot each input channel separately
        for j in range(n_input_ch):
            ax = axes[i] if n_input_ch ==1 else axes[i,j]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(f[j,:,:], cmap='gray')
            ax.set(ylabel=i+1)
    
    fig.supylabel('filters')

fig.tight_layout()
fig.suptitle(f'layer 1 filters for {frontend} frontend\n norm method: {norm_method}', y=1.03)
fig.set(facecolor='white')

save_name = os.path.join('..', 'plots', 'figures', 'convnet-filter-visualization', f'filters-{frontend}_frontend-norm_{norm_position}-method_{norm_method}.png')
plt.savefig(save_name, dpi=200, facecolor='white', bbox_inches='tight', transparent=False)


#%%
# the backend
# print(model.model)
# print(model.model.conv_2)



#%% calculate weight strengths


#%% 
