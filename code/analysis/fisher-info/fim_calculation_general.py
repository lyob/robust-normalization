#%%
%reload_ext autoreload
%autoreload 2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
import os

from fim_utils import Eigendistortion, NthLayer
from art.utils import load_mnist

# code_root = os.path.abspath('/mnt/ceph/users/blyo1/syLab/robust-normalization/code/')
code_root = os.path.abspath('../..')
os.chdir(code_root)
print('the root directory is', os.path.abspath('.'))

# %% import the model I want to analyze

# load the model
def load_model(model_name, norm_method):
    if model_name[:7] == 'convnet':
        from mnist_layer_norm import Net_both, Net_1, Net_2
        simple_channels = 16
        complex_channels = 16
        ksize = 5

        if frontend=='learned_conv':
            conv_1 = nn.Conv2d(in_channels=1, out_channels=simple_channels+complex_channels, kernel_size=ksize, stride=2, padding=ksize//2)
        if norm_position=='both':
            lenet = Net_both(conv_1, simple_channels + complex_channels, normalize=norm_method)
        
        return lenet

# load model weights into the model
def load_model_weights(model, norm_method):
    model_folder_name = f'{frontend}_frontend-norm_{norm_position}'
    load_folder = os.path.join('..', 'results', model_name, 'trained_models', model_folder_name)
    load_name = os.path.join(load_folder, f'{model_name}-lr_{lr}-wd_{wd}-seed_{seed}-normalize_{norm_method}.pth')
    model.load_state_dict(torch.load(load_name, map_location=device))
    model.eval()
    return model

# select an image to feed into the model
def select_input_img(dataset, idx):
    if dataset == 'mnist':
        (_, _), (x_test, y_test), _, _ = load_mnist()
        input_img_view = x_test[idx].squeeze(2)
        input_img = torch.Tensor(np.swapaxes(x_test, 1, 3)[idx]).unsqueeze(0)

    print('this is the image we will feed into the network:')
    plt.imshow(input_img_view)
    plt.colorbar()
    plt.show()
    return input_img

def calc_metrics(eig_vals, inverse=False):
    ev_logdet = torch.sum(torch.log(eig_vals[:400]))
    ev_sum = torch.sum(eig_vals)
    # PR
    ev_sum_sq = torch.sum(eig_vals)**2
    ev_sq_sum = torch.sum(eig_vals**2)
    if inverse:
        pr = ev_sq_sum / ev_sum_sq
    pr = ev_sum_sq / ev_sq_sum
    npr = pr / len(eig_vals)
    return ev_logdet, ev_sum, pr, npr

# calculate the FIM of the model wrt the input image, and return a measure of model sensitivity
def calc_model_fim(model_name, norm_method, model, input_img, layers, layer_names, show_plot=True):    
    metrics = {}
    eigvecs = {}
    if show_plot:
        fig, ax = plt.subplots(1, 1, sharey='all')
    for idx, l in enumerate(layers):
        print(f'layer {l}')
        metrics_per_layer = {}

        # select the layer        
        layer = NthLayer(model, norm_method, layer=l)

        # find the eigendistortions for the model
        ed_lenet = Eigendistortion(input_img, layer)

        # synthesize the eigenvalues and eigenvectors
        eigvec, eigval, eigind = ed_lenet.synthesize()
        max_eigval = eigval[0]
        print(f'max eigenvalue is {max_eigval}')

        # resize the eigvecs to a matrix shape
        eigvec = eigvec.view(eigvec.size()[0], -1)
        eigvecs[l] = eigvec
       
        if show_plot: 
            # plot the eigendistortions
            ax.plot(eigval, '.', label=f'layer {l}: {layer_names[l]}')
            model_name = '3 layer LeNet' if model_name[:7]=='convnet' else model_name
            ax.set(title=f'Eigenvalues for {model_name}, nm={norm_method}', xlabel='Eigenvector index', ylabel='Eigenvalue')
            ax.legend()
            fig.tight_layout()

        # calculate the volume of sensitivity and its proxies
        ev_logdet, ev_sum, pr, npr = calc_metrics(eigval)

        # # calculate the volume of the sensitivity via the log determinant
        # ev_logdet = torch.sum(torch.log(eigval[:400]))
        # print(f'log determinant of the FIM is {ev_logdet}')

        # ev_sum = torch.sum(eigval)
        # print(f'sum of the eigenvalues is {ev_sum}')

        metrics_per_layer['logdet'] = ev_logdet
        metrics_per_layer['sum'] = ev_sum
        metrics_per_layer['max'] = max_eigval
        metrics_per_layer['pr'] = pr
        metrics_per_layer['npr'] = npr


        metrics[l] = metrics_per_layer
    return metrics, eigval, eigvecs

#################################################################################################################################
#################################################################################################################################

#%% define parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parameters
lenet_parameters = {
    'model_name' : 'convnet4',
    'frontend' : 'learned_conv',  # learned_conv or vone_filterbank or frozen_conv
    'norm_position' : 'both',
    'dataset': 'mnist',
    'seed' : 1,
    'norm_methods' : ['ln', 'in', 'bn', 'gn', 'nn', 'lrnb', 'lrnc', 'lrns'],
    # 'norm_method' : ['nn'],
    'lr' : 0.01,
    'wd' : 0.005,
    'layers' : [0,1,2,3,4,5,6],
    'layer_names': ['conv1', 'nm1', 'relu1', 'conv2', 'nm2', 'relu2', 'fc1']
}

parameters = lenet_parameters

model_name = parameters.get('model_name')
frontend = parameters.get('frontend')
norm_methods = parameters.get('norm_methods')
dataset = parameters.get('dataset')
seed = parameters.get('seed')
norm_position = parameters.get('norm_position')
lr = parameters.get('lr')
wd = parameters.get('wd')
layers = parameters.get('layers')
layer_names = parameters.get('layer_names')

#%% get metrics for one image, many norm methods
# select and display input image
img_num = 0
input_img = select_input_img(dataset, img_num)
def get_metrics_per_image(model_name, norm_methods, input_img, layers, layer_names):
    metrics = {}
    eigvals = {}
    eigvecs = {}
    for nm in norm_methods:
        print(f'norm method is {nm}\n')
        model = load_model(model_name, nm)
        model_with_weights = load_model_weights(model, nm)
        metrics_per_nm, eigvals_per_nm, eigvecs_per_nm = calc_model_fim(model_name, nm, model_with_weights, input_img, layers, layer_names, show_plot=False)
        metrics[nm] = metrics_per_nm
        eigvals[nm] = eigvals_per_nm
        eigvecs[nm] = eigvecs_per_nm
        print(f'------------------------\n')
    return (metrics, eigvals, eigvecs)
        
metrics_per_img = get_metrics_per_image(model_name, norm_methods, input_img, layers, layer_names)
(metrics, eigvals, eigvecs) = metrics_per_img

# save the metrics
model_save_name = 'lenet' if model_name[:7]=='convnet' else model_name
metric_save_dir = os.path.join('.', 'analysis', 'fisher-info', 'saved-metrics', model_name)
if not os.path.exists(metric_save_dir):
    os.makedirs(metric_save_dir, exist_ok=True)
metric_save_name = os.path.join(metric_save_dir, f'per-nm-metrics-seed={seed}-img_num={img_num}.pkl')
pickle.dump(metrics_per_img, open(metric_save_name,'wb'))


#%% get metrics for many images, one norm method
norm_method = 'nn'
def get_metrics_per_nm(model_name, norm_method, dataset, layers, layer_names):
    metrics = {}
    eigvals = {}
    eigvecs = {}
    for img_idx in range(6):
        input_img = select_input_img(dataset, img_idx)
        model = load_model(model_name, norm_method)
        model_with_weights = load_model_weights(model, norm_method)
        metrics_per_img, eigvals_per_img, eigvecs_per_img = calc_model_fim(model_name, norm_method, model_with_weights, input_img, layers, layer_names, show_plot=False)
        metrics[img_idx] = metrics_per_img
        eigvals[img_idx] = eigvals_per_img
        eigvecs[img_idx] = eigvecs_per_img
    return (metrics, eigvals, eigvecs)

metrics_per_nm = get_metrics_per_nm(model_name, norm_method, dataset, layers, layer_names)
(metrics, eigvals, eigvecs) = metrics_per_nm

# save the metrics
model_save_name = 'lenet' if model_name[:7]=='convnet' else model_name
metric_save_dir = os.path.join('.', 'analysis', 'fisher-info', 'saved-metrics', model_name)
if not os.path.exists(metric_save_dir):
    os.makedirs(metric_save_dir, exist_ok=True)
metric_save_name = os.path.join(metric_save_dir, f'per-img-metrics-seed={seed}-norm_method={norm_method}.pkl')
pickle.dump(metrics_per_nm, open(metric_save_name, 'wb'))




# %% plot how the FIM changed between norm methods (use the per-one-img method above ^^)
colorlist = {
    "bn": (0.21607792, 0.39736958, 0.61948028),
    "gn": (0.20344718, 0.56074869, 0.65649508),
    "in": (0.25187832, 0.71827158, 0.67872193),
    "ln": (0.54578602, 0.8544913, 0.69848331),
    "lrnc":(1.0, 0.7686274509803922, 0.0),
    "lrns":(0.9882352941176471, 0.5529411764705883, 0.3843137254901961),
    "lrnb":(0.7634747047461135, 0.3348456555528834, 0.225892295531744),
    "nn":(0., 0., 0.)
}

def plot_metrics(metric, title, sharey='all'):
    fig, ax = plt.subplots(len(layers), 1, figsize=(8,10), sharey=sharey, sharex='all')  # sharey = 'all' or 'none'
    for nm in norm_method:
        for idx, l in enumerate(layers):
            # layer l
            barlist = ax[idx].bar(nm, metrics[nm][l][metric].numpy())
            ax[idx].set(title=f'layer {l}: {layer_names[l]}')
            barlist[0].set_color(colorlist[nm])
    fig.suptitle(title)
    fig.tight_layout()
    # fig.show()

sharey='all'  # 'all' or 'none'
plot_metrics('ev_sum', 'sum of eigenvalues', sharey)
plot_metrics('max_ev', 'max eigenvalue', sharey)
plot_metrics('ev_logdet', 'logdet of eigenvalues (largest 400 only)', sharey)
plot_metrics('pr', 'PR of eigenvalues', sharey)
plot_metrics('npr', 'normalized PR of eigenvalues', sharey)

# %% calculate the change in sensitivity, relative to no norm
# load the saved metric
metric_seed = 1
metric_img_idx = 3

metric_load_dir = os.path.join('.', 'analysis', 'fisher-info', 'saved-metrics', model_name)
metric_load_file = f'metrics-seed={metric_seed}-img_num={metric_img_idx}.pkl'
file = open(os.path.join(metric_load_dir, metric_load_file), 'rb')
metrics = loaded_metric = pickle.load(file)

# calculate change in chosen metric, relative to no norm
def calc_relative_metric_change(metric, model_name, suptitle, type='subtraction'):
    if model_name[:7] == 'convnet':
        delta_layer1 = {}
        delta_layer2 = {}
        for nm in norm_method:
            if type=='subtraction':
                delta_layer1[nm] = loaded_metric[nm][1][metric] - loaded_metric[nm][0][metric]
                delta_layer2[nm] = loaded_metric[nm][4][metric] - loaded_metric[nm][3][metric]
            elif type=='division':
                delta_layer1[nm] = loaded_metric[nm][1][metric] / loaded_metric[nm][0][metric]
                delta_layer2[nm] = loaded_metric[nm][4][metric] / loaded_metric[nm][3][metric]
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        for nm in norm_method:
            barlist1 = ax[0].bar(nm, delta_layer1[nm])
            barlist2 = ax[1].bar(nm, delta_layer2[nm])
            barlist1[0].set_color(colorlist[nm])
            barlist2[0].set_color(colorlist[nm])
        ax[0].set(title='first normalization layer')
        # ax[1].set(title='second normalization layer', ylim=(-1000, 4000))
        ax[1].set(title='second normalization layer')
        fig.suptitle(f'$\Delta$ in {suptitle} via {type}')
        # fig.show()

# comparisons = ['subtraction', 'division']
comparisons = ['division']
for c in comparisons:
    calc_relative_metric_change('ev_sum', model_name, 'sum of eigenvalues', c)
    calc_relative_metric_change('max_ev', model_name, 'maximum eigenvalue', c)
    calc_relative_metric_change('ev_logdet', model_name, 'log determinant', c)
    calc_relative_metric_change('pr', model_name, 'pr', c)
    calc_relative_metric_change('npr', model_name, 'npr', c)


# %% plot how the metrics change between layers
selected_norm_method = 'nn'
metric_labels = ['ev_sum', 'max_ev', 'ev_logdet', 'pr', 'npr']

def plot_metrics_layerwise(metrics, metric_labels, norm_method, layers, layer_names, title, sharey='all'):
    fig, ax = plt.subplots(len(metric_labels), 1, figsize=(8,10), sharex='all')  # sharey = 'all' or 'none'
    for idx, m in enumerate(metric_labels):
        # layer l
        per_metric_data = [metrics[norm_method][l][m].item() for l in layers]
        print(per_metric_data)
        ax[idx].plot(layer_names, per_metric_data)
        ax[idx].set(title=f'{m}')
        # barlist[0].set_color(colorlist[nm])
    fig.suptitle(title)
    fig.tight_layout()
    
plot_metrics_layerwise(metrics, metric_labels, selected_norm_method, layers, layer_names, title=f'norm: {selected_norm_method}, img_idx: {metric_img_idx}')

plt.plot()


# %%
