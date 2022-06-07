#%% import packages
%reload_ext autoreload
%autoreload 2

import os
import sys
from glob import glob
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from art.utils import load_mnist

#%% set up parameters 
seed = 0

## manifolds analysis parameters
manifold_type = 'exemplar' # 'class' for traditional label based manifolds, 'exemplar' for individual exemplar manifolds
img_idx = [0, 1, 2, 3]  # can be list of ints or False
img_idx = list(range(50))
P = len(img_idx) # number of manifolds, i.e. the number of images 
M = 50 # number of examples per manifold, i.e. the number of images that lie in an epsilon ball around the image  
N = 2000 # maximum number of features to use ??

# determine the type of adversarial examples to use for constructing the manifolds
eps = 0.1  # 8/255, 0
max_iter = 1
eps_step_factor = 1
eps_step = eps / eps_step_factor

random = False  # adversarial perturbation if false, random perturbation if true

#%% seed everything
def seed_everything(seed):
    #initiate seed to try to make the result reproducible 
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
seed_everything(seed)

#%% Load train and test dataset, and show an example. 
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
plt.imshow(x_test[3])

x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)

x_train.shape, x_test.shape


#%% define the MNIST model
os.chdir('/Users/blyo/Documents/research/chung-lab/robust-normalization/code')
from mnist_layer_norm import Net_both
os.chdir('./analysis/exemplar-manifolds')
from helpers import load_model, art_wrap_model, accuracy

# model and dataset details
## regular ResNet18 ('CIFAR_ResNet18') or VOneResNet18 with Gaussian Noise ('CIFAR_VOneResNet18')
cifar_parameters = {
    'model_load_name': 'resnet',
    'model_name': 'CIFAR_ResNet18',
    'dataset': 'CIFAR',
    'normalize': 'nn',
    'model_seed': 1,
    'lr': 0.01,
    'wd': 0.0005
}
lenet_parameters = {
    'model_load_name': 'convnet4',
    'model_name': 'lenet',
    'dataset': 'mnist',
    'normalize': 'nn',
    'frontend': 'learned_conv',
    'model_seed': 1,
    'lr': 0.01,
    'wd': 0.005
}
parameters = lenet_parameters


model_load_name = parameters.get('model_load_name')
model_name = parameters.get('model_name')
dataset = parameters.get('dataset')
normalize = parameters.get('normalize')
frontend = parameters.get('frontend')
model_seed = parameters.get('model_seed')
lr = parameters.get('lr')
wd = parameters.get('wd')

# define the leNet model
simple_channels = 16
complex_channels = 16
ksize = 5
conv_1 = nn.Conv2d(in_channels=1, out_channels=simple_channels+complex_channels, kernel_size=ksize, stride=2, padding=ksize//2)
model = Net_both(conv_1, simple_channels + complex_channels, normalize=normalize)

#%% load model state and dataset
global device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# load the model weights
load_dir = os.path.join('..', '..', '..')
model_path = os.path.join(load_dir, 'results', model_load_name, 'trained_models', f'{frontend}_frontend-norm_both', f'{model_load_name}-lr_{lr}-wd_{wd}-seed_{model_seed}-normalize_{normalize}.pth')
model.load_state_dict(torch.load(model_path, map_location=device))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

# wrap the model with adversarial robustness toolbox, for generating adversarial stimuli.
from art.estimators.classification import PyTorchClassifier
classifier = PyTorchClassifier(
    device_type=device, 
    model=model, 
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28), 
    nb_classes=10,
)


#%% get model clean accuracy
# n_images = 10000
# x_test_sample = x_test[:n_images]
predictions = classifier.predict(x_test)
clean_accuracy = accuracy(predictions, y_test)
print("Accuracy on benign test examples: {}%".format(clean_accuracy * 100))


#%% generate adversarial manifold stimuli and DISPLAY first image
%autoreload 2
from helpers import perturb_stimuli, construct_manifold_stimuli

# either generate new adversarial dataset, or load a pre-generated perturbed dataset
generate_new_adv_dataset = False

if generate_new_adv_dataset:

    # preprocessing step before perturbing
    X, Y = construct_manifold_stimuli(x_test, y_test, manifold_type, P=P, M=M, img_choice=img_idx)
    print(X.shape)
    print(type(X))

    # perturb the stimuli
    seed_everything(seed)  # without this the perturbations are different with every run
    if (eps == 0):
        print('eps is 0')
        X_adv = X
    else:
        print(f'eps is {eps}')
        X_adv = perturb_stimuli(
            X, 
            Y, 
            classifier, 
            eps=eps, 
            eps_step_factor=eps_step_factor, 
            max_iter=max_iter, 
            random=random
        )

    print(f'stimuli shape: {X_adv.shape}')
    print(type(X_adv))

    # display one of the images:
    print('the type of img_idx is', type(img_idx))

    if type(img_idx) == list:
        img_display_idx = M*0
    if img_idx == False:
        img_display_idx = 0
    print(f'displaying the number {np.argmax(Y[img_display_idx])}')

    fig, ax = plt.subplots(3, 1, figsize=(5, 12))
    ax[0].imshow(np.swapaxes(X[img_display_idx].squeeze(), 0, 1))
    ax[0].set(title=f'benign input')
    ax[1].imshow(np.swapaxes(X_adv[img_display_idx].squeeze(), 0, 1))
    ax[1].set(title=f'adv input (1st)')
    ax[2].imshow(np.swapaxes(X_adv[img_display_idx+1].squeeze(), 0, 1))
    ax[2].set(title=f'adv input (2nd)')
    plt.tight_layout()

    # save the perturbed image dataset
    data_to_save = {'X_adv': X_adv, 'X': X, 'Y': Y, 'img_idx': img_idx}
    save_dir = 'adversarial_dataset'
    if len(img_idx) < 10:
        save_name = f'adversarial_dataset-P={P}-M={M}-N={N}-img_idx={img_idx}-run=1.pkl'
    else:
        save_name = f'adversarial_dataset-P={P}-M={M}-N={N}-range={len(img_idx)}-run=1.pkl'
    
    save_file = open(os.path.join(save_dir, save_name), 'wb')
    pickle.dump(data_to_save, save_file)
    save_file.close()
else:
    # load saved pre-generated dataset
    load_dir = os.path.join('adversarial_dataset')
    if len(img_idx) < 10:
        load_file = f'adversarial_dataset-P={P}-M={M}-N={N}-img_idx={img_idx}-run=1.pkl'
    else:
        load_file = f'adversarial_dataset-P={P}-M={M}-N={N}-range={len(img_idx)}-run=1.pkl'
    file = open(os.path.join(load_dir, load_file), 'rb')
    loaded_data = pickle.load(file)
    
    X_adv = loaded_data['X_adv']
    X = loaded_data['X']
    Y = loaded_data['Y']
    img_idx = loaded_data['img_idx']
    
# get adversarial accuracy
adv_accuracy = accuracy(classifier.predict(X_adv), Y)
print(f"Accuracy on adversarial test examples: {adv_accuracy * 100}")
    
Y_adv = np.argmax(Y, axis=1)
Y_adv = Y_adv[[i*M for i in range(len(img_idx))]]

#%% extract activations
%autoreload 2
from helpers import Hook, model_layer_map, MFTMA_analyze_activations

# apply hooks to the model, to extract intermediate representations
hooks = {}

for layer_name, module in model_layer_map(model_name, model, norm=normalize).items():
    print(f'Adding hook to layer: {layer_name}')
    hooks[layer_name] = Hook(module, layer_name)

# run the perturbed stimuli through the model
Y_hat = model(torch.tensor(X_adv))

# put activations and pixels into a dictionary with layer names
features_dict = {'0.pixels' : X_adv}
features_dict.update({layer_name: hook.activations for layer_name, hook in hooks.items()})

# print the feature names and shapes, to check it's as expected:
features_dict_description = {key : features_dict[key].shape for key in features_dict.keys()}
print('---')
for k, v in features_dict_description.items():
    print(k, v)


#%% run MFTMA analysis
# run MFTMA analysis on all features in the features_dict -- this can take a few minutes!
NT=100
seeded_analysis=False
seed_everything(seed)
df = MFTMA_analyze_activations(features_dict, img_idx, P, M, N, NT=NT, seeded=seeded_analysis, seed=seed, labels=Y_adv)

## add additional meta data
df['model'] = model_name
df['manifold_type'] = manifold_type
df['norm_method'] = normalize
df['clean_accuracy'] = clean_accuracy
df['adv_accuracy'] = adv_accuracy
df['eps'] = eps
df['eps_step'] = eps_step
df['max_iter'] = max_iter
df['random'] = random

# Let's take a peak at what we've created
print(df.head(3))


#%% save the results
# store the results
results_dir = 'results'
analysis_run_number = 2
if len(img_idx) < 10:
    file_name = f'model={model_name}-manifold={manifold_type}-norm={normalize}-eps={eps}-iter={max_iter}-random={random}-seeded={seeded_analysis}-seed_analysis={seed}-num_manifolds={P}-img_idx={img_idx}-NT={NT}-run_number={analysis_run_number}.csv'
else:
    file_name = f'model={model_name}-manifold={manifold_type}-norm={normalize}-eps={eps}-iter={max_iter}-random={random}-seeded={seeded_analysis}-seed_analysis={seed}-num_manifolds={P}-range={len(img_idx)}-NT={NT}-run_number={analysis_run_number}.csv' 
 

save_file = os.path.join(results_dir, model_name, file_name)
print(save_file)
df.to_csv(save_file)



#%% checking to see whether features and projection matrices are the same 
# print(list(df))  # list headers 

def get_layer_values(layer_name: str, header_name: str, exemplar_idx: int):
    return df[df['layer']==layer_name][header_name].iloc[exemplar_idx]

# get features for the 0th manifold and compare them
conv1_features = get_layer_values('1.conv1', 'features', 0)
norm1_features = get_layer_values('2.norm', 'features', 0)
# features are the same in layer 1
print('1st layer features before and after norm are the same:', np.allclose(conv1_features, norm1_features))

conv2_features = get_layer_values('4.conv2', 'features', 0)
norm2_features = get_layer_values('5.norm', 'features', 0)
# features are the same in layer 2
print('2nd layer features before and after norm are the same:', np.allclose(conv2_features, norm2_features))


# compare the extracted projection matrices
conv1_projmat = get_layer_values('1.conv1', 'projection_matrix', 0)
norm1_projmat = get_layer_values('2.norm', 'projection_matrix', 0)
# projection matrices are the same in layer 1
print('1st layer projection matrices before and after norm are the same˙:', np.allclose(conv1_projmat, norm1_projmat))

conv2_projmat = get_layer_values('4.conv2', 'projection_matrix', 0)
norm2_projmat = get_layer_values('5.norm', 'projection_matrix', 0)
# no projection in layer 2
# print(np.allclose(conv2_projmat, norm2_projmat))
print(np.allclose(conv2_projmat, norm2_projmat))


#%% analyzing each layer separately 
from mftma_small.manifold_analysis import manifold_analysis
kappa = 0
NT = 1000
# X = conv2_features
conv2_features_0 = get_layer_values('4.conv2', 'features', 0)
conv2_features_1 = get_layer_values('4.conv2', 'features', 1)

X = [conv2_features_0, conv2_features_1]

# seed_everything(seed)
capacity_all, radius_all, dimension_all = manifold_analysis(X, kappa, NT)
print(capacity_all, radius_all, dimension_all)


norm2_features_0 = get_layer_values('5.norm', 'features', 0)
norm2_features_1 = get_layer_values('5.norm', 'features', 1)

X2 = [norm2_features_0, norm2_features_1]

# seed_everything(seed)
capacity_all, radius_all, dimension_all = manifold_analysis(X2, kappa, NT)
print(capacity_all, radius_all, dimension_all)




# %%
