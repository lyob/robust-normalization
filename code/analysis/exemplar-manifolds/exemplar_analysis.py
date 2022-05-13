#%% import packages
%reload_ext autoreload
%autoreload 2

import os
import sys
from glob import glob

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
P = 50 # number of manifolds, i.e. the number of images 
M = 50 # number of examples per manifold, i.e. the number of images that lie in an epsilon ball around the image  
N = 2000 # maximum number of features to use ??

# determine the type of adversarial examples to use for constructing the manifolds
eps = 0.1  # 8/255, 0
max_iter = 1
eps_step_factor = 1
eps_step = eps / eps_step_factor

random = False  # adversarial perturbation if false, random perturbation if true

#%% model and dataset details
## regular ResNet18 ('CIFAR_ResNet18') or VOneResNet18 with Gaussian Noise ('CIFAR_VOneResNet18')
model_name = 'CIFAR_ResNet18'
dataset = 'CIFAR'

# how about MNIST with a LeNet
model_name = 'lenet'  # MNIST_ConvNet
dataset = 'mnist'

#%% Load train and test dataset, and show an example. 
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
plt.imshow(x_train[7])

x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)

x_train.shape, x_test.shape


#%% define the MNIST model
os.chdir('/Users/blyo/Documents/research/chung-lab/robust-normalization/code')
from mnist_layer_norm import Net_both
os.chdir('./analysis/exemplar-manifolds')
from helpers import load_model, art_wrap_model, accuracy

lenet_parameters = {
    'version': 'convnet4',
    'normalize': 'nn',
    'frontend': 'learned_conv',
    'seed': 1,
    'lr': 0.01,
    'wd': 0.005
}
parameters = lenet_parameters
normalize = parameters.get('normalize')
frontend = parameters.get('frontend')
version = parameters.get('version')
seed = parameters.get('seed')
wd = parameters.get('wd')
lr = parameters.get('lr')

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
model_path = os.path.join(load_dir, 'results', version, 'trained_models', f'{frontend}_frontend-norm_both', f'{version}-lr_{lr}-wd_{wd}-seed_{seed}-normalize_{normalize}.pth')
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


#%% generate adversarial manifold stimuli
from helpers import perturb_stimuli, construct_manifold_stimuli

X, Y = construct_manifold_stimuli(x_test, y_test, manifold_type, P=P, M=M)
print(X.shape)
print(type(X))
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

# get adversarial accuracy
adv_accuracy = accuracy(classifier.predict(X_adv), Y)
print(f"Accuracy on adversarial test examples: {adv_accuracy * 100}")


#%% extract activations
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
df = MFTMA_analyze_activations(features_dict, P, M, N=N, seed=seed)

## add additional meta data
df['model'] = model_name
df['manifold_type'] = manifold_type
df['clean_accuracy'] = clean_accuracy
df['adv_accuracy'] = adv_accuracy
df['eps'] = eps
df['eps_step'] = eps_step
df['max_iter'] = max_iter
df['random'] = random

# Let's take a peak at what we've created
print(df.head(3))

# store the results

# where to save results and how to name the files
results_dir = 'results'
file_name = f'dataset={dataset}-model={model_name}-manifold={manifold_type}-eps={eps}-iter={max_iter}-random={random}-seed={seed}.csv'

save_file = os.path.join(results_dir, file_name)
print(save_file)
df.to_csv(save_file)


# %%
