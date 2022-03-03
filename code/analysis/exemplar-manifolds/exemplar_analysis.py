#%% import packages
%load_ext autoreload
%autoreload 2

import os
import sys
from glob import glob

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from art.utils import load_mnist

#%% set up parameters 
seed = 0

## 2. exemplar manifolds analysis example parameters 
manifold_type = 'exemplar' # 'class' for traditional label based manifolds, 'exemplar' for individual exemplar manifolds
P = 50 # number of manifolds
M = 50 # number of examples per manifold
N = 2000 # maximum number of features to use

# determine the type of adversarial examples to use for constructing the manifolds
eps = 8/255
eps = 0
max_iter = 1
eps_step_factor = 1
eps_step = eps / eps_step_factor
random = False # adversarial perturbation if false, random perturbation if true

#%% model and dataset details

## regular ResNet18 ('CIFAR_ResNet18') or VOneResNet18 with Gaussian Noise ('CIFAR_VOneResNet18')
model_name = 'CIFAR_ResNet18'
dataset = 'CIFAR'

# how about MNIST with ConvNet?
model_name = 'MNIST_ConvNet'
dataset = 'MNIST'

# where to save results and how to name the files
results_dir = 'results'
file_name = f'model_{model_name}-manifold_{manifold_type}-eps_{eps}-iter_{max_iter}-random_{random}-seed_{seed}.csv'

#%% Load train and test dataset, and show an example. 
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
plt.imshow(x_train[7])

x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)

x_train.shape, x_test.shape


#%% define the MNIST model

# sys.path.append('..')
os.chdir('../..')
from mnist_layer_norm import Net
simple_channels = 16
complex_channels = 16
ksize = 5
normalize = 'nn'

conv_1 = nn.Conv2d(in_channels=1, out_channels=simple_channels+complex_channels, kernel_size=ksize, stride=2, padding=ksize//2)
model = Net(conv_1, simple_channels + complex_channels, normalize=normalize)


#%% load model state and dataset
from helpers import load_model, art_wrap_model, accuracy

global device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# load the model
model_path = os.path.abspath(os.path.join('..', 'results', 'mnist_regularize', 'trained_models', 'standard', f'standard-lr_0.01-wd_0.0005-seed_17-normalize_{normalize}.pth'))
model.load_state_dict(torch.load(model_path, map_location=device))


import torch.optim as optim
lr = 0.01
wd = 0.0005
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
save_file = os.path.abspath(os.path.join('exemplar-manifolds', results_dir, file_name))
print(save_file)
df.to_csv(save_file)


# %%
