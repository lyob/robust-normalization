#%% import necessary stuff for mftma analysis
import numpy as np
import os
folder_path = '../'
os.chdir(folder_path)

from cifar_layer_norm import ResNet, BasicBlock
from mnist_layer_norm import Net

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from importlib import reload
import mftma
reload(mftma)
from mftma.manifold_analysis_correlation import manifold_analysis_corr
from mftma.utils.analyze_pytorch import analyze
from mftma.utils.activation_extractor import extractor

from art.utils import load_mnist, load_cifar10
from robustness import attacker
from robustness.datasets import CIFAR
import dill

#%% import trained model and select normalization method

dataset_name = "cifar"  # cifar or mnist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(device)

# import trained models
def import_trained_model(n, dataset_name):
    save_folder = os.path.join("..", "results", f"{dataset_name}_regularize", "trained_models", "standard")
    if (dataset_name=="mnist"):
        model_name_base = "standard-lr_0.01-wd_0.0005-seed_17-normalize_"
        model_name = os.path.join(save_folder, f"{model_name_base}{n}.pth")
        print(model_name)

        # load model
        simple_channels = 16
        complex_channels = 16
        ksize = 5
        conv_1 = nn.Conv2d(in_channels=1, out_channels=simple_channels+complex_channels, kernel_size=ksize, stride=2, padding=ksize//2)
        model = Net(conv_1, simple_channels + complex_channels, normalize=n)
        model = model.to(device)

        sd = torch.load(model_name, map_location=device)
        print('sd', sd)
        model.load_state_dict(sd)
        model.eval()
    
    if (dataset_name=="cifar"):
        model_name_folder = f"standard-normalize_{n}-wd_0.0005-seed_17"
        model_name = os.path.join(save_folder, model_name_folder, "checkpoint.pt.best")

        # define architecture
        arch = ResNet(BasicBlock, [2,2,2,2], normalize=n)

        # load dataset
        ds = CIFAR(data_path='../../datasests/')
        
        # load previously saved model
        model = attacker.AttackerModel(arch, ds)

        # load saved weights
        checkpoint = torch.load(model_name, pickle_module=dill, map_location=device)
        
        sd = checkpoint["model"]
        sd = {k[len('module.'):]:v for k,v in sd.items()}
        model.load_state_dict(sd)
        model.eval()
        print(model.model)
        # model = model.to(device)
        print("=> loaded checkpoint '{}' (epoch {})".format(model_name, checkpoint['epoch']))

    return model

# for idx, m in enumerate(normalize_method):
trained_models = {}
normalize_method = ["bn", "gn", "in", "ln", "lrnb", "lrnc", "lrns", "nn"]
models = dict()
for idx, norm in enumerate(normalize_method):
    print(f'importing the {norm} model')
    model = import_trained_model(norm, dataset_name)
    models[norm] = model

#%% create manifold dataset and extract activations

reload(mftma)
# reload(sys.modules['mftma'])
reload(mftma.utils.make_manifold_data)
reload(mftma.utils.activation_extractor)
from mftma.utils.make_manifold_data import make_manifold_data
from mftma.utils.activation_extractor import extractor

# create the manifold dataset
def create_manifold_dataset(model, dataset_name):
    sampled_classes = 10
    examples_per_class = 50
     
    # load dataset
    if dataset_name=="mnist":
        (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
        x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
        train_dataset = (x_train, y_train)
    elif dataset_name=="cifar":
        (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
        x_train = x_train.transpose(0,3,1,2).astype(np.float32)
        train_dataset = (x_train, y_train)
    
    # transpose dataset from tuple of arrays into array of tuples
    train_dataset = list(zip(x_train, y_train))
    # print(train_dataset[0][0].shape)

    data = make_manifold_data(train_dataset, sampled_classes, examples_per_class, seed=0)
    data = [d.to(device) for d in data]
    
    # extract activations from the model
    # activations = extractor(model, data, layer_types=['Conv2d', 'Linear'])
    activations = extractor(model, data)
    return activations

model_activations = dict()
for model_name, model in models.items():
    print(f'creating manifold dataset for model {model_name}...')
    activations = create_manifold_dataset(model, dataset_name)
    print('extracted layers:\n', list(activations.keys()))
    model_activations[model_name] = activations


#%% prepare activations for analysis
'''
we need to convert the extracted activations into the correct shape so 
that we can pass them into `manifold_analysis_correlation.py`.
'''

# activations is an OrderedDict, with keys:layer_name, items:data
def prepare_data_for_analysis(activations):
    for layer, data in activations.items():
        X = [d.reshape(d.shape[0], -1).T for d in data]

        # Get the number of features in the flattened data
        N = X[0].shape[0]
        print(f'layer: {layer}, N: {N}')
        
        # If N is greater than 5000, do the random projection to 5000 features
        if N > 5000:
            print("Projecting {}".format(layer))
            M = np.random.randn(5000, N)
            M /= np.sqrt(np.sum(M*M, axis=1, keepdims=True))
            X = [np.matmul(M, d) for d in X]

        activations[layer] = X
    return activations

model_activations2 = dict()
for model_name, activations in model_activations.items():
    print(f'preparing model {model_name} for analysis...')
    model_activations2[model_name] = prepare_data_for_analysis(activations)


#%% run mftma analysis on the prepped activations and store results for plotting

def calculate_harmonic_std(array):
    n = len(array)
    s2 = np.var(1/array)
    mean = 1/n * np.sum(1/array)

    var = 1/n * s2 / mean**4
    return var**0.5

def analyze(activations):
    capacities = []
    radii = []
    dimensions = []
    correlations = []

    for layer_name, X, in activations.items():
        # Analyze each layer's activations
        a, r, d, r0, K = manifold_analysis_corr(X, 0, 300, n_reps=1)
        
        # Compute the mean values
        a_mean = 1/np.mean(1/a) 
        r_mean = np.mean(r)
        d_mean = np.mean(d)
        
        print(f"{layer_name}\n \
            capacity: {a_mean:4f}\n \
            radius: {r_mean:4f}\n \
            dimension: {d_mean:4f}\n \
            correlation: {r0:4f}"
        )
        
        # Store for later
        capacities.append(a_mean)
        radii.append(r_mean)
        dimensions.append(d_mean)
        correlations.append(r0)

    return capacities, radii, dimensions, correlations

for model_name, activations in model_activations2.items():
    print(f'running analysis on model {model_name}...')
    capacities, radii, dimensions, correlations = analyze(activations)


#%% plot the results
'''
we plot the results of the analysis we just ran. Note we won't plot the results of the
final linear layer because this is the model output after the model has already occurred
'''
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

axes[0].plot(capacities, linewidth=5)
axes[1].plot(radii, linewidth=5)
axes[2].plot(dimensions, linewidth=5)
axes[3].plot(correlations, linewidth=5)

axes[0].set_ylabel(r'$\alpha_M$', fontsize=18)
axes[1].set_ylabel(r'$R_M$', fontsize=18)
axes[2].set_ylabel(r'$D_M$', fontsize=18)
axes[3].set_ylabel(r'$\rho_{center}$', fontsize=18)

names = list(activations.keys())
names = [n.split('_')[1] + ' ' + n.split('_')[2] for n in names]
for ax in axes:
    ax.set_xticks([i for i, _ in enumerate(names)])
    ax.set_xticklabels(names, rotation=90, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()
plt.show()

#%% 


