#%% import necessary stuff for mftma analysis
import numpy as np
import os
import sys
import argparse
import pickle

sys.path.append('..')
os.chdir('../')

from cifar_layer_norm import ResNet, BasicBlock
from mnist_layer_norm import Net

import torch
import torch.nn as nn

os.chdir('./analysis')
from mftma.manifold_analysis_correlation import manifold_analysis_corr
from mftma.utils.analyze_pytorch import analyze
from mftma.utils.make_manifold_data import make_manifold_data
from mftma.utils.activation_extractor import extractor

from art.utils import load_mnist, load_cifar10
from robustness import attacker
from robustness.datasets import CIFAR
import dill

#%% seed everything helper function
def seed_everything(seed):
    #initiate seed to try to make the result reproducible 
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


#%% import trained model and select normalization method

# import trained models
def import_trained_model(n, dataset_name):
    save_folder = os.path.join("..", "..", "results", f"{dataset_name}_regularize", "trained_models", "standard")
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
        model = model.to(device)
        print("=> loaded checkpoint '{}' (epoch {})".format(model_name, checkpoint['epoch']))

    return model


#%% create manifold dataset and extract activations

# create the manifold dataset
def create_manifold_dataset(model, dataset_name, model_name, ep=0):
    sampled_classes = 10
    examples_per_class = 50
     
    # load dataset
    if eps == 0:
        if dataset_name=="mnist":
            (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
            x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)
            test_dataset = (x_test, x_test)
        elif dataset_name=="cifar":
            (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
            x_test = x_test.transpose(0,3,1,2).astype(np.float32)
            test_dataset = (x_test, y_test)
        
    elif ep in ['1.0', '2.0', '4.0', '6.0', '8.0']:
        if dataset_name=="cifar":
            save_path = os.path.join('..', '..', 'results', 'cifar_regularize', 'adv_dataset', 'standard')
            file_name = f'standard-normalize_{model_name}-wd_0.0005-seed_17-eps_{ep}.pkl'
            load_file = os.path.join(save_path, file_name)
            test_dataset = pickle.load(open(load_file, 'rb'))
            x_test = test_dataset['x']
            y_test = test_dataset['y']

    # transpose dataset from tuple of arrays into array of tuples
    test_dataset = list(zip(x_test, y_test))

    data = make_manifold_data(test_dataset, sampled_classes, examples_per_class, seed=0)
    data = [d.to(device) for d in data]
    
    # extract activations from the model
    activations = extractor(model, data, layer_types=['Conv2d', 'Linear'])
    return activations


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


#%% run mftma analysis on the prepped activations and store results for plotting

def calculate_harmonic_std(array):
    n = len(array)
    s2 = np.var(1/array)
    mean = 1/n * np.sum(1/array)

    var = 1/n * s2 / mean**4
    return var**0.5

def analyze(activations):
    metrics = {}
    capacities = dict()
    radii = dict()
    dimensions = dict()
    correlations = dict()

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
        capacities[layer_name] = a_mean
        radii[layer_name] = r_mean
        dimensions[layer_name] = d_mean
        correlations[layer_name] = r0

    metrics['capacities'] = capacities
    metrics['radii'] = radii
    metrics['dimensions'] = dimensions
    metrics['correlations'] = correlations

    return metrics


#%% save/export the results
def save_results(metrics, base_save_folder, dataset_name, model_name):
    # set the save path
    save_folder = os.path.join(base_save_folder, dataset_name)
    print(f'the save folder is {save_folder}')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    # save python object using pickle
    save_name_base = f'metrics_{model_name}'
    save_name = os.path.join(save_folder, save_name_base + '.pkl')
    pickle.dump(metrics, open(save_name,'wb'))


#%% main fn
if __name__ == '__main__':
    print("we are running!", flush=True)
    parser = argparse.ArgumentParser(description='Run MFTMA analysis on trained MNIST/CIFAR norm models')
    parser.add_argument('--norm_method', help='The normalization method')
    parser.add_argument('--dataset_name', help='The dataset the model was trained on')
    parser.add_argument('--save_folder', help='The folder to save the results of the analysis')
    parser.add_argument('--seed', help='set the seed of the run.')
    parser.add_argument('--ep', help='set the eps level')
    args = parser.parse_args()

    dataset_name = args.dataset_name  # cifar or mnist
    assert dataset_name in ["mnist", "cifar"], "Dataset should either be `mnist` or `cifar`."

    save_folder = args.save_folder
    if (args.save_folder==None):
        save_folder = '../../results/mftma/'

    model_name = args.norm_method
    normalize_methods = ["bn", "gn", "in", "ln", "lrnb", "lrnc", "lrns", "nn"]
    if (model_name != None):
        assert model_name in normalize_methods, "Chosen method should be a valid norm."
        model_name = [model_name]
    else:
        model_name = normalize_methods
    print(f'normalization methods to be analyzed: {model_name}')

    ep = args.ep
    print(f'eps level to be analyzed: {ep}')

    seed_everything(int(args.seed))


    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(device)

    for m in model_name:
        print(f'importing the {m} normalization model')
        model = import_trained_model(m, dataset_name)

        print(f'creating manifold dataset for norm model {m}...')
        activations = create_manifold_dataset(model, dataset_name, model_name=m, ep=ep)
        print('extracted layers:\n', list(activations.keys()))

        print(f'preparing norm model {m} for analysis...')
        activations = prepare_data_for_analysis(activations)

        print(f'running analysis on norm model {m}...')
        metrics = analyze(activations)
        
        print('saving the results of the analysis...')
        save_results(metrics, save_folder, dataset_name, m)


# %%
