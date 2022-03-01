#%% import packages
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
from art.estimators.classification import PyTorchClassifier
import argparse

from helpers import accuracy, perturb_stimuli, construct_manifold_stimuli, Hook, model_layer_map, MFTMA_analyze_activations

sys.path.append('..')
os.chdir('..')
from mnist_layer_norm import Net


# Load train and test dataset, and show an example. 
def load_dataset():
    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
    # plt.imshow(x_train[7])

    # x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
    x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)

    # print('x_train shape', x_train.shape)
    # print('x_test shape', x_test.shape)

    return x_test, y_test, min_pixel_value, max_pixel_value


# define the MNIST model
def load_model(norm_method):
    simple_channels = 16
    complex_channels = 16
    ksize = 5

    conv_1 = nn.Conv2d(in_channels=1, out_channels=simple_channels+complex_channels, kernel_size=ksize, stride=2, padding=ksize//2)
    model = Net(conv_1, simple_channels + complex_channels, normalize=norm_method)
    return model


# load model state and dataset
def load_model_state(normalize, model):
    # load the model
    model_path = os.path.abspath(os.path.join('..', '..', 'results', 'mnist_regularize', 'trained_models', 'standard', f'standard-lr_0.01-wd_0.0005-seed_17-normalize_{normalize}.pth'))
    model.load_state_dict(torch.load(model_path, map_location=device))

    lr = 0.01
    wd = 0.0005
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    _, _, min_pixel_value, max_pixel_value = load_dataset()

    # wrap the model with adversarial robustness toolbox, for generating adversarial stimuli.
    classifier = PyTorchClassifier(
        device_type=device, 
        model=model, 
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, 28, 28), 
        nb_classes=10,
    )
    return classifier 


# get model clean accuracy
def get_clean_accuracy(classifier, **kwargs):
    x_test, y_test, _, _ = load_dataset()

    if kwargs and kwargs['n_images']:
        n_images = kwargs['n_images']  # n_images = 10000
        x_test = x_test[:n_images]

    predictions = classifier.predict(x_test)
    clean_accuracy = accuracy(predictions, y_test)
    print("Accuracy on benign test examples: {}%".format(clean_accuracy * 100))
    return clean_accuracy


# generate adversarial manifold stimuli
def create_manifold_stimuli(classifier, manifold_type, P, M, eps, eps_step_factor, max_iter, random):
    print('constructing manifold stimuli...')

    x_test, y_test, _, _ = load_dataset()
    X, Y = construct_manifold_stimuli(x_test, y_test, manifold_type, P=P, M=M)
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

    # get adversarial accuracy
    adv_accuracy = accuracy(classifier.predict(X_adv), Y)
    print(f"Accuracy on adversarial test examples: {adv_accuracy * 100}")

    return X_adv, adv_accuracy


# apply hooks to the model, to extract intermediate representations
def extract_representations(model_name, model, normalize, X_adv):
    print('extracting representations...')

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
    print('---')

    return features_dict


# run MFTMA analysis on all features in the features_dict -- this can take a few minutes!
def run_mftma(features_dict, P, M, N, seed, model_name, manifold_type, clean_accuracy, adv_accuracy, eps, max_iter, random):
    print('running mftma...')

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

    return df


def save_results(df, results_dir, file_name):
    print('saving results...')

    # store the results
    save_file = os.path.abspath(os.path.join('exemplar-manifolds', results_dir, file_name))
    df.to_csv(save_file)

    print(f'results saved at: {results_dir}')


if __name__ == "__main__":
    print("we are running!", flush=True)
    parser = argparse.ArgumentParser(description='Run MFTMA analysis on trained MNIST/CIFAR norm models')
    parser.add_argument('--norm_method', help='The normalization method')
    parser.add_argument('--seed', help='set the seed of the run.')
    parser.add_argument('--eps', help='set the eps level')
    parser.add_argument('--num_images', help='number of images to use from test dataset')
    args = parser.parse_args()

    assert args.norm_method in ['bn', 'in', 'ln', 'gn', 'nn', 'lrnc', 'lrns', 'lrnb'], 'Must input a valid norm method.'

    if args.seed:
        seed = int(args.seed)
    else:
        seed = 0

    # manifold parameters
    manifold_type = 'exemplar' # 'class' for traditional label based manifolds, 'exemplar' for individual exemplar manifolds
    P = 50 # number of manifolds
    M = 50 # number of examples per manifold
    N = 2000 # maximum number of features to use

    # determine the type of adversarial examples to use for constructing the manifolds
    if args.eps:
        eps = {args.eps}/255
    else:
        eps = 8/255
    max_iter = 1
    eps_step_factor = 1
    eps_step = eps / eps_step_factor
    random = False # adversarial perturbation if false, random perturbation if true
    
    # model and dataset details
    model_name = 'MNIST_ConvNet'
    dataset = 'MNIST'

    # where to save results and how to name the files
    results_dir = 'results'
    file_name = f'model_{model_name}-manifold_{manifold_type}-eps_{eps}-iter_{max_iter}-random_{random}-seed_{seed}.csv'

    global device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    # x_test, y_test, _, _ = load_dataset()
    model = load_model(args.norm_method)
    classifier = load_model_state(args.norm_method, model)
    clean_acc = get_clean_accuracy(classifier)
    X_adv, adv_accuracy = create_manifold_stimuli(classifier, manifold_type, P, M, eps, eps_step_factor, max_iter, random)
    features_dict = extract_representations(model_name, model, args.norm_method, X_adv)
    df = run_mftma(features_dict, P, M, N, seed, model_name, manifold_type, clean_acc, adv_accuracy, eps, max_iter, random)
    save_results(df, results_dir, file_name)


        