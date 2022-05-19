# import packages
import os
import sys
import random
from tkinter import E

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle

from art.utils import load_mnist
from art.estimators.classification import PyTorchClassifier
import argparse
import submitit

os.chdir('/mnt/ceph/users/blyo1/syLab/robust-normalization/code/analysis/exemplar-manifolds')
sys.path.append('/mnt/ceph/users/blyo1/syLab/robust-normalization/code/analysis/exemplar-manifolds')
from helpers import accuracy, perturb_stimuli, construct_manifold_stimuli, Hook, model_layer_map, MFTMA_analyze_activations

from mnist_layer_norm import Net_both

def seed_everything(seed):
    #initiate seed to try to make the result reproducible 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Load train and test dataset, and show an example. 
def load_dataset():
    (_, _), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
    # plt.imshow(x_train[7])

    # x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
    x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)

    # print('x_train shape', x_train.shape)
    # print('x_test shape', x_test.shape)
    return x_test, y_test, min_pixel_value, max_pixel_value


# define the MNIST model
def load_model(model_name, norm_method):
    if model_name[:7] == 'convnet':
        simple_channels = 16
        complex_channels = 16
        ksize = 5

        conv_1 = nn.Conv2d(in_channels=1, out_channels=simple_channels+complex_channels, kernel_size=ksize, stride=2, padding=ksize//2)
        model = Net_both(conv_1, simple_channels + complex_channels, normalize=norm_method)
    return model


# load model state and dataset
def load_model_state(model_load_name, norm_method, model, wd, lr, model_load_seed=1, run_number=1):
    # load the model
    load_dir = os.path.join('../../..')
    model_path = os.path.join(load_dir, 'results', model_load_name, 'trained_models', f'learned_conv_frontend-norm_both', f'{model_load_name}-lr_{lr}-wd_{wd}-seed_{model_load_seed}-normalize_{norm_method}.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

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
def create_manifold_stimuli(generate_new: bool, classifier, manifold_type, P, M, N, img_idx, eps, eps_step_factor, max_iter, random, seed, run_number: int):
    print('constructing manifold stimuli...')

    if generate_new:
        print('generating new dataset...')
        x_test, y_test, _, _ = load_dataset()
        X, Y = construct_manifold_stimuli(x_test, y_test, manifold_type, P=P, M=M, img_choice=img_idx)
        if (eps == 0):
            X_adv = X
        else:
            seed_everything(seed)
            X_adv = perturb_stimuli(
                X, 
                Y, 
                classifier, 
                eps=eps, 
                eps_step_factor=eps_step_factor, 
                max_iter=max_iter, 
                random=random 
            )
        data_to_save = {'X_adv': X_adv, 'X': X, 'Y': Y, 'img_idx': img_idx}
        save_dir = 'adversarial_dataset'
        if len(img_idx) < 10:
            save_name = f'adversarial_dataset-P={P}-M={M}-N={N}-img_idx={img_idx}-run={run_number}.pkl'
        else:
            save_name = f'adversarial_dataset-P={P}-M={M}-N={N}-range={len(img_idx)}-run={run_number}.pkl'
            
        save_file = open(os.path.join(save_dir, save_name), 'wb')
        pickle.dump(data_to_save, save_file)
        save_file.close()
    else:
        print('loading saved dataset...')
        load_dir = os.path.join('adversarial_dataset')
        if len(img_idx) < 10:
            load_file = f'adversarial_dataset-P={P}-M={M}-N={N}-img_idx={img_idx}-run={run_number}.pkl'
        else:
            load_file = f'adversarial_dataset-P={P}-M={M}-N={N}-range={len(img_idx)}-run={run_number}.pkl'
        file = open(os.path.join(load_dir, load_file), 'rb')
        loaded_data = pickle.load(file)
        
        X_adv = loaded_data['X_adv']
        Y = loaded_data['Y']

    print(f'stimuli shape: {X_adv.shape}')
    print(f'X_adv type: {type(X_adv)}')

    # get adversarial accuracy
    adv_accuracy = accuracy(classifier.predict(X_adv), Y)
    print(f"Accuracy on adversarial test examples: {adv_accuracy * 100}")
    
    Y_adv = np.argmax(Y, axis=1)
    Y_adv = Y_adv[[i*M for i in range(len(img_idx))]]
    
    return X_adv, Y_adv, adv_accuracy

# apply hooks to the model, to extract intermediate representations
def extract_representations(model_name, model, normalize, X_adv):
    print('extracting representations...')

    hooks = {}
    for layer_name, module in model_layer_map(model_name, model, norm=normalize).items():
        print(f'Adding hook to layer: {layer_name}')
        hooks[layer_name] = Hook(module, layer_name)

    # run the perturbed stimuli through the model
    Y_hat = model(torch.tensor(X_adv).to(device))

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
def run_mftma(features_dict, img_idx, Y_adv, P, M, N, NT, seeded, seed, model_name, manifold_type, norm_method, clean_accuracy, adv_accuracy, eps, eps_step_factor, max_iter, random):
    print('running mftma...')

    # seed_everything(seed)
    df = MFTMA_analyze_activations(features_dict, img_idx, P, M, N, NT=NT, seeded=seeded, seed=seed, labels=Y_adv)

    eps_step = eps/eps_step_factor

    ## add additional meta data
    df['model'] = model_name
    df['manifold_type'] = manifold_type
    df['norm_method'] = norm_method
    df['clean_accuracy'] = clean_accuracy
    df['adv_accuracy'] = adv_accuracy
    df['eps'] = eps
    df['eps_step'] = eps_step
    df['max_iter'] = max_iter
    df['random'] = random
    return df


def save_results(df, results_dir, model_save_name, file_name):
    print('saving results...')

    # store the results
    save_dir = os.path.join('.', results_dir, model_save_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_file = os.path.join(save_dir, file_name)
    df.to_csv(save_file)

    print(f'results saved at: {save_dir}')


#%% ANALYSIS
def run_analysis(
                    model_load_name, model_save_name, 
                    norm_method, wd, lr, model_load_seed, 
                    generate_new, manifold_type, eps, max_iter, eps_step_factor, attack_mode, random, img_idx, dataset_run_number, seed_dataset,
                    P, M, N, NT, is_seeded, seed_analysis, analysis_run_number, 
                    results_dir='results'
                ):
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_model(model_load_name, norm_method)
    classifier = load_model_state(model_load_name, norm_method, model, wd, lr, model_load_seed)
    clean_acc = get_clean_accuracy(classifier)

    # create or load manifold dataset
    X_adv, Y_adv, adv_accuracy = create_manifold_stimuli(generate_new, classifier, manifold_type, P, M, N, img_idx, eps, eps_step_factor, max_iter, random, seed_dataset, dataset_run_number)

    # extract model features        
    features_dict = extract_representations(model_save_name, model, norm_method, X_adv)

    if not is_seeded:
        seed_analysis == is_seeded

    # run the mftma analysis
    df = run_mftma(features_dict, img_idx, Y_adv, P, M, N, NT, is_seeded, seed_analysis, model_load_name, manifold_type, norm_method, clean_acc, adv_accuracy, eps, eps_step_factor, max_iter, random)

    # if is_seeded:
    #     seed_save_string = f'{seed_analysis}'
    # else:
    #     seed_save_string = f'False'

    if len(img_idx) < 10:
        file_name = f'model={model_save_name}-manifold={manifold_type}-norm={norm_method}-eps={eps}-iter={max_iter}-random={random}-seeded={is_seeded}-seed_analysis={seed_analysis}-num_manifolds={P}-img_idx={img_idx}-NT={NT}-run_number={analysis_run_number}.csv'
    else:
        file_name = f'model={model_save_name}-manifold={manifold_type}-norm={norm_method}-eps={eps}-iter={max_iter}-random={random}-seeded={is_seeded}-seed_analysis={seed_analysis}-num_manifolds={P}-range={len(img_idx)}-NT={NT}-run_number={analysis_run_number}.csv'
    save_results(df, results_dir, model_save_name, file_name)


#%% MAIN
def main():
    print("we are running!", flush=True)
    
    ############################################################################################
    # parameters pertaining to model and dataset details
    model_load_name = 'convnet4'
    model_save_name = 'lenet'
    dataset = 'mnist'
    
    # parameters for loading the weights of the trained model
    norm_method = ['nn']
    # norm_method = ['nn', 'bn', 'in', 'gn', 'ln', 'lrnb', 'lrnc', 'lrns']
    wd = 0.005
    lr = 0.01
    model_load_seed = [1]
    
    # parameter for creating (exemplar) manifold
    generate_new = False
    manifold_type = 'exemplar' # 'class' for traditional label based manifolds, 'exemplar' for individual exemplar manifolds
    eps = [0.1]
    max_iter = 1
    eps_step_factor = 1
    attack_mode = 'inf'  # inf, 1, 2, None, default=inf
    random = False # adversarial perturbation if false, random perturbation if true
    # img_idx = [0, 4]  # select which images from the test dataset to condition on, can be list of ints or `False`
    img_idx = list(range(50))  # select which images from the test dataset to condition on, can be list of ints or `False`
    dataset_run_number = 1
    seed_dataset = 1
    
    # parameters for running analysis
    P = len(img_idx) # number of manifolds, i.e. the number of images
    M = 50 # number of examples per manifold, i.e. the number of images that lie in an epsilon ball around the image
    N = 2000 # maximum number of features to use
    NT = 2000  # number of sampled directions
    is_seeded = False
    seed_analysis = [0, 1, 2, 3, 4, 5]
    analysis_run_number = [1]
    
    base_save_folder = 'results'
    

    cluster = 'flatiron'
    resources = 'cpu'  # cpu or gpu

    #########################################################################################
    # processing paths
    log_folder = os.path.join('../../..', 'slurm_jobs', cluster, 'mftma', f'{manifold_type}-manifolds', 'logs/%j')
    # base_save_folder = os.path.join('..', base_save_folder, 'vgg')

    # establish executor for submitit
    ex = submitit.AutoExecutor(folder=log_folder)

    if ex.cluster == 'slurm':
        print('submitit executor will schedule jobs on slurm!')
    else:
        print(f"!!! Slurm executable `srun` not found. Will execute jobs on '{ex.cluster}'")

    # slurm parameters
    if resources == 'cpu':
        # cpu usage
        ex.update_parameters(
            slurm_job_name='mftma',
            nodes=1,
            slurm_partition="ccn", 
            cpus_per_task=4,
            mem_gb=4,  # 32gb for train mode, 8gb for eval mode
            timeout_min=60
        )
    elif resources == 'gpu':
        # gpu usage
        ex.update_parameters(
            slurm_job_name='mftma_gpu',
            nodes=1,
            slurm_partition='gpu',
            slurm_gpus_per_task=1,
            slurm_constraint='a100',
            cpus_per_task=4,
            mem_gb=8,
            timeout_min=60
        )

    # submit the jobs!
    jobs = []
    with ex.batch():
        # iterate through all the parameters
        for ls in model_load_seed:
            for n in norm_method:
                for e in eps:
                    for r in analysis_run_number:
                        for s_a in seed_analysis:
                            job = ex.submit(
                                run_analysis, 
                                model_load_name, model_save_name, 
                                n, wd, lr, ls,
                                generate_new, manifold_type, e, max_iter, eps_step_factor, attack_mode, random, img_idx, dataset_run_number, seed_dataset,
                                P, M, N, NT, is_seeded, s_a, r,
                                base_save_folder
                            )
                            jobs.append(job)
    print('all jobs submitted!')

    idx = 0
    for ls in model_load_seed:
        for n in norm_method:
            for e in eps:
                for r in analysis_run_number:
                    for s in seed_analysis:
                        print(f'Job {jobs[idx].job_id} === seed: {s}, norm method: {n}, eps: {e}, run_number: {r}')
                        idx += 1


if __name__ == "__main__":
    main()



