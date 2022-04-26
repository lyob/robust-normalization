import os
import random
import time
import pickle

import submitit
import numpy as np

import torch as ch
import torch.nn as nn
import torch.optim as optim

from robustness import model_utils, train, defaults
from robustness.datasets import CIFAR

import cox.store as store
from cox.utils import Parameters

from art.utils import load_cifar10
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent


def seed_everything(seed):
    #initiate seed to try to make the result reproducible 
    ch.manual_seed(seed)
    ch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def test_model(model, n_images=10000):
    #load cifar dataset using ART
    (_, _), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
    x_test = x_test.transpose(0,3,1,2).astype(np.float32)
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape((3, 1, 1))
    std = np.array([0.2023, 0.1994, 0.2010]).reshape((3, 1, 1))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(min_pixel_value, max_pixel_value),
        preprocessing=(mean, std),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
    )

    #eval with clean image
    predictions = classifier.predict(x_test[:n_images])
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test[:n_images], axis=1)) / n_images
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))
    return accuracy, classifier, x_test, y_test


def train_model(cluster, base_save_folder, model_name, seed, norm_method, weight_decay, learning_rate):
    model_save_name = f'nm:{norm_method}-seed:{seed}-wd:{weight_decay}'
    print(f'model name: {model_save_name}')

    # ensuring reproducibility
    seed_everything(seed)

    # save folder
    save_folder = os.path.join(base_save_folder, model_name, 'trained_models')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Create a cox store for logging
    out_store = store.Store(save_folder, exp_id = model_save_name)

    # Hard-coded dataset, architecture, batch size, workers
    if cluster == 'flatiron':
        datapath='/mnt/ceph/users/blyo1/syLab/robust-normalization/datasets/'  # flatiron cluster
    ds = CIFAR(datapath)
    model, _ = model_utils.make_and_restore_model(arch='VGG11', dataset=ds)
    train_loader, val_loader = ds.make_loaders(batch_size=128, workers=8)

    # Hard-coded base parameters
    train_kwargs = {
        'out_dir': "train_out",
        'adv_train': 0,
        'epochs': 120,
        'step_lr': 40,
        'weight_decay': weight_decay
    }
    train_args = Parameters(train_kwargs)

    # Fill whatever parameters are missing from the defaults
    train_args = defaults.check_and_fill_args(train_args, defaults.TRAINING_ARGS, CIFAR)


    # Train a model
    print('training starting! ...')
    model = train.train_model(train_args, model, (train_loader, val_loader), store=out_store)

    # Test the performance on a clean dataset
    model = model.eval()
    model = model.module.model if hasattr(model, 'module') else model.model
    _, _, _, _ = test_model(model)


def eval_model(cluster, base_save_folder, model_name, seed, norm_method, weight_decay, eps, attack_mode='inf'):
    # ensuring reproducibility
    seed_everything(seed)

    # load trained model weights
    load_folder = os.path.join(base_save_folder, model_name, 'trained_models')
    model_load_name = model_save_name = f'nm:{norm_method}-seed:{seed}-wd:{weight_decay}'
    model_weights = os.path.join(load_folder, model_load_name, 'checkpoint.pt.best')

    # load dataset and prep model
    if cluster == 'flatiron':
        datapath='/mnt/ceph/users/blyo1/syLab/robust-normalization/datasets/'  # flatiron cluster
    ds = CIFAR(datapath)
    model, _ = model_utils.make_and_restore_model(arch='VGG11', dataset=ds, resume_path=model_weights)
    model = model.eval()
    model = model.module.model if hasattr(model, 'module') else model.model

    # Test performance on clean dataset
    n_images = 10000
    accuracy, classifier, x_test, y_test = test_model(model, n_images)
    saved_perf = {}
    saved_perf['clean'] = accuracy

    eps = [float(i) for i in eps]
    eps_cifar = [i/255.0 for i in eps]
    record = np.zeros((len(eps_cifar)))

    norm = np.inf if attack_mode == 'inf' else int(attack_mode)

    adv_dataset_folder = os.path.join(base_save_folder, model_name, 'adv_dataset')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for ep_idx, ep in enumerate(eps_cifar):
        print(f'Attack strength: {ep}')
        attack = ProjectedGradientDescent(
            estimator= classifier,
            norm = norm,
            max_iter=32,
            eps=ep,
            eps_step=ep/16,
            targeted=False)
        x_test_adv = attack.generate(x=x_test[:n_images], y=y_test[:n_images])

        # save perturbed dataset
        adv_dataset = {}
        adv_dataset['x'] = x_test_adv
        adv_dataset['y'] = y_test[:n_images]
        adv_dataset_name = os.path.join(adv_dataset_folder, f'{model_save_name}-eps:{ep}.pkl')
        pickle.dump(adv_dataset, open(adv_dataset_name, 'wb'))
        
        # calculate accuracy of network on adversarial inputs
        predictions = classifier.predict(x_test_adv)
        adv_accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test[:n_images], axis=1)) / len(y_test[:n_images])
        print(f"Accuracy on adversarial test examples: {adv_accuracy*100}",flush=True)

        record[ep_idx] = adv_accuracy
    
    saved_perf['perturbed'] = record

    # define save path
    save_folder = os.path.join(base_save_folder, model_name, 'eval_models')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # save prediction performance for all eps for both clean and perturbed datasets
    eps = [str(i) for i in eps]
    eps_save_name = '_'.join(eps)
    save_name = os.path.join(save_folder, f'{model_save_name}-eps:{eps_save_name}.pkl')
    pickle.dump(saved_perf, open(save_name,'wb'))


def main():
    t0 = time.time()

    print('we are running! :)')

    # set parameters
    cluster = 'flatiron'  # flatiron or nyu-greene
    base_save_folder = 'results'
    model_name = 'vgg1'  
    mode = 'val'  # train, val
    weight_decay = [0.0005]
    seed = [2]
    norm_method = ['nn']
    eps = [0, 1.0, 2.0]
    attack_mode = 'inf'  # inf, 1, 2, None, default=inf
    learning_rate = 0.01
    
    # parameter processing
    log_folder = os.path.join('..', 'slurm_jobs', cluster, 'train_and_eval', 'logs')
    base_save_folder = os.path.join('..', base_save_folder, 'vgg')

    ex = submitit.AutoExecutor(folder=log_folder)

    if ex.cluster == 'slurm':
        print('submitit executor will schedule jobs on slurm!')
    else:
        print(f"!!! Slurm executable `srun` not found. Will execute jobs on '{ex.cluster}'")

    # slurm parameters
    ex.update_parameters(
        nodes=1, 
        slurm_partition="gpu", 
        slurm_gpus_per_task=1, 
        slurm_constraint='a100', 
        cpus_per_task=12, 
        mem_gb=16, 
        timeout_min=60
    )

    # submit the jobs!
    jobs = []
    with ex.batch():
        # iterate through all the parameters
        for s in seed:
            for n in norm_method:
                for wd in weight_decay:
                    if mode=='train':
                        job = ex.submit(train_model, cluster, base_save_folder, model_name, s, n, wd, learning_rate)
                    elif mode=='eval':
                        job = ex.submit(eval_model, cluster, base_save_folder, model_name, s, n, wd, eps, attack_mode)
                    jobs.append(job)
    print('all jobs submitted!')

    idx = 0
    for s in seed:
        for n in norm_method:
            for wd in weight_decay:
                print(f'Job {jobs[idx].job_id} === seed: {s}, norm method: {n}, weight decay: {wd}')
                idx += 1

if __name__ == '__main__':
    main()