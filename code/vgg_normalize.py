import torch as ch
import numpy as np
from robustness import model_utils, train, defaults
from robustness.datasets import CIFAR
from cox.utils import Parameters
import cox.store
import torch as ch
import os
import random
import submitit
import time


def seed_everything(seed):
    #initiate seed to try to make the result reproducible 
    ch.manual_seed(seed)
    ch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_model(cluster, save_folder, model_name, mode, seed, weight_decay, eps, attack_mode, learning_rate):
    # ensuring reproducibility
    seed_everything(seed)

    # save folder
    save_folder = os.path.join(save_folder, model_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Hard-coded dataset, architecture, batch size, workers
    if cluster == 'flatiron':
        datapath='/mnt/ceph/users/blyo1/syLab/robust-normalization/datasets/'  # flatiron cluster
    ds = CIFAR(datapath)
    model, _ = model_utils.make_and_restore_model(arch='vgg', dataset=ds)
    train_loader, val_loader = ds.make_loaders(batch_size=128, workers=8)

    # Create a cox store for logging
    out_store = cox.store.Store(save_folder)

    # Hard-coded base parameters
    train_kwargs = {
        'out_dir': "train_out",
        'adv_train': 0,
        'epochs': 1,
        'step_lr': 40,
        'weight_decay': weight_decay
    }
    train_args = Parameters(train_kwargs)

    # Fill whatever parameters are missing from the defaults
    train_args = defaults.check_and_fill_args(train_args, defaults.TRAINING_ARGS, CIFAR)
    train_args = defaults.check_and_fill_args(train_args, defaults.PGD_ARGS, CIFAR)

    # Train a model
    print('training starting! ...')
    train.train_model(train_args, model, (train_loader, val_loader), store=out_store)

def main():
    t0 = time.time()

    print('we are running! :)')

    # parameters
    cluster = 'flatiron'  # flatiron or nyu-greene
    base_save_folder = 'results'
    log_folder = os.path.join('slurm_jobs', cluster, 'train_and_eval')
    model_name = 'vgg1'  
    mode = 'train'  # train, val
    weight_decay = [0.0005]
    seed = [2]
    eps = ['0', '1.0', '2.0']
    eps = '_'.join(eps)
    attack_mode = 'inf'  # inf, 1, 2, None, default=inf
    learning_rate = 0.01
    save_folder = f'../{base_save_folder}/vgg'

    ex = submitit.AutoExecutor(folder=log_folder)

    if ex.cluster == 'slurm':
        print('submitit executor will schedule jobs on slurm!')
    else:
        print(f"!!! Slurm executable `srun` not found. Will execute jobs on '{ex.cluster}'")

    ex.update_parameters(mem_gb=4, cpus_per_task=1, tasks_per_node=1, nodes=1, timeout_min=60, slurm_partition="ccn")

    jobs = []
    with ex.batch():
        # iterate through all the parameters
        for e in eps:
            for s in seed:
                for wd in weight_decay:
                    job = ex.submit(train_model, cluster, save_folder, model_name, mode, s, wd, e, attack_mode, learning_rate)
                    print(f'scheduled {job}')
                    jobs.append(job)

if __name__ == '__main__':
    main()