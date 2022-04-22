import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import dill
import h5py as h5
import os
import argparse

# we use the adversarial robustness toolbox (art) to evaluate models against adversarial attacks
# Read more at https://github.com/Trusted-AI/adversarial-robustness-toolbox
from art.attacks.evasion import ProjectedGradientDescent, FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from art.utils import load_cifar10

# we use robustness (https://github.com/MadryLab/robustness) to train, evaluate, and explore neural networks. 
# Read more at https://adversarial-robustness-toolbox.readthedocs.io/en/latest/.
from robustness import model_utils, train, defaults, attacker
from robustness.datasets import CIFAR

from cifar_layer_norm import ResNet, BasicBlock

# We use cox (http://github.com/MadryLab/cox) to log, store and analyze
# results. Read more at https//cox.readthedocs.io.
from cox.utils import Parameters
import cox.store

def seed_everything(seed: int):
    import random
    #initiate seed to try to make the result reproducible 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

#hook to extract input of module
def hook_fn_in(module,inp,outp):
    feature = torch.flatten(inp[0],start_dim=1).detach().cpu().numpy()
    feature_bank.append(feature)

#hook to extract output of module
def hook_fn(module,inp,outp):
    feature = torch.flatten(outp,start_dim=1).detach().cpu().numpy()
    feature_bank.append(feature)
    

def load_model(dataset, weight=None, normalize='nn'):
    #Use ResNet18
    arch = ResNet(BasicBlock, [2,2,2,2], normalize=normalize)
    
    model = attacker.AttackerModel(arch, dataset)

    if weight and os.path.isfile(weight):  # load previously saved model
        # load saved weights
        checkpoint = torch.load(weight, pickle_module=dill, map_location=device)
        
        sd = checkpoint["model"]
        sd = {k[len('module.'):]:v for k,v in sd.items()}
        model.load_state_dict(sd)
        model.eval()
        # print(model.model)
        print("=> loaded checkpoint '{}' (epoch {})".format(weight, checkpoint['epoch']))

        model = model.eval()
    elif weight:
        error_msg = f'=> no checkpoint found at {weight}'
        raise ValueError(error_msg)
    else:  # create new model
        model = model
        # model, _ = model_utils.make_and_restore_model(arch=model, dataset=dataset)
    # model = model.cuda()
    model = model.to(device)
    return model


def main(save_folder, model_name, seed, cluster, mode='train', normalize='nn', weight_decay=0.0, eps=None, attack_mode= 'inf',):
    print(f'Save folder: {save_folder}, model_name: {model_name}, seed: {seed}, mode: {mode}, \
        attack_mode: {attack_mode}, eps: {eps}, normalize: {normalize}, wd: {weight_decay}')
    
    if cluster=='nyu':
        ds = CIFAR(data_path='/scratch/bl3021/research/sy-lab/robust-normalization/datasets/')  # nyu cluster
    if cluster=='flatiron':    
        ds = CIFAR(data_path='/mnt/ceph/users/blyo1/syLab/robust-normalization/datasets/')  # flatiron cluster
        # df = CIFAR(data_path='../../datasets/')
    train_loader, val_loader = ds.make_loaders(batch_size=128, workers=8)
    save_name_base = f"{model_name}-normalize_{normalize}-wd_{weight_decay}-seed_{seed}"
    
    if mode == 'train':
        """
        we train the model using the Robustness library
        """
        save_path = os.path.join(save_folder, 'trained_models', model_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        model = load_model(ds, normalize=normalize)
        save_name = save_name_base
        out_store = cox.store.Store(save_path, exp_id = save_name)  # create a cox Store for saving results of the training (read https://cox.readthedocs.io/en/latest/cox.store.html)
        # Hard-coded base parameters
        train_kwargs = {
            'out_dir': "train_out",
            'adv_train': 0,
            'epochs': 120,
            'step_lr': 40,
            'weight_decay':weight_decay
        }
        
        train_args = Parameters(train_kwargs)
        
        # Fill whatever parameters are missing from the defaults
        train_args = defaults.check_and_fill_args(train_args,
                                defaults.TRAINING_ARGS, CIFAR)
        # train model
        model = train.train_model(train_args, model, (train_loader, val_loader), store=out_store)  


        """
        now we evaluate the same model using ART
        """
        #load cifar dataset
        (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
        x_train = x_train.transpose(0,3,1,2).astype(np.float32)
        x_test = x_test.transpose(0,3,1,2).astype(np.float32)

        mean = np.array([0.4914, 0.4822, 0.4465]).reshape((3, 1, 1))
        std = np.array([0.2023, 0.1994, 0.2010]).reshape((3, 1, 1))
        # we can also get these values using our CIFAR dataset from `robustness.datasets`:
        # mean = ds.mean.reshape((3,1,1))
        # std = ds.std.reshape((3,1,1))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # robustness wraps the model in a few layers -- this will get the main VOneResNet18 for evaluation with ART.
        if hasattr(model, 'module'):
            model = model.module.model
        else:
            model = model.model

        # https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/estimators/classification.html#pytorch-classifier
        classifier = PyTorchClassifier(
            model=model,
            clip_values=(min_pixel_value, max_pixel_value),
            preprocessing=(mean, std),
            loss=criterion,
            optimizer=optimizer,
            input_shape=(3, 32, 32),
            nb_classes=10,
        )

        # eval with clean image
        predictions = classifier.predict(x_test)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        print("Accuracy on benign test examples: {}%".format(accuracy * 100))

    if mode == 'val' or mode == 'extract':
        eps = [float(i) for i in eps]
        eps_cifar = [i/255.0 for i in eps] 
        save_path = os.path.join(save_folder, 'trained_models')
        save_name = os.path.join(save_path, model_name, save_name_base, 'checkpoint.pt.best')
        model = load_model(ds, normalize=normalize, weight=save_name)

        
        #load cifar dataset using ART
        (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
        x_train = x_train.transpose(0,3,1,2).astype(np.float32)
        x_test = x_test.transpose(0,3,1,2).astype(np.float32)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        # robustness wraps the model in a few layers -- this will get the main VOneResNet18 for evaluation with ART.
        if hasattr(model, 'module'):
            model = model.module.model
        else:
            model = model.model
        
        mean = np.array([0.4914, 0.4822, 0.4465]).reshape((3, 1, 1))
        std = np.array([0.2023, 0.1994, 0.2010]).reshape((3, 1, 1))
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
        predictions = classifier.predict(x_test)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        print("Accuracy on benign test examples: {}%".format(accuracy * 100))

        """
        end of extract mode
        """        

        if attack_mode == 'inf':
            norm = np.inf
        else:
            norm = int(attack_mode)
        
        if mode == 'val':
            # to speed things up a bit let's just do evaluation on 1000 images. Final analysis ideally on full test set.
            n_images = 10000
            predictions = classifier.predict(x_test[:n_images])
            accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test[:n_images], axis=1)) / len(y_test[:n_images])
            print("Accuracy on benign test examples: {}%".format(accuracy * 100))

            saved_perf = {}
            saved_perf['clean'] = accuracy
            
            record = np.zeros((len(eps_cifar)))
    
            # scan over k to find a reasonable number of averages
            for ep_idx in range(len(eps)):
                ep = eps_cifar[ep_idx]
                print(f'Attack strength: {ep}', flush=True)
                # generate images with an ensemble, to help w/ noisy gradients
                attack = ProjectedGradientDescent(
                    estimator= classifier,
                    norm = norm,
                    max_iter=32,
                    eps=ep,
                    eps_step=ep/16,
                    targeted=False)
                x_test_adv = attack.generate(x=x_test[:n_images], y=y_test[:n_images])

                # save adversarial examples
                adv_dataset = {}
                adv_dataset['x'] = x_test_adv
                adv_dataset['y'] = y_test[:n_images]
                save_path_eval = os.path.join(save_folder, 'adv_dataset', model_name)
                if not os.path.exists(save_path_eval):
                    os.makedirs(save_path_eval, exist_ok=True)
                adv_save_name = os.path.join(save_path_eval, f'{save_name_base}-eps_{eps[ep_idx]}.pkl')
                pickle.dump(adv_dataset, open(adv_save_name, 'wb'))

                # calculate accuracy of network on adversarial inputs
                predictions = classifier.predict(x_test_adv)
                accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test[:n_images], axis=1)) / len(y_test[:n_images])
                print(f"Accuracy on adversarial test examples: {accuracy*100}",flush=True)
                record[ep_idx] = accuracy
            saved_perf['perturbed'] = record
            
            # save prediction performance
            eps = [str(i) for i in eps]
            save_path_eval = os.path.join(save_folder,'eval_models', model_name)
            if not os.path.exists(save_path_eval):
                os.makedirs(save_path_eval, exist_ok=True)
            save_name = os.path.join(save_path_eval, save_name_base + '-eps_' + '_'.join(eps) + '.pkl')
            pickle.dump(saved_perf, open(save_name,'wb'))

    
if __name__ == '__main__':
    print("we are running!", flush=True)
    parser = argparse.ArgumentParser(description='Run MNIST experiments')
    parser.add_argument('--cluster', help='The cluster you are using, either `nyu` or `flatiron`.')
    parser.add_argument('--save_folder',help='The folder to save model')
    parser.add_argument('--model_name',help='Model name')
    parser.add_argument('--mode', help='Mode to run, choose from (train), (val), (extract)',default='train')
    parser.add_argument('--weight_decay',help='Weight decay for optimizer', type=float)
    parser.add_argument('--seed', help='Fix seed for reproducibility',type=int)
    parser.add_argument('--normalize',help='The type of normalization to use.')
    parser.add_argument('--eps',help='Epsilon for adversarial attack')
    parser.add_argument('--attack_mode',help='Type of adversarial attack. Choose from inf, 1, 2',default='inf')
    # parser.add_argument('--learning_rate',help='The learning rate for the optimizer')

    args = parser.parse_args()
    learning_rate = 0.01
    eps = args.eps.split('_')

    save_folder = os.path.join('..', args.save_folder, 'resnet')
    # if not os.path.exists(save_folder):
        # os.makedirs(save_folder, exist_ok=True)
    seed_everything(args.seed)

    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, flush=True)

    global feature_bank
    feature_bank = []

    main(save_folder, args.model_name, args.seed, mode=args.mode, normalize= args.normalize,
         weight_decay=args.weight_decay, attack_mode=args.attack_mode, eps=eps, cluster=args.cluster)
