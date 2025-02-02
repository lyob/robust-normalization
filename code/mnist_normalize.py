import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import argparse
import os
import pickle
import h5py as h5
import sys
import random
sys.path.insert(0,'..')

from art.attacks.evasion import ProjectedGradientDescent, FastGradientMethod
from art.estimators.classification import PyTorchClassifier, EnsembleClassifier
from art.utils import load_mnist

from mnist_layer_norm import Net
from vonenet.vonenet import VOneNet

folder_path = '../results'
os.chdir(folder_path)
    
def seed_everything(seed):
    #initiate seed to try to make the result reproducible 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def calculate_norm(model):
    norm_dict = {}
    for name, param in model.named_parameters():
        if name == 'conv_2.weight' or name == 'fc_1.weight':
            param = param.detach().cpu().numpy()
            if name == 'conv_2.weight':
                chan_out, chan_in, h, w = param.shape
                param = np.reshape(param, (chan_out, chan_in, h*w))
                norm_dict[name] = np.sum(np.sqrt(np.sum(param**2, axis=2)))
            else:
                norm_dict[name] = np.sum(np.sqrt(np.sum(param**2, axis=1)))
    return norm_dict

def main(save_folder, model_name, seed, lr, wd, mode, eps, normalize):
    print(f'Save folder: {save_folder}, model_name: {model_name}, seed: {seed}, mode: {mode}, lr: {lr}, wd: {wd}, normalize: {normalize}', flush=True)
    seed_everything(seed)
    #load and process data
    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

    x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
    x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)
    simple_channels = 16
    complex_channels = 16
    ksize = 5
    
    if model_name == 'standard':
        conv_1 = nn.Conv2d(in_channels=1, out_channels=simple_channels+complex_channels, 
        kernel_size=ksize, stride=2, padding=ksize//2)
        model = Net(conv_1, simple_channels + complex_channels, normalize=normalize)

    if model_name == 'vone_convnet':
        model = VOneNet(simple_channels=simple_channels, complex_channels=complex_channels, normalize=normalize)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    if mode == 'train':    
        classifier = PyTorchClassifier(
            model=model,
            clip_values=(min_pixel_value, max_pixel_value),
            loss=criterion,
            optimizer=optimizer,
            input_shape=(1, 28, 28),
            nb_classes=10,
        )
    
        classifier.fit(x_train, y_train, batch_size=64, nb_epochs=5)
  
        # to speed things up a bit let's just do evaluation on 1000 images. Final analysis ideally on full test set.
        n_images = 10000
        predictions = classifier.predict(x_test[:n_images])
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test[:n_images], axis=1)) / len(y_test[:n_images])
        print("Accuracy on benign test examples: {}%".format(accuracy * 100))

        save_path = os.path.join(save_folder, 'trained_models', model_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        save_name = os.path.join(save_path, f'{model_name}-lr_{str(lr)}-wd_{str(wd)}-seed_{str(seed)}-normalize_{normalize}.pth')
        torch.save(classifier.model.state_dict(),save_name)
        record = {}
        record['accuracy'] = accuracy
        norm = calculate_norm(classifier.model)
        record['norm'] = norm
        save_name_record = save_name[:-4] + '.pkl'
        pickle.dump(record,open(save_name_record,'wb'))
        
    if mode == 'val':
        eps = [float(i) for i in eps]
        save_path = os.path.join(save_folder, 'trained_models', model_name)
        save_name = os.path.join(save_path, f'{model_name}-lr_{str(lr)}-wd_{str(wd)}-seed_{str(seed)}-normalize_{normalize}.pth')
        model.load_state_dict(torch.load(save_name))
        model.eval()
        
        classifier = PyTorchClassifier(
            model=model,
            clip_values=(min_pixel_value, max_pixel_value),
            loss=criterion,
            optimizer=optimizer,
            input_shape=(1, 28, 28),
            nb_classes=10,)
        
        n_images = 10000
        predictions = classifier.predict(x_test)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        print("Accuracy on benign test examples: {}%".format(accuracy * 100), flush=True)
        
        to_save = {}
        to_save['clean'] = accuracy
        record = np.zeros((len(eps)))
        
        for ep_idx in range(len(eps)):
            ep = eps[ep_idx]
            print("Attack Strength: ", ep, flush=True)
            attack = ProjectedGradientDescent(estimator=classifier,
                                              norm=np.inf,
                                              max_iter=50,
                                              eps=ep,
                                              eps_step=ep/32,
                                              targeted=False)
            x_test_adv = attack.generate(x=x_test[:n_images], y=y_test[:n_images])
            
            predictions = classifier.predict(x_test_adv)
            accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test[:n_images], axis=1)) / len(y_test[:n_images])
            print(f"Accuracy on adversarial test examples: {accuracy*100}",flush=True)
            record[ep_idx] = accuracy
        eps = [str(i) for i in eps]
        save_path_eval = os.path.join(save_folder, 'eval_models', model_name)
        if not os.path.exists(save_path_eval):
            os.makedirs(save_path_eval, exist_ok=True)
        save_name_eval = os.path.join(save_path_eval, model_name + '-lr_' + str(lr) + '-wd_' + str(wd) + '-seed_' + str(seed) + 
        '-normalize_' + normalize + '-eps_' + '_'.join(eps) + '.pkl')
        
        to_save['perturbed'] = record
        save_file = open(save_name_eval, 'wb')
        pickle.dump(to_save, save_file)
        save_file.close()
        print(to_save, flush=True)

if __name__ == '__main__':
    print("we are running!", flush=True)
    parser = argparse.ArgumentParser(description='Run MNIST experiments on batchNorm, L2-regularizer and noise...')
    parser.add_argument('--save_folder',help='The folder to save model')
    parser.add_argument('--model_name',help='Model name')
    parser.add_argument('--seed', help='Fix seed for reproducibility',type=int)
    parser.add_argument('--learning_rate', help='Learning rate to train model', type=float)
    parser.add_argument('--weight_decay', help='Amount of weight decay (L2 regularizer)', type=float)
    parser.add_argument('--normalize', help='Normalize Type')
    parser.add_argument('--mode', help='Mode to run, choose from (train), (val), (extract)',default='train')
    parser.add_argument('--eps', help="Adversarial attack strength")

    args = parser.parse_args()
    eps = args.eps.split("_")    
    save_folder = os.path.join(args.save_folder, 'mnist_regularize')
    main(save_folder, args.model_name, args.seed, args.learning_rate, args.weight_decay, args.mode, eps, args.normalize)
