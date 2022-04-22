from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import argparse
import os
import pickle
import sys
sys.path.insert(0,'..')

from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist, load_cifar10

from mnist_layer_norm import Net_both, Net_1, Net_2
from robustness.datasets import CIFAR

folder_path = '..'
os.chdir(folder_path)
    
def seed_everything(seed: int):
    import random
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

def main(save_folder, frontend, model_name, seed, lr, wd, mode, eps, norm_method, norm_position, width_scale=1):
    print(f'Save folder: {save_folder}, model_name: {model_name}, frontend: {frontend}, seed: {seed}, mode: {mode}, lr: {lr}, wd: {wd}, norm_method: {norm_method}, norm_position: {norm_position}, width_scale: {width_scale}', flush=True)
    #load and process data
    dataset='cifar'

    if dataset=='mnist':
        (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
        x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
        x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)
    elif dataset=='cifar':
        # ds = CIFAR(data_path='/mnt/ceph/users/blyo1/syLab/robust-normalization/datasets/')  # flatiron cluster
        (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
        x_train = x_train.transpose(0,3,1,2).astype(np.float32)
        x_test = x_test.transpose(0,3,1,2).astype(np.float32)
        mean = np.array([0.4914, 0.4822, 0.4465]).reshape((3, 1, 1))
        std = np.array([0.2023, 0.1994, 0.2010]).reshape((3, 1, 1))
    # train_loader, val_loader = ds.make_loaders(batch_size=128, workers=8)
    
    simple_channels = 16
    complex_channels = 16
    in_channels = int((simple_channels + complex_channels) * width_scale)
    ksize = 5

    if frontend=='learned_conv' or frontend=='frozen_conv':
        conv_1 = nn.Conv2d(in_channels=3, out_channels=in_channels, kernel_size=ksize, stride=2, padding=ksize//2)
        
        if norm_position == 'both':
            model = Net_both(conv_1, in_channels, width_scale, normalize=norm_method)
        if norm_position == '1':
            model = Net_1(conv_1, in_channels, normalize=norm_method)
        if norm_position == '2':
            model = Net_2(conv_1, in_channels, normalize=norm_method)

        if frontend=='frozen_conv':
            # load conv_1 weights from pre-trained model 
            load_path = os.path.join('code', 'saved_model_weights')
            load_name = os.path.join(load_path, f'convnet3-lr_0.01-wd_0.0005-seed_17-normalize_nn.pth')
            
            extracted_weights = torch.load(load_name, map_location=device)
            fixed_weights = {}
            fixed_weights['conv_1.weight'] = extracted_weights['conv_1.weight']
            fixed_weights['conv_1.bias'] = extracted_weights['conv_1.bias']
            
            model.load_state_dict(fixed_weights, strict=False)
            model.conv_1.requires_grad = False
            # also set the requires_grad flag of the weight and bias to False, just in case
            model.conv_1.weight.requires_grad = False
            model.conv_1.bias.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    
    if mode == 'train':    
        classifier = PyTorchClassifier(
            model=model,
            clip_values=(min_pixel_value, max_pixel_value),
            preprocessing=(mean, std),
            loss=criterion,
            optimizer=optimizer,
            input_shape=(1, 28, 28),
            nb_classes=10,
        )

        print('we are now training the model!')
        classifier.fit(x_train, y_train, batch_size=64, nb_epochs=5)
        print('training complete!')

        # to speed things up a bit let's just do evaluation on 1000 images. Final analysis ideally on full test set.
        # n_images = 10000
        print('we are now testing the model!')
        predictions = classifier.predict(x_test)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        print("Accuracy on benign test examples: {}%".format(accuracy * 100))

        model_folder_name = f'{frontend}_frontend-norm_{norm_position}-ws_{width_scale}'
        save_path = os.path.join(save_folder, 'trained_models', model_folder_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        save_name = os.path.join(save_path, f'{model_name}-lr_{lr}-wd_{wd}-seed_{seed}-normalize_{norm_method}-ws_{width_scale}.pth')
        torch.save(classifier.model.state_dict(), save_name)
        record = {}
        record['accuracy'] = accuracy
        norm = calculate_norm(classifier.model)
        record['norm'] = norm
        save_name_record = save_name[:-4] + '.pkl'
        pickle.dump(record,open(save_name_record,'wb'))
        
    if mode == 'val':
        # training_seed = 3  # if I want to hold the training seed constant
        eval_seed = 3  # hold the eval seed constant, while the seed indicates the training seed

        # load the trained weights
        eps = [float(i) for i in eps]
        model_folder_name = f'{frontend}_frontend-norm_{norm_position}-ws_{width_scale}'
        save_path = os.path.join(save_folder, 'trained_models', model_folder_name)
        save_name = os.path.join(save_path, f'{model_name}-lr_{lr}-wd_{wd}-seed_{seed}-normalize_{norm_method}-ws_{width_scale}.pth')
        model.load_state_dict(torch.load(save_name, map_location=device))
        print('device is', device)
        model.eval()
        
        classifier = PyTorchClassifier(
            model=model,
            clip_values=(min_pixel_value, max_pixel_value),
            loss=criterion,
            optimizer=optimizer,
            input_shape=(1, 28, 28),
            nb_classes=10,)
        
        n_images = 1000
        
        predictions = classifier.predict(x_test)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        
        print("Accuracy on benign test examples: {}%".format(accuracy * 100), flush=True)
        
        to_save = {}
        to_save['clean'] = accuracy
        record = np.zeros((len(eps)))
        
        # print(timer() - start)
        print('starting the validation')
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
        # print(timer() - start)
        eps = [str(i) for i in eps]
        save_path_eval = os.path.join(save_folder, 'eval_models', model_folder_name)
        if not os.path.exists(save_path_eval):
            os.makedirs(save_path_eval, exist_ok=True)
        # save_name_eval = os.path.join(save_path_eval, model_name + '-lr_' + str(lr) + '-wd_' + str(wd) + '-ev_seed_' + str(eval_seed) + '-tr_seed_' + str(seed) + 
        # '-normalize_' + norm_method + '-eps_' + '_'.join(eps) + '.pkl')
        eps_str = '_'.join(eps)
        save_name_eval = os.path.join(save_path_eval, f'{model_name}-lr_{lr}-ws_{width_scale}-wd_{wd}-ev_seed_{eval_seed}-tr_seed_{seed}-normalize_{norm_method}-eps_{eps_str}.pkl')
        
        to_save['perturbed'] = record
        save_file = open(save_name_eval, 'wb')
        pickle.dump(to_save, save_file)
        save_file.close()
        print(to_save, flush=True)

if __name__ == '__main__':
    print("we are running!", flush=True)
    # global start
    # start = timer()

    parser = argparse.ArgumentParser(description='Run MNIST experiments on batchNorm, L2-regularizer and noise...')
    parser.add_argument('--save_folder',help='The folder to save model')
    parser.add_argument('--frontend', help='vone_frontend or learned_conv_frontend')
    parser.add_argument('--norm_position', help='instances of normalization, either 1, 2, or both')
    parser.add_argument('--model_name',help='Model name')
    parser.add_argument('--seed', help='Fix seed for reproducibility',type=int)
    # parser.add_argument('--learning_rate', help='Learning rate to train model', type=float)
    parser.add_argument('--weight_decay', help='Amount of weight decay (L2 regularizer)', type=float)
    parser.add_argument('--normalize', help='norm_method Type')
    parser.add_argument('--mode', help='Mode to run, choose from (train), (val), (extract)',default='train')
    parser.add_argument('--eps', help="Adversarial attack strength")
    parser.add_argument('--width_scale', help='scalar parameter for controlling the width of each layer', type=float)

    # torch.autograd.set_detect_anomaly(True)
    args = parser.parse_args()

    # weight_decay=0.0005
    weight_decay=args.weight_decay
    learning_rate=0.01

    eps = args.eps.split("_")
    save_folder = os.path.join(args.save_folder, args.model_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    seed_everything(args.seed)

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(save_folder, args.frontend, args.model_name, args.seed, learning_rate, weight_decay, args.mode, eps, args.normalize, args.norm_position, args.width_scale)
