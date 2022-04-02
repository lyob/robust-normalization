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
sys.path.insert(0,'..')

from art.attacks.evasion import ProjectedGradientDescent, FastGradientMethod
from art.estimators.classification import PyTorchClassifier, EnsembleClassifier
from art.utils import load_mnist

from mnist_layer_norm import Net, Net_1, Net_2
from vonenet.vonenet import VOneNet

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
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms()
    
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

def main(save_folder, frontend, model_name, seed, lr, wd, mode, eps, norm_method, norm_position):
    print(f'Save folder: {save_folder}, model_name: {model_name}, frontend: {frontend}, seed: {seed}, mode: {mode}, lr: {lr}, wd: {wd}, norm_method: {norm_method}, norm_position: {norm_position}', flush=True)
    #load and process data
    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

    x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
    x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)
    simple_channels = 16
    complex_channels = 16
    ksize = 5
    
    if model_name == 'standard':
        conv_1 = nn.Conv2d(in_channels=1, out_channels=simple_channels+complex_channels, kernel_size=ksize, stride=2, padding=ksize//2)
        model = Net(conv_1, simple_channels + complex_channels, normalize=norm_method)

    if model_name == 'convnet' or model_name == 'convnet2' or model_name == 'convnet3':
        if frontend=='vone_filterbank':
            model = VOneNet(simple_channels=simple_channels, complex_channels=complex_channels, norm_method=norm_method)
        
        elif frontend=='learned_conv' or frontend=='frozen_conv':
            conv_1 = nn.Conv2d(in_channels=1, out_channels=simple_channels+complex_channels, kernel_size=ksize, stride=2, padding=ksize//2)
            
            if norm_position == 'both':
                model = Net(conv_1, simple_channels + complex_channels, normalize=norm_method)
            elif norm_position == '1':
                model = Net_1(conv_1, simple_channels + complex_channels, normalize=norm_method)
            elif norm_position == '2':
                model = Net_2(conv_1, simple_channels + complex_channels, normalize=norm_method)

            if frontend=='frozen_conv':
                # load conv_1 weights from pre-trained model 
                load_path = os.path.join('code', 'saved_model_weights')
                load_name = os.path.join(load_path, f'convnet-lr_0.01-wd_0.0005-seed_1-normalize_nn.pth')
                
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

        model_folder_name = f'{frontend}_frontend-norm_{norm_position}'
        save_path = os.path.join(save_folder, 'trained_models', model_folder_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        save_name = os.path.join(save_path, f'{model_name}-lr_{str(lr)}-wd_{str(wd)}-seed_{str(seed)}-normalize_{norm_method}.pth')
        torch.save(classifier.model.state_dict(), save_name)
        record = {}
        record['accuracy'] = accuracy
        norm = calculate_norm(classifier.model)
        record['norm'] = norm
        save_name_record = save_name[:-4] + '.pkl'
        pickle.dump(record,open(save_name_record,'wb'))
        
    if mode == 'val':
        eps = [float(i) for i in eps]
        model_folder_name = f'{frontend}_frontend-norm_{norm_position}'
        save_path = os.path.join(save_folder, 'trained_models', model_folder_name)
        save_name = os.path.join(save_path, f'{model_name}-lr_{str(lr)}-wd_{str(wd)}-seed_{str(seed)}-normalize_{norm_method}.pth')
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
        save_path_eval = os.path.join(save_folder, 'eval_models', model_folder_name)
        if not os.path.exists(save_path_eval):
            os.makedirs(save_path_eval, exist_ok=True)
        save_name_eval = os.path.join(save_path_eval, model_name + '-lr_' + str(lr) + '-wd_' + str(wd) + '-seed_' + str(seed) + 
        '-normalize_' + norm_method + '-eps_' + '_'.join(eps) + '.pkl')
        
        to_save['perturbed'] = record
        save_file = open(save_name_eval, 'wb')
        pickle.dump(to_save, save_file)
        save_file.close()
        print(to_save, flush=True)

if __name__ == '__main__':
    print("we are running!", flush=True)
    parser = argparse.ArgumentParser(description='Run MNIST experiments on batchNorm, L2-regularizer and noise...')
    parser.add_argument('--save_folder',help='The folder to save model')
    parser.add_argument('--frontend', help='vone_frontend or learned_conv_frontend')
    parser.add_argument('--norm_position', help='instances of normalization, either 1, 2, or both')
    parser.add_argument('--model_name',help='Model name')
    parser.add_argument('--seed', help='Fix seed for reproducibility',type=int)
    # parser.add_argument('--learning_rate', help='Learning rate to train model', type=float)
    # parser.add_argument('--weight_decay', help='Amount of weight decay (L2 regularizer)', type=float)
    parser.add_argument('--normalize', help='norm_method Type')
    parser.add_argument('--mode', help='Mode to run, choose from (train), (val), (extract)',default='train')
    parser.add_argument('--eps', help="Adversarial attack strength")

    torch.autograd.set_detect_anomaly(True)

    weight_decay=0.0005
    learning_rate=0.01

    args = parser.parse_args()
    eps = args.eps.split("_")
    save_folder = os.path.join(args.save_folder, args.model_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    seed_everything(args.seed)

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(save_folder, args.frontend, args.model_name, args.seed, learning_rate, weight_decay, args.mode, eps, args.normalize, args.norm_position)
