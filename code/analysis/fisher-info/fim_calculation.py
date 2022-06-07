#%%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import os

from art.utils import load_mnist

code_root = os.path.abspath('/mnt/ceph/users/blyo1/syLab/robust-normalization/code/')
os.chdir(code_root)
print('the root directory is', os.path.abspath('.'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% import the model I want to analyze
from mnist_layer_norm import Net_both, Net_1, Net_2
simple_channels = 16
complex_channels = 16
ksize = 5

model_name = 'convnet4'
frontend = 'learned_conv'  # learned_conv or vone_filterbank or frozen_conv
norm_position = 'both'
seed = 17
norm_method = 'nn'
lr = 0.01
wd = 0.0005

# we can do this by comparing the frozen weights against the pre-trained weights
conv_1 = nn.Conv2d(in_channels=1, out_channels=simple_channels+complex_channels, kernel_size=ksize, stride=2, padding=ksize//2)
lenet = Net_both(conv_1, simple_channels + complex_channels, normalize=norm_method)

#%% load the learned weights into the model
model_folder_name = f'{frontend}_frontend-norm_{norm_position}'
load_folder = os.path.join('..', 'results', model_name, 'trained_models', model_folder_name)
load_name = os.path.join(load_folder, f'{model_name}-lr_{lr}-wd_{wd}-seed_{seed}-normalize_{norm_method}.pth')
lenet.load_state_dict(torch.load(load_name, map_location=device))
lenet.eval()

#%% load the dataset
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
x_train_input = np.swapaxes(x_train, 1, 3)
x_test_input = np.swapaxes(x_test, 1, 3)

# %% calculate the FIM
import warnings

def jacobian(y: torch.Tensor, x: torch.Tensor):
    if x.numel() > 1E4:
        warnings.warn("Calculation of Jacobian with input dimensionality greater than 1E4 may take too long; consider"
                      "an iterative method (e.g. power method, randomized svd) instead.")

    J = torch.stack([torch.autograd.grad([y[i].sum()], [x], retain_graph=True, create_graph=True)[0] for i in range(
        y.size(0))], dim=-1).squeeze().t()

    if y.shape[0] == 1:  # need to return a 2D tensor even if y dimensionality is 1
        J = J.unsqueeze(0)

    return J.detach()    

class Eigendistortion:
    def __init__(self, base_signal: torch.Tensor, model: torch.nn.Module):
        assert len(base_signal.shape) == 4, "Input must be torch.Size([batch=1, n_channels, im_height, im_width])"
        assert base_signal.shape[0] == 1, "Batch dim must be 1. Image batch synthesis is not available."

        self.batch_size, self.n_channels, self.im_height, self.im_width = base_signal.shape

        self.model = model
        # flatten and attach gradient and reshape to image
        self._input_flat = base_signal.flatten().unsqueeze(1).requires_grad_(True)

        self.base_signal = self._input_flat.view(*base_signal.shape)
        self.base_representation = self.model(self.base_signal)

        if len(self.base_representation) > 1:
            self._representation_flat = torch.cat([s.squeeze().view(-1) for s in self.base_representation]).unsqueeze(1)
        else:
            self._representation_flat = self.base_representation.squeeze().view(-1).unsqueeze(1)

        print(f"\nInitializing Eigendistortion -- "
              f"Input dim: {len(self._input_flat.squeeze())} | Output dim: {len(self._representation_flat.squeeze())}")

        self.jacobian = None
        self.synthesized_signal = None  # eigendistortion
        self.synthesized_eigenvalues = None
        self.synthesized_eigenindex = None

    def compute_jacobian(self):
        if self.jacobian is None:
            J = jacobian(self._representation_flat, self._input_flat)
            self.jacobian = J
        else:
            print("Jacobian already computed, returning self.jacobian")
            J = self.jacobian
        return J

    def _synthesize_exact(self):
        # compute exact Jacobian
        J = self.compute_jacobian()
        F = J.T @ J
        eig_vals, eig_vecs = torch.linalg.eigh(F, UPLO="U")
        eig_vecs = eig_vecs.flip(dims=(1,))
        eig_vals = eig_vals.flip(dims=(0,))
        return eig_vals, eig_vecs

    def synthesize(self, 
                   method: str = 'exact',
                   seed: int = None):

        if seed is not None:
            assert isinstance(seed, int), "random seed must be integer"
            torch.manual_seed(seed)

        if method == 'exact':
            print(f"Computing all eigendistortions")
            eig_vals, eig_vecs = self._synthesize_exact()
            eig_vecs = self._vector_to_image(eig_vecs.detach())
            eig_vecs_ind = torch.arange(len(eig_vecs))

        # reshape to (n x num_chans x h x w)
        self.synthesized_signal = torch.stack(eig_vecs, 0) if len(eig_vecs) != 0 else []

        self.synthesized_eigenvalues = torch.abs(eig_vals.detach())
        self.synthesized_eigenindex = eig_vecs_ind

        return self.synthesized_signal, self.synthesized_eigenvalues, self.synthesized_eigenindex

    def _vector_to_image(self, vecs):
        imgs = [vecs[:, i].reshape((self.n_channels, self.im_height, self.im_width)) for i in range(vecs.shape[1])]
        return imgs

# %% create a class that takes the nth layer output of a given model
class NthLayer(torch.nn.Module):
    '''Wrap any model to get the response of an intermediate Layer.
    '''
    
    def __init__(self, model, norm_method, layer=None):
        '''
        Parameters
        ----------
        model: PyTorch model
        layer: int, which model response layer to output
        '''
        super().__init__()
        try:
            # vgg
            features = list(model.features)
        except:
            if not hasattr(model, 'conv1'):
                # lenet
                nm1 = {
                    'bn': model.bn1,
                    'gn': model.gn1,
                    'in': model.in1,
                    'ln': model.ln1,
                    'lrnc': model.lrn_channel,
                    'lrnb': model.lrn_both,
                    'lrns': model.lrn_spatial,
                    'nn': nn.Identity(),
                }
                nm2 = {
                    'bn': model.bn2,
                    'gn': model.gn2,
                    'in': model.in2,
                    'ln': model.ln2,
                    'lrnc': model.lrn_channel,
                    'lrnb': model.lrn_both,
                    'lrns': model.lrn_spatial,
                    'nn': nn.Identity(),
                }
                features = (
                    [model.conv_1] +
                    [nm1[norm_method]] +
                    [model.relu] +
                    [model.conv_2] + 
                    [nm2[norm_method]] + 
                    [model.relu] + 
                    [model.fc_1]
                )
            else:
                # resnet
                features = (
                    [model.conv1, model.bn1, model.relu, model.maxpool] + 
                    [l for l in model.layer1] + 
                    [l for l in model.layer2] + 
                    [l for l in model.layer3] + 
                    [l for l in model.layer4] + 
                    [model.avgpool, model.fc]
                )
            self.features = nn.ModuleList(features).eval()

        if layer is None:
            # default is last layer if unspecified
            layer = len(self.features)
        self.layer = layer

    def forward(self, x):
        for ii, mdl in enumerate(self.features):
            x = mdl(x)
            if ii == self.layer:
                return x

lenet_1 = NthLayer(lenet, 'nn', layer=1)
# lenet_2 = NthLayer(lenet, layer=2)

#%% select an image to feed into the model
img = x_test[1].squeeze(2)
print('this is the image we will feed into the network:')
plt.imshow(img)
plt.colorbar()
plt.show()

#%% find the eigendistortions for the model
input_img = torch.Tensor(x_test_input[1]).unsqueeze(0)
ed_lenet_1 = Eigendistortion(input_img, lenet_1)
ed_lenet_2 = Eigendistortion(input_img, lenet_2)

# %% synthesize the eigenvalues and eigenvectors
eigdist1, eigval1, eigind1 = ed_lenet_1.synthesize()
eigdist2, eigval2, eigind2 = ed_lenet_2.synthesize()


# %% plot the eigendistortions
eigdist1 = eigdist1.view(eigdist1.size()[0], -1)

fig, ax = plt.subplots(1, 2)
# mat = ax[0].imshow(eigdist[:100, :100], vmin=-0.1, vmax=0.1, cmap='coolwarm')
mat = ax[0].imshow(eigdist1[200:300, :100], vmin=-0.1, vmax=0.1, cmap='coolwarm')
ax[0].set(title='Eigendistortions', xlabel='Eigenvector index', ylabel='Entry')
ax[1].plot(eigval1, '.')
ax[1].set(title='Eigenvalues', xlabel='Eigenvector index', ylabel='Eigenvalue')
fig.colorbar(mat, ax=ax[0], location='right', anchor=(0, 0.5), shrink=0.5)
fig.tight_layout()

fig, ax = plt.subplots(1, 2)
# mat = ax[0].imshow(eigdist[:100, :100], vmin=-0.1, vmax=0.1, cmap='coolwarm')
ax[0].plot(eigval1, '.')
ax[0].set(title='Eigenvalues layer 1', xlabel='Eigenvector index', ylabel='Eigenvalue')
ax[1].plot(eigval2, '.')
ax[1].set(title='Eigenvalues layer 2', xlabel='Eigenvector index', ylabel='Eigenvalue')
fig.tight_layout()


#%% calculate the volume of the eigenvalues
ev_vol1 = torch.prod(eigval1[:10])
ev_vol2 = torch.prod(eigval2[:10])
print(f'layer 1: {ev_vol1}, layer 2: {ev_vol2}')

ev_sum1 = torch.sum(eigval1)
ev_sum2 = torch.sum(eigval2)
print(f'layer 1: {ev_sum1}, layer 2: {ev_sum2}')



# %%
