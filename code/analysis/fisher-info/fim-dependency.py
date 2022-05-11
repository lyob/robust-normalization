'''
this code is to investigate how the determinant of the FIM changes with functions and other parameters
'''
#%%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

#%% define and run one layer linear network
class LinearModelOneLayer(nn.Module):
    '''The simplest model we can make. Here the Jacobian will be the weight matrix M'''
    def __init__(self, n, m):
        super().__init__()
        torch.manual_seed(0)
        self.M = nn.Linear(n, m, bias=False)
    
    def forward(self, x):
        y = self.M(x)  # this computes y = x @ M.T
        return y

n = 25  # input vector dimension
m = 10  # output vector dimension

mdl_linear = LinearModelOneLayer(n, m)

# example input
x0 = torch.ones((1, 1, 1, n))
# output on random (initial) weights
y0 = mdl_linear(x0)

fig, ax = plt.subplots(2, 1, sharex='all', sharey='all')
ax[0].stem(x0.squeeze(), use_line_collection=True)
ax[0].set(title=f'{n:d}D Input')

ax[1].stem(y0.squeeze().detach(), use_line_collection=True, markerfmt='C1o')
ax[1].set(title=f'{m:d}D Output')

fig.tight_layout()

#%% extract the eigenvalues of the FIM of the layer wrt inputs
mdl_linear.eval()
M = mdl_linear.M.weight.detach()
# print(M)

# plot weight matrix
fig, ax = plt.subplots(4, 1, figsize=(5, 12))
ax[0].imshow(M)
ax[0].set(title=f'weight matrix of 1 layer linear model')

# turn weight matrix into vector
def vectorize(M):
    return M.view(n*m, -1).squeeze().numpy()
N = vectorize(M)

# weight distribution
_, _, _ = ax[1].hist(N, 20)
ax[1].set(title=f'distribution of weights')

# calculate the FIM
F = M.T @ M  # should have shape (n, n)
ax[2].imshow(F)
ax[2].set(title=f'FIM')

# eigendecomposition of fisher info matrix
def eigdecomp(F):
    eig_vals, eig_vecs = torch.linalg.eigh(F, UPLO="U")
    eig_vals = eig_vals.flip(dims=(0,))
    eig_vals = eig_vals.numpy()
    return eig_vals
eig_vals = eigdecomp(F)

# plot eigenspectrum
ax[3].plot(eig_vals, '.' )
ax[3].set(title=f'eigenvalues of the FIM')

# calculate the volume of the FIM

# calculate the participation ratio of the FIM eigenvalues
def calc_PR(eig_vals, inverse=False):
    ev_sum_sq = np.sum(eig_vals)**2
    ev_sq_sum = np.sum(eig_vals**2)
    if inverse:
        return ev_sq_sum / ev_sum_sq
    return  ev_sum_sq / ev_sq_sum

def calc_NPR(eig_vals, inverse=False):
    pr = calc_PR(eig_vals, inverse=inverse)
    dim = len(eig_vals)
    return pr / dim

def calc_measures(eig_vals, output_dim):
    volumes = {}
    volumes['eigval_prod'] = np.prod(eig_vals[:output_dim])
    volumes['max_eigval'] = eig_vals[0]
    volumes['eigval_sum'] = np.sum(eig_vals)
    volumes['eigval_sum_normalized'] = np.sum(eig_vals) / output_dim
    volumes['pr'] = calc_PR(eig_vals)
    volumes['npr'] = calc_NPR(eig_vals) 
    volumes['log_det'] = np.sum(np.log2(eig_vals[:output_dim]))
    return volumes

volumes = calc_measures(eig_vals, m)
print(volumes)

fig.tight_layout()

#%% applying scale transformations to the representation
def scale(weights, amplitude):
    return weights * amplitude

# Since this is a linear network, we can simply multiply the input by the weight matrix to get the output
y1 = x0 @ M.T
y1 == y0  # True

M_scale2 = scale(M, 2)
y_scale2 = x0 @ M_scale2.T

print(f'The output elements are scaled by 2:', torch.allclose((y_scale2 / y1).squeeze(), torch.ones(len(y1))*2))

F_scale2 = M_scale2.T @ M_scale2
ev_scale2 = eigdecomp(F_scale2)

fig, ax = plt.subplots(2, 1)
ax[0].plot(eig_vals, '.')
ax[0].set(title=f'eigenvalues of the FIM')
ax[1].plot(ev_scale2, '.' )
ax[1].set(title=f'eigenvalues of the scaled FIM')
fig.tight_layout()

# check that the decay rate in the new eigenspectrum is the same as the old one
ratios_old = eig_vals[1:] / eig_vals[0]
ratios_new = ev_scale2[1:] / ev_scale2[0]
print(f'the relationship between the max eig and all the other eigvals are preserved:', np.allclose(ratios_old, ratios_new))

print(f'The new eigenvalues are scaled by {ev_scale2[0] / eig_vals[0]}')
# this makes sense because any scaled FIM is a product of 2 weight matrices, which are both scaled by 2.

#%% let's see if this holds across a range of scale values
fig, ax = plt.subplots(3, 2, figsize=(12, 10), sharex='all')

for s in range(1, 10, 1):
    Ms = scale(M, s)
    Fs = Ms.T @ Ms
    evs = eigdecomp(Fs)
    vols = calc_measures(evs, m)
    print(vols['eigval_sum_normalized'])
    ax[0,0].plot(s, vols['max_eigval'], '.', color='blue')
    ax[1,0].plot(s, vols['eigval_sum'], '.', color='blue')
    ax[2,0].plot(s, vols['eigval_sum_normalized'], '.', color='blue')
    ax[0,1].plot(s, vols['pr'], '.', color='blue')
    ax[1,1].plot(s, vols['npr'], '.', color='blue')
    ax[2,1].plot(s, vols['log_det'], '.', color='blue')

xvals = np.arange(1, 9, 0.001)
yvals_maxeig = 0.7987494*(xvals**2)
yvals_npr = 0.2871488*np.ones(len(xvals))
yvals_pr = 7.17872*np.ones(len(xvals))
yvals_sum = 3.2836154*(xvals**2)
yvals_sum_norm = 0.328361*(xvals**2)
yvals_log_det = 20*np.log2(xvals) - 19

ax[0,0].set(title=f'max eigval')
ax[0,0].plot(xvals, yvals_maxeig, alpha=0.3, color='orange', label='$y = A_0 \cdot x^2$')
ax[0,0].legend()

ax[1,0].set(title=f'sum of eigvals')
ax[1,0].plot(xvals, yvals_sum, alpha=0.3, color='orange', label='$y = D_0 \cdot x^2$')
ax[1,0].legend()

ax[2,0].set(title=f'normalized sum of eigvals')
ax[2,0].plot(xvals, yvals_sum_norm, alpha=0.3, color='orange', label='$y = E_0 \cdot x^2$')
ax[2,0].legend()

ax[0,1].set(title=f'PR', ylim=[0., 12])
ax[0,1].plot(xvals, yvals_pr, alpha=0.3, color='orange', label='$y = C_0$')
ax[0,1].legend()

ax[1,1].set(title=f'normalized PR', ylim=[0, 0.5])
ax[1,1].plot(xvals, yvals_npr, alpha=0.3, color='orange', label='$y = B_0$')
ax[1,1].legend()

ax[2,1].set(title=f'log determinant')
ax[2,1].plot(xvals, yvals_log_det, alpha=0.3, color='orange', label='$y = F_0 \cdot \ln(x) + G_0$')
ax[2,1].legend()

fig.suptitle('how eigenvalue measures change with the size of scale transformations')
fig.tight_layout()

#%% number of layer neurons and see how that affects the measures
weights = {}
evals = {}

n = 25  # input vector stays the same
for m in range(5, 41, 5):
    mdl_linear = LinearModelOneLayer(n, m)
    W = mdl_linear.M.weight.detach()
    F = W.T @ W
    weights[m] = F

    ev = eigdecomp(F)
    evals[m] = ev

fig, ax = plt.subplots(8, 1, figsize=(5, 10), sharex='all')
for i in range(1, 9, 1):
    ax[i-1].plot(evals[5*i])
    ax[i-1].set(title=f'n={i*5}')
fig.suptitle('eigenspectrum dependence on layer dimension')
fig.tight_layout()

fig, ax = plt.subplots(8, 1, figsize=(5, 10), sharex='all')
for i in range(1, 9, 1):
    ax[i-1].plot(evals[5*i])
    ax[i-1].set(title=f'n={i*5}', ylim=[0, 2])
fig.suptitle('eigenspectrum vs layer dimension, sharing y axis')
fig.tight_layout()

#%% how do the volume measures scale with neuron number?
fig, ax = plt.subplots(3, 2, figsize=(12, 10), sharex='all')
for i in range(5, 41, 5):
    vols = calc_measures(evals[i], i)  # the value of m changes  
    # print(vols['npr'])
    ax[0,0].plot(i, vols['max_eigval'], '.', color='blue')
    ax[1,0].plot(i, vols['eigval_sum'], '.', color='blue')
    ax[2,0].plot(i, vols['eigval_sum_normalized'], '.', color='blue')
    ax[0,1].plot(i, vols['pr'], '.', color='blue')
    ax[1,1].plot(i, vols['npr'], '.', color='blue')
    ax[2,1].plot(i, vols['log_det'], '.', color='blue')

xvals = np.arange(5, 40, 0.001)
yvals_maxeig = 0.034*xvals + 0.45
yvals_sum = 0.34*xvals
yvals_sum_norm = 0.338 * np.ones(len(xvals))
# yvals_npr = 0.145*np.log2(0.41*xvals)
yvals_npr = 0.145*np.log2(xvals) - 0.18
# yvals_pr = 3.7*np.log2(0.4*xvals)
yvals_pr = 3.7*np.log2(xvals) - 4.9

ax[0,0].set(title=f'max eigval')
ax[0,0].plot(xvals, yvals_maxeig, alpha=0.3, color='orange', label='$y = A_0 x + A_1$')
ax[0,0].legend()

ax[1,0].set(title=f'sum of eigvals')
ax[1,0].plot(xvals, yvals_sum, alpha=0.3, color='orange', label='$y = B_0 x$')
ax[1,0].legend()

ax[2,0].set(title=f'normalized sum of eigvals', ylim=[0.2, 0.45])
ax[2,0].plot(xvals, yvals_sum_norm, alpha=0.3, color='orange', label='$y = C_0$')
ax[2,0].legend()

ax[0,1].set(title=f'PR')
ax[0,1].plot(xvals, yvals_pr, alpha=0.3, color='orange', label='$y = E_0 \ \ln(x) + E_1$')
ax[0,1].legend()

ax[1,1].set(title=f'normalized PR')
ax[1,1].plot(xvals, yvals_npr, alpha=0.3, color='orange', label='$y = D_0 \ \ln(x) + D_1$')
ax[1,1].legend()

ax[2,1].set(title=f'log determinant')

fig.suptitle('how eigenvalue measures change with the layer dimension')
fig.tight_layout()

#%% expand the network to 2, 3 layers and calculate each layer's sensitivity to the input
# this is equivalent to y = (x @ M1.T) @ M2.T
class LinearModelTwoLayers(nn.Module):
    '''The simplest model we can make. Here the Jacobian will be the weight matrix M'''
    def __init__(self, n, m, l):
        super().__init__()
        torch.manual_seed(0)
        self.M1 = nn.Linear(n, m, bias=False)
        self.M2 = nn.Linear(m, l, bias=False)
    
    def forward(self, x):
        x = self.M1(x)  # this computes x = x @ M.T
        y = self.M2(x)  # this computes y = x @ M.T
        return y

n = 25  # input dim
m = 10  # intermediate dim
l = 10  # output dim
mdl_2l = LinearModelTwoLayers(n, m, l)

# example input
x0 = torch.ones((1, 1, 1, n))
# output on random (initial) weights
y0 = mdl_2l(x0)

fig, ax = plt.subplots(2, 1, sharex='all', sharey='all')
ax[0].stem(x0.squeeze(), use_line_collection=True)
ax[0].set(title=f'{n:d}D Input')

ax[1].stem(y0.squeeze().detach(), use_line_collection=True, markerfmt='C1o')
ax[1].set(title=f'{l:d}D Output')

fig.tight_layout()

#%% extract eigvals of the FIM of each layer wrt inputs
mdl_2l.eval()
M1 = mdl_2l.M1.weight.detach()
M2 = mdl_2l.M2.weight.detach()
Mboth = M2 @ M1
# print(torch.allclose(((x0 @ M1.T) @ M2.T), x0 @ (M2 @ M1).T))

fig, ax = plt.subplots(3, 2, figsize=(10, 10))
ax[0,0].imshow(M1)
ax[0,0].set(title=f'1st weight matrix of 2 layer linear model')
ax[0,1].imshow(Mboth)
ax[0,1].set(title=f'2nd weight matrix of 2 layer linear model')

F1 = M1.T @ M1  # should have shape (n, n)
Fboth = Mboth.T @ Mboth  # should have shape (n, n)
ax[1,0].imshow(F1)
ax[1,0].set(title=f'FIM of first layer wrt input')
ax[1,1].imshow(Fboth)
ax[1,1].set(title=f'FIM of both layers wrt input')

eigvals1 = eigdecomp(F1)
eigvals_both = eigdecomp(Fboth)

ax[2,0].plot(eigvals1, '.' )
ax[2,0].set(title=f'eigvals of the 1st layer FIM')
ax[2,1].plot(eigvals_both, '.' )
ax[2,1].set(title=f'eigvals of the FIM for both layers')

fig.tight_layout()

#%% calculate measures
volumes1 = calc_measures(eigvals1, m)
volumes_both = calc_measures(eigvals_both, l)
print(volumes1)
print(volumes_both)

#%% scale transformations
volumes1 = {}
volumesb = {}
fig, ax = plt.subplots(6, 2, figsize=(12, 12), sharex='all')
for s in range(1, 10, 1):
    M1s = scale(M1, s)
    Mbs = scale(Mboth, s)
    
    F1s = M1s.T @ M1s
    Fbs = Mbs.T @ Mbs

    ev1 = eigdecomp(F1s)
    evb = eigdecomp(Fbs)

    vol1 = calc_measures(ev1, m)
    volb = calc_measures(evb, l)

    volumes1[s] = vol1
    volumesb[s] = volb

    # print(volb['npr'])

    ax[0,0].plot(s, vol1['max_eigval'], '.', color='blue')
    ax[0,0].set(title=f'max eigenvalue')
    ax[1,0].plot(s, vol1['eigval_sum'], '.', color='blue')
    ax[1,0].set(title=f'sum of eigenvalues')
    ax[2,0].plot(s, vol1['eigval_sum_normalized'], '.', color='blue')
    ax[2,0].set(title=f'sum of eigenvalues, normalized')
    ax[3,0].plot(s, vol1['pr'], '.', color='blue')
    ax[3,0].set(title=f'PR', ylim=[4.5, 10])
    ax[4,0].plot(s, vol1['npr'], '.', color='blue')
    ax[4,0].set(title=f'normalized PR', ylim=[0.2, 0.4])
    ax[5,0].plot(s, vol1['log_det'], '.', color='blue')
    ax[5,0].set(title=f'log determinant, normalized')
    
    ax[0,1].plot(s, volb['max_eigval'], '.', color='blue')
    ax[0,1].set(title=f'max eigenvalue')
    ax[1,1].plot(s, volb['eigval_sum'], '.', color='blue')
    ax[1,1].set(title=f'sum of eigenvalues')
    ax[2,1].plot(s, volb['eigval_sum_normalized'], '.', color='blue')
    ax[2,1].set(title=f'sum of eigenvalues, normalized')
    ax[3,1].plot(s, volb['pr'], '.', color='blue')
    ax[3,1].set(title=f'PR', ylim=[4, 5])
    ax[4,1].plot(s, volb['npr'], '.', color='blue')
    ax[4,1].set(title=f'normalized PR', ylim=[0, 0.4])
    ax[5,1].plot(s, volb['log_det'], '.', color='blue')
    ax[5,1].set(title=f'log determinant, normalized')

    fig.suptitle('how the FIM of the 2 layer LN changes with scale; first layer FIM on the left, FIM of both layers on the right.')
    fig.tight_layout()

#%% plot the change in these measures as we go from one layer to the next
measure_diffs = {}
measure_divs = {}
fig, ax = plt.subplots(6, 1, figsize=(5, 10), sharex='all')
fig2, ax2 = plt.subplots(6, 1, figsize=(5, 10), sharex='all')
for s in range(1, 10, 1):
    diff = {}
    div = {}
    for key in volumesb[1].keys():
        diff[key] = volumesb[s][key] - volumes1[s][key]
        div[key] = volumesb[s][key] / volumes1[s][key]
    measure_diffs[s] = diff
    measure_divs[s] = div

    i = 0
    for key in volumesb[1].keys():
        if key=='eigval_prod':
            pass
        else:
            ax[i].plot(s, measure_diffs[s][key], '.', color='blue')
            ax2[i].plot(s, measure_divs[s][key], '.', color='blue')
            i += 1

ax[0].set(title='max eigenvalue')
ax[1].set(title=f'sum of eigenvalues')
ax[2].set(title=f'sum of eigenvalues, normalized')
ax[3].set(title=f'PR', ylim=[-5, 0])
ax[4].set(title=f'normalized PR', ylim=[-2, 2])
ax[5].set(title=f'log determinant, normalized', ylim=[-40, -20])
fig.suptitle('measures: (second layer - first layer)')

ax2[0].set(title='max eigenvalue', ylim=[0, 1])
ax2[1].set(title=f'sum of eigenvalues', ylim=[0, 1])
ax2[2].set(title=f'sum of eigenvalues, normalized', ylim=[0, 1])
ax2[3].set(title=f'PR', ylim=[0,1])
ax2[4].set(title=f'normalized PR', ylim=[0, 1])
ax2[5].set(title=f'log determinant, normalized', ylim=[-2,3])
fig2.suptitle('measures: (second layer / first layer)')

fig.tight_layout()
fig2.tight_layout()





# %% playing around with dependencies
'''
what I've discovered from the above:
1. The PR is invariant to scale transformations of the FIM eigenvalues. 
   It is sensitive to the layer dimension (i.e. the number of neurons in the layer).
2. PR can be made insensitive to layer dim by dividing by the number of neurons in the layer.
   Since this normalized PR (NPR) goes from 1 (large PR) to 0 (small PR), it only measures the rate of decay.
3. We can do something similar with the sum of eigvals: to be insensitive to layer dim, we can divide of num of neurons.
   Unlike the NPR, we are still sensitive to the max eigval and the size of eigvalues.
'''

'''
next steps:
1. see how FIM changes when we apply transformations to the representation (simple ones, and normalization)
1. see what happens when the number of neurons in a layer is doubled
2. expand the network to 2, 3 layers, and calculate each layer's sensitivity to the input
3. train the network to do classification/regression and see whether the FIM measures change. 
'''

# %%
