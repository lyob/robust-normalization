#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import pickle

#%%
# parameters
dataset = "mnist"
model_name = "standard"
lr = 0.01
wd = 0.0005
seed = 17
if dataset=="mnist":
    normalize = ["lrnb", "lrns", "gn", "ln", "nn", "lrnc", "in", "bn"]
    eps = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
    eps_plot = eps.copy()
    eps_plot.insert(0, 0)
elif dataset=="cifar":
    normalize = ["nn", "lrnb", "lrnc", "lrns", "ln", "bn", "gn", "in"]
    eps = [1.0, 2.0, 4.0, 6.0, 8.0]
    eps_plot = [i/255.0 for i in eps] 
    eps_plot.insert(0, 0)

eps_name = [str(i) for i in eps]
eps_name = '_'.join(eps_name)

#%%
# open results files 
results = {}
for _, n in enumerate(normalize):
    if dataset=="mnist":
        file_name = f'{model_name}-lr_{lr}-wd_{wd}-seed_{seed}-normalize_{n}-eps_{eps_name}.pkl'
    elif dataset=="cifar":
        file_name = f'{model_name}-normalize_{n}-wd_{wd}-seed_{seed}-eps_{eps_name}.pkl'
    file_path = os.path.join('..', 'results', f'{dataset}_regularize', 'eval_models', model_name, file_name)
    with open(file_path, 'rb') as f:
        out = pickle.load(f)
        results[n] = out

# %%
# plot the results
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
print(len(results.keys()))
colors = cm.viridis(np.linspace(0, 1, len(results)))

idx = 0
for name, accuracies in results.items():
    accs = list(accuracies['perturbed'])
    accs.insert(0, accuracies['clean']) 
    
    ax.plot(eps_plot, accs, 'go-', color=colors[idx], markersize=5, label=name)
    idx += 1
ax.set(xlabel="attack strength", ylabel="accuracy", title=dataset)
if dataset=='cifar':
    plt.xticks((0, 0.01, 0.02, 0.03))
if dataset=='mnist':
    plt.xticks((0, 0.05, 0.1, 0.15, 0.2))
ax.legend()

# save the figure
save_name = os.path.join('.', f'robustness_{dataset}.png')
plt.savefig(save_name, dpi=800, bbox_inches='tight', transparent=False)
# plt.show()
# plt.close()

# %%


