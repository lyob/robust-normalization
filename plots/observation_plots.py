#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import pickle

#%%
# parameters
dataset = "cifar"
model_name = "standard"
lr = 0.01
wd = 0.0005
seed = 17
if dataset=="mnist":
    normalize = ["lrnb", "lrns", "gn", "ln", "nn", "lrnc", "in", "bn"]
elif dataset=="cifar":
    normalize = ["lrnb", "nn", "lrnc", "lrns", "bn", "gn", "in", "ln"]
eps = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
eps = [str(i) for i in eps]
eps_name = '_'.join(eps)
print(eps_name)

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
        results[n] = out["perturbed"]

print(results)

# %%
# plot the results
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
colors = cm.viridis(np.linspace(0, 1, len(results)))
idx = 0
for name, results in results.items():
    ax.plot(eps, results, 'go-', color=colors[idx], markersize=5, label=name)
    idx += 1
ax.set(xlabel="attack strength", ylabel="accuracy", title=dataset)
ax.legend()

# %%


