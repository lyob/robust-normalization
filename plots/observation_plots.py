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
normalize = ["lrnb", "lrns", "gn", "ln", "nn", "lrnc", "in", "bn"]
eps = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
eps = [str(i) for i in eps]
eps_name = '_'.join(eps)
print(eps_name)

#%%
# open results files 
results = {}
for _, n in enumerate(normalize):
    file_name = f'{model_name}-lr_{str(lr)}-wd_{str(wd)}-seed_{str(seed)}-normalize_{n}-eps_{eps_name}.pkl'
    file_path = os.path.join('..', 'results', f'{dataset}_regularize', 'eval_models', 'standard', file_name)
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
ax.set(xlabel="attack strength", ylabel="accuracy", title="MNIST")
ax.legend()

# %%


