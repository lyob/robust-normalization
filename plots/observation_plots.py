#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import pickle
import pandas as pd
import seaborn as sns
from cycler import cycler

#%%
# parameters
dataset = "mnist"
model_name = "convnet"
frontend = 'vone_filterbank' # vone_filterbank or learned_conv
norm_position = 'both'
# seed = 17
seed = [1,2,3,4,5]
lr = 0.01
wd = 0.0005

if dataset=="mnist":
    # normalize = ["lrnb", "lrns", "gn", "ln", "nn", "lrnc", "in", "bn"]
    normalize = ["bn", "gn", "in", "ln", "lrnb", "lrnc", "lrns", "nn"]
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
if model_name=='convnet' and type(seed)==list:
    for _, n in enumerate(normalize):
        results_per_nm = {}
        for s in seed:
            file_name = f'{model_name}-lr_{lr}-wd_{wd}-seed_{s}-normalize_{n}-eps_{eps_name}.pkl'
            file_path = os.path.join('..', 'results', f'{model_name}', 'eval_models', f'{frontend}_frontend-norm_{norm_position}', file_name)
            with open(file_path, 'rb') as f:
                out = pickle.load(f)
            results_per_nm[s] = list(out['perturbed'])
            results_per_nm[s].insert(0, out['clean'])
        results[n] = results_per_nm
    # print(results)

else:
    for _, n in enumerate(normalize):
        if dataset=="mnist":
            file_name = f'{model_name}-lr_{lr}-wd_{wd}-seed_{seed}-normalize_{n}-eps_{eps_name}.pkl'
        elif dataset=="cifar":
            file_name = f'{model_name}-normalize_{n}-wd_{wd}-seed_{seed}-eps_{eps_name}.pkl'
                    
        # print(os.path.abspath('../results/'))
        # file_path = os.path.join('..', 'results', f'{dataset}_regularize', 'eval_models', model_name, file_name)
        file_path = os.path.join('..', 'results', f'{dataset}_regularize', 'eval_models', model_name, file_name)
        with open(file_path, 'rb') as f:
            out = pickle.load(f)
            results[n] = list(out['perturbed'])
            results[n].insert(0, out['clean'])

# %%
print(results.keys())
print(results['nn'])


#%% plot the data

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# colors = cm.Spectral(np.linspace(0, 1, len(results)))
# colors = np.append(colors, cm.Spectral(np.linspace(0.65, 1, len(results)//2)), axis=0)
# colors = cm.Set2(np.linspace(0,1,len(results)))

if type(seed)==list:
    df = pd.DataFrame(results)
    df2 = pd.DataFrame(columns=['norm method', 'eps', 'mean', 'sd'])

    for _, norm_method in enumerate(normalize):
        for j in range(len(eps_plot)):
            one_nm_one_eps = np.array([df[norm_method][i][j] for i in df.axes[0]])
            mean = one_nm_one_eps.mean()
            sd = one_nm_one_eps.std()

            new_data = pd.DataFrame({'norm method': [norm_method], 'eps': [eps_plot[j]], 'mean': [mean], 'sd': [sd]})
            df2 = df2.append(new_data, ignore_index=True)

    df2 = df2.groupby(['eps', 'norm method']).mean().sort_values(by=['eps'])

    palette = sns.diverging_palette(220, 20, n=len(results))
    print(palette)
    palette.insert(-1, (0,0,0))
    palette.pop()

    ax = sns.lineplot(
        data=df2,
        x='eps',
        y='mean',
        hue='norm method',
        ax=ax,
        ci='sd',
        palette=palette
    )
    sns.despine()
    if norm_position != 'both':
        norm_statement = f'normalization at layer {norm_position} only'
    else:
        norm_statement = f'normalization at both (1&2) layers'
    ax.set(xlabel="attack strength", ylabel="accuracy", title=f'{model_name} with {frontend} frontend, {norm_statement}. (5 seeds)')
    plt.xticks((0, 0.05, 0.1, 0.15, 0.2))

            
else:
    colors=sns.diverging_palette(220, 20, n=len(results))
    colors.insert(-1, (0,0,0))
    colors.pop()

    idx = 0
    for name, accuracies in results.items():
        ax.plot(eps_plot, accuracies, 'go-', color=colors[idx], markersize=5, label=name)
        idx += 1


    ax.set(xlabel="attack strength", ylabel="accuracy", title=dataset)
    if dataset=='cifar':
        plt.xticks((0, 0.01, 0.02, 0.03))
    if dataset=='mnist':
        plt.xticks((0, 0.05, 0.1, 0.15, 0.2))
    ax.legend()

    
save_name = os.path.join('.', f'robustness-{model_name}-{frontend}_frontend-norm_{norm_position}.png')
plt.savefig(save_name, dpi=400, facecolor='white', bbox_inches='tight', transparent=False)
# plt.show()
# plt.close()

# %%


