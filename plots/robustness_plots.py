'''
Uses the eval files to plot the robustness of each normalization method. 
'''

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import pickle
import pandas as pd
import seaborn as sns

#%%
# parameters
dataset = "mnist"
model_name = "convnet4"
# frontend = 'vone_filterbank' # vone_filterbank or learned_conv
frontend = 'learned_conv'
# frontend = 'frozen_conv'
norm_position = 'both'
seed = [1,2,3,4,5,6,7,8,9,10]
seed = [1,2,3]

tr_seed = False
tr_seed = [3]
lr = 0.01
# wd = 0.0005
wd = 0.0005
# wd = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]

if dataset=="mnist":
    # normalize = ["lrnb", "lrns", "gn", "ln", "nn", "lrnc", "in", "bn"]
    normalize = ["bn", "gn", "in", "ln", "lrnc", "lrns", "lrnb"]
    normalize = ["bn", "gn", "in", "ln", "lrns", "lrnb"]
    normalize = ['nn']
    normalize = ["bn", "gn", "in", "ln", "lrnc", "lrns", "lrnb", 'nn']
    eps = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
    eps_plot = eps.copy()
    eps_plot.insert(0, 0)

    # set color palette, and set last color to black
    # palette = sns.diverging_palette(220, 20, n=len(results))
    palette_all = {
        "bn": (0.21607792, 0.39736958, 0.61948028),
        "gn": (0.20344718, 0.56074869, 0.65649508),
        "in": (0.25187832, 0.71827158, 0.67872193),
        "ln": (0.54578602, 0.8544913, 0.69848331),
        "lrnc":(1.0, 0.7686274509803922, 0.0),
        "lrns":(0.9882352941176471, 0.5529411764705883, 0.3843137254901961),
        "lrnb":(0.7634747047461135, 0.3348456555528834, 0.225892295531744),
        "nn":(0., 0., 0.)
    }

    palette = []
    for n in normalize:
        palette.append(palette_all[n])
    print(palette)

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
if (model_name=='convnet3' or model_name[:7]=='convnet') and type(seed)==list:
    for _, n in enumerate(normalize):
        results_per_nm = {}
        if type(tr_seed)!=list:
            print('NOT disentangling train and val mode, both seeds are the same')
            for s in seed:
                file_name = f'{model_name}-lr_{lr}-wd_{wd}-seed_{s}-normalize_{n}-eps_{eps_name}.pkl'
                file_path = os.path.join('..', 'results', f'{model_name}', 'eval_models', f'{frontend}_frontend-norm_{norm_position}', file_name)
                with open(file_path, 'rb') as f:
                    out = pickle.load(f)
                results_per_nm[s] = list(out['perturbed'])
                results_per_nm[s].insert(0, out['clean'])
        else:
            multiple_seeds = seed if len(tr_seed)==1 else tr_seed
            for s in multiple_seeds:
                if len(tr_seed)==1:
                    file_name = f'{model_name}-lr_{lr}-wd_{wd}-seed_{s}-tr_seed_{tr_seed[0]}-normalize_{n}-eps_{eps_name}.pkl'
                else:
                    file_name = f'{model_name}-lr_{lr}-wd_{wd}-seed_{seed[0]}-tr_seed_{s}-normalize_{n}-eps_{eps_name}.pkl'    
                file_path = os.path.join('..', 'results', f'{model_name}', 'eval_models', f'{frontend}_frontend-norm_{norm_position}', file_name)
                with open(file_path, 'rb') as f:
                    out = pickle.load(f)
                results_per_nm[s] = list(out['perturbed'])
                results_per_nm[s].insert(0, out['clean'])
        results[n] = results_per_nm
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

#%% plot the data

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

if type(seed)==list:
    df = pd.DataFrame(results)
    # df2 = pd.DataFrame(columns=['norm method', 'eps', 'mean', 'sd'])
    df2 = pd.DataFrame(columns=['norm method', 'eps', 'seed', 'acc'])

    for _, norm_method in enumerate(normalize):
        for j in range(len(eps_plot)):
            for i in df.axes[0]:
                one_nm_one_eps_one_seed = df[norm_method][i][j]

                new_data = pd.DataFrame({'norm method': [norm_method], 'eps': [eps_plot[j]], 'seed': [i], 'acc': [one_nm_one_eps_one_seed]})
                df2 = df2.append(new_data, ignore_index=True)
    
    # df2 = df2[df2['eps'] == 0]

    ax = sns.lineplot(
        data=df2,
        x='eps',
        y='acc',
        hue='norm method',
        style='norm method',
        ax=ax,
        palette=palette,
        ci='sd',
        err_style='band',
        markers= True,
        dashes=False,
        hue_order=normalize,
        markersize=8
    )
    sns.despine()
    if norm_position != 'both':
        norm_statement = f'normalization at layer {norm_position} only'
    else:
        norm_statement = f'normalization at both (1&2) layers'
    ax.set(xlabel="attack strength", ylabel="accuracy")
    # ax.set(title=f'{model_name} with {frontend} frontend, {norm_statement}. \nseeds = {seed}')
    ax.set(title=f'convnet trained on MNIST, seed={seed}\n weight decay = {wd}')
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

    
# save_name = os.path.join('.', f'robustness-{model_name}-{frontend}_frontend-norm_{norm_position}-seeds_{seed}-wd_{wd}.png')
# plt.savefig(save_name, dpi=400, facecolor='white', bbox_inches='tight', transparent=False)
# plt.show()
# plt.close()

# %%
selected_nm = 'ln'
for selected_nm in normalize:
    df3 = pd.melt(df2, id_vars=['norm method', 'seed', 'eps'], value_vars=['acc'])
    df3 = df3[df3['norm method']==selected_nm]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax = sns.lineplot(
        data=df3,
        x='eps',
        y='value',
        hue='seed',
        legend='full',
        style='seed',
        ax=ax,
        markers= True,
        dashes=False,
        # hue_order=np.arange(1, 10),
        markersize=8
    )
    sns.despine()
    tr_seed_title = f'\n training seeds: {tr_seed}, eval seeds: {seed}' if type(tr_seed) == list else ''
    ax.set(xlabel="attack strength", ylabel="accuracy", title=f'norm method: {selected_nm}, weight decay: {wd}, {tr_seed_title}')

    if type(tr_seed)!=list:
        save_folder = os.path.join('.', 'figures', 'per-seed-robustness', f'{selected_nm}')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        save_name = os.path.join(save_folder,  f'per-seed-robustness-norm_{selected_nm}-wd_{wd}.png')
    if type(tr_seed)==list: 
        save_folder = os.path.join('.', 'figures', 'train-eval-disentangle', f'{selected_nm}')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        save_name = os.path.join(save_folder, f'per-seed-robustness-wd_{wd}-valseed_{seed}-trainseed_{tr_seed}.png')
    plt.savefig(save_name, dpi=400, facecolor='white', bbox_inches='tight', transparent=False)

# %%

# %%
