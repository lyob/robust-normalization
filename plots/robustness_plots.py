'''
Uses the eval files to plot the robustness of each normalization method. 
'''

#%%
# import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
import os
import pickle
import pandas as pd
import seaborn as sns

#%% select experiment
params_convnet_disentangling_tr_and_eval_seeds = {
    'dataset': 'mnist',
    'model_name': 'convnet_widthscale',
    'frontend': 'learned_conv',
    'norm_position': 'both',
    'seed': [1,2,3,4,5,6,7,8,9,10],
    'tr_seed': [3],
    'lr': 0.01,
    'wd': 0.0005,  # 0.0, 0.0005, 0.005, 0.05
    'normalize': ["bn", 'gn', "in", "ln", "lrnc", "lrns", "lrnb", 'nn']
}

params_convnet_widthscale = {
    'dataset': 'mnist',
    'model_name': 'convnet_widthscale',
    'frontend': 'learned_conv',
    'norm_position': 'both',
    'seed': [1,2,3,4,5],
    'lr': 0.01,
    'wd': 0.0005,  # 0.0, 0.0005, 0.005, 0.05
    'ws': [2.5],  # 1.5, 2.0, 2.5, 3.0
    'normalize': ["bn", "in", "ln", "lrnc", "lrns", "lrnb", 'nn']  # gn for 2.0 and 3.0
}

params_resnet = {
    'dataset': 'cifar',
    'model_name': 'resnet1',  # resnet1 or resnet2
    'frontend': 'learned_conv',
    'norm_position': 'both',
    'seed': [1,2,3,4,5,6,7,8,9,10],
    'lr': 0.01,
    'wd': 0.0,  # 0.0, 0.0005, 0.005, 0.05
    # 'normalize': ['bn', 'gn', 'in', 'ln', 'lrnc', 'lrns', 'lrnb', 'nn']
    'normalize': ['nn']
}

params_vgg = {
    'dataset': 'cifar',
    'model_name': 'vgg2',
    'seed': [1,2,3,4,5],
    'wd': 0.0005,
    'normalize': ['bn', 'gn', 'in', 'ln', 'lrnc', 'lrns', 'lrnb', 'nn'],
    # 'normalize': ['lrnc', 'nn'],
    # 'normalize': ['lrns', 'nn'],
    # 'normalize': ['lrnb', 'nn'],
    'run_num': 2
}

parameters = params_vgg

#%% parameters
dataset = parameters.get('dataset')
model_name = parameters.get('model_name')
frontend = parameters.get('frontend')
norm_position = parameters.get('norm_position')
seed = parameters.get('seed')
tr_seed = parameters.get('tr_seed')
lr = parameters.get('lr')
wd = parameters.get('wd')
ws = parameters.get('ws')
run_num = parameters.get('run_num')
normalize = parameters.get('normalize')

if dataset=="mnist":
    eps = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
    eps_plot = eps.copy()
    eps_plot.insert(0, 0)
    dataset_name = 'mnist'

elif dataset=="cifar":
    eps = [1.0, 2.0, 4.0, 6.0, 8.0]
    eps_plot = [i/255.0 for i in eps] 
    eps_plot.insert(0, 0)
    dataset_name = 'cifar-10'

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
# print(palette)

eps_name = [str(i) for i in eps]
eps_name = '_'.join(eps_name)

#%%
# open results files 
results = {}
if (model_name[:7]=='convnet'):
    for _, n in enumerate(normalize):
        if type(seed)==list:
            results_per_nm = {}
            ws_string = f'-ws_{ws[0]}' if type(ws)==list else ''
            print(ws_string)
            if type(tr_seed)!=list:
                print('NOT disentangling train and val mode, both seeds are the same')
                for s in seed:
                    # file_name = f'{model_name}-lr_{lr}{ws_string}-wd_{wd}-seed_{s}-normalize_{n}-eps_{eps_name}.pkl'
                    file_name = f'{model_name}-lr_{lr}{ws_string}-wd_{wd}-ev_seed_3-tr_seed_{s}-normalize_{n}-eps_{eps_name}.pkl'
                    file_path = os.path.join('..', 'results', f'{model_name}', 'eval_models', f'{frontend}_frontend-norm_{norm_position}{ws_string}', file_name)
                    with open(file_path, 'rb') as f:
                        out = pickle.load(f)
                    results_per_nm[s] = list(out['perturbed'])
                    results_per_nm[s].insert(0, out['clean'])
            elif type(tr_seed)==list:
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
        elif type(seed)!=list:
            file_name = f'{model_name}-lr_{lr}-wd_{wd}-seed_{seed}-normalize_{n}-eps_{eps_name}.pkl'
            file_path = os.path.join('..', 'results', 'resnet', 'eval_models', model_name, file_name)
            with open(file_path, 'rb') as f:
                out = pickle.load(f)
                results[n] = list(out['perturbed'])
                results[n].insert(0, out['clean'])

elif (model_name[:6]=='resnet'):
    for _, n in enumerate(normalize):
        if type(seed)==list:
            results_per_nm = {}
            for s in seed:
                file_name = f'{model_name}-normalize_{n}-wd_{wd}-seed_{s}-eps_{eps_name}.pkl'
                file_path = os.path.join('..', 'results', 'resnet', 'eval_models', model_name, file_name)
                with open(file_path, 'rb') as f:
                    out = pickle.load(f)
                results_per_nm[s] = list(out['perturbed'])
                results_per_nm[s].insert(0, out['clean'])
            results[n] = results_per_nm
            
        elif type(seed)!=list:
            file_name = f'{model_name}-normalize_{n}-wd_{wd}-seed_{seed}-eps_{eps_name}.pkl'
            file_path = os.path.join('..', 'results', 'resnet', 'eval_models', model_name, file_name)
            with open(file_path, 'rb') as f:
                out = pickle.load(f)
            results[n] = list(out['perturbed'])
            results[n].insert(0, out['clean'])
elif (model_name[:3]=='vgg'):
    for _, n in enumerate(normalize):
        results_per_nm = {}
        for s in seed:
            file_name = f'nm:{n}-seed:{s}-wd:{wd}-run:{run_num}-eps:{eps_name}.pkl'
            file_path = os.path.join('..', 'results', 'vgg', model_name, 'eval_models', file_name)
            with open(file_path, 'rb') as f:
                out = pickle.load(f)
            results_per_nm[s] = list(out['perturbed'])
            results_per_nm[s].insert(0, out['clean'])
        results[n] = results_per_nm



#%% plot the data

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

if type(seed)==list:
    df = pd.DataFrame(results)
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
    # ax.set(title=f'{model_name} with {frontend} frontend, {norm_statement}. \nseeds = {seed}')
    if dataset=='mnist':
        plt.xticks((0, 0.05, 0.1, 0.15, 0.2))
    elif dataset=='cifar':
        plt.xticks(eps_plot, labels=['0.0', '1.0', '2.0', '4.0', '6.0', '8.0'])
        xlabel_info = '(x/255)'
    
    zoomed=False

    if model_name[:7] == 'convnet':
        model = model_name[:7]
        xlabel_info = ''
    elif model_name[:6] == 'resnet':
        model = model_name[:6]
        ax.set_ylim(top=1.0)
    elif model_name[:3] == 'vgg':
        model = model_name[:3]
        ax.set_ylim(top=1.0)
        if zoomed:
            ax.set_ylim([0.1, 0.7])
            ax.set_xlim([0,0.02])

    title_info = f', width scale = {ws[0]}' if type(ws)==list else ''
    ax.set(xlabel=f"attack strength {xlabel_info}", ylabel="accuracy")
    ax.set(title=f'{model} trained on {dataset_name}, seeds={seed}\n weight decay = {wd}{title_info}')

            
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

if model=='convnet':
    save_path = os.path.join('.', 'figures', 'robustness-convnet_widthscale', f'ws-{ws[0]}')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    save_name = os.path.join(save_path, f'robustness-{model_name}-{frontend}_frontend-norm_{norm_position}-seeds_{seed}-wd_{wd}.png')
elif model=='resnet':
    save_path = os.path.join('.', 'figures', f'{model}-robustness')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    save_name = os.path.join(save_path, f'robustness-{model_name}-seeds_{seed}-wd_{wd}-nm_{normalize}.png')
elif model=='vgg':
    save_path = os.path.join('.', 'figures', f'{model}-robustness')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    if zoomed: 
        save_path = os.path.join(save_path, 'zoomed')
    norm_method = normalize if normalize != ['bn', 'gn', 'in', 'ln', 'lrnc', 'lrns', 'lrnb', 'nn'] else 'all'
    save_name = os.path.join(save_path, f'robustness-model={model_name}-norm={norm_method}-seeds={seed}-wd={wd}.png')
plt.savefig(save_name, dpi=400, facecolor='white', bbox_inches='tight', transparent=False)


# %%
# selected_nm = 'ln'
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
    # plt.savefig(save_name, dpi=400, facecolor='white', bbox_inches='tight', transparent=False)

# %%

# %%
