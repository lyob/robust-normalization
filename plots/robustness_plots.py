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
seed = [1,2,3,4,5, 17]
# seed = [2,3]
lr = 0.01
wd = 0.0005

if dataset=="mnist":
    # normalize = ["lrnb", "lrns", "gn", "ln", "nn", "lrnc", "in", "bn"]
    normalize = ["bn", "gn", "in", "ln", "lrnc", "lrns", "lrnb"]
    normalize = ["bn", "gn", "in", "ln", "lrns", "lrnb"]
    normalize = ["bn", "gn", "in", "ln", "lrnc", "lrns", "lrnb", 'nn']
    normalize = ['nn']
    eps = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
    eps_plot = eps.copy()
    eps_plot.insert(0, 0)

    # set color palette, and set last color to black
    # palette = sns.diverging_palette(220, 20, n=len(results))
    palette_all = {
        "bn": (0.24715576253545807, 0.49918708160096675, 0.5765599057376697),
        "gn": (0.44221678412697046, 0.6256298550214647, 0.6823748435735669),
        "in": (0.6453947655470793, 0.757334217374988, 0.792592996324836),
        "ln": (0.924371914496006, 0.8581356111320103, 0.8422135595757048),
        "lrnc":(0.8714732581609897, 0.6860920090908339, 0.6395837763165448),
        "lrns":(0.8163733610811298, 0.5068892575940598, 0.4285220787909039),
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
# print(results['bn'])


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

    ax = sns.lineplot(
        data=df2,
        x='eps',
        # y='mean',
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
    ax.set(title=f'convnet trained on MNIST, seed={seed}')
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

    
save_name = os.path.join('.', f'robustness-{model_name}-{frontend}_frontend-norm_{norm_position}-seeds_{seed}.png')
plt.savefig(save_name, dpi=400, facecolor='white', bbox_inches='tight', transparent=False)
# plt.show()
# plt.close()

# %%



# %%
