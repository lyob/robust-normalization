'''
What is the effect of weight decay on the variability in robustness?
'''

#%% libraries
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
import seaborn as sns

#%% parameters 
dataset = 'mnist'
model_name = 'convnet4'
frontend = 'learned_conv'
norm_position = 'both'
seed = [1,2,3,4,5,6,7,8,9,10]
lr = 0.01
wd = [0.0, 0.001, 0.002, 0.005, 0.01]
# wd = [0.0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
selected_eps = 0.05

if dataset=='mnist':
    # normalize = ['bn', 'gn', 'in', 'ln', 'lrnb', 'lrnc', 'lrns', 'nn']
    normalize = ['lrnc', 'nn', 'lrnb', 'ln']
    eps = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
    eps_plot = eps.copy()
    eps_plot.insert(0,0)

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
    
eps_name = [str(i) for i in eps]
eps_name = '_'.join(eps_name)

#%% open the results files
results = {}
if (model_name[:7] == 'convnet'):
    for w in wd:
        results_per_wd = {}
        for _, n in enumerate(normalize):
            results_per_nm = {}
            for s in seed:
                file_name = f'{model_name}-lr_{lr}-wd_{w}-seed_{s}-normalize_{n}-eps_{eps_name}.pkl'
                file_path = os.path.join('..', 'results', f'{model_name}', 'eval_models', f'{frontend}_frontend-norm_{norm_position}', file_name)
                with open(file_path, 'rb') as f:
                    out = pickle.load(f)
                results_per_nm[s] = list(out['perturbed'])
                results_per_nm[s].insert(0, out['clean'])
            results_per_wd[n] = results_per_nm
        results[w] = results_per_wd

#%% plot the data

for selected_eps in [0.0, 0.01, 0.03, 0.05, 0.1, 0.15]:
    fig, ax = plt.subplots(1,1,figsize=(8,6))

    df = pd.DataFrame(results)
    df2 = pd.DataFrame(columns=['norm method', 'eps', 'seed', 'acc', 'weight decay'])

    for w in df.axes[1]:
        for _, norm_method in enumerate(normalize):
            for s in seed:
                for e in range(len(eps_plot)):
                    one_wd_one_nm_one_eps_one_seed = df[w][norm_method][s][e]

                    new_data = pd.DataFrame({'norm method': [norm_method], 'eps': [eps_plot[e]], 'seed': [s], 'acc': [one_wd_one_nm_one_eps_one_seed], 'weight decay': [w]})
                    df2 = df2.append(new_data, ignore_index=True)

    df2 = df2[df2['eps']==selected_eps]
    # df2 = df2[df2['norm method']=='lrnc']
    # df2 = df2.groupby(['weight decay', 'norm method']).mean().sort_values(by=['weight decay'])


    ax = sns.lineplot(
        data=df2,
        x='weight decay',
        # y='mean',
        y='acc',
        hue='norm method',
        style='norm method',
        ax=ax,
        palette=palette,
        ci='sd',
        # err_style='band',
        markers= True,
        dashes=False,
        hue_order=normalize,
        markersize=8
    )
    sns.despine()

    # norm_statement = f''
    ax.set(xlabel="weight decay", ylabel="accuracy")
    # ax.set(title=f'{model_name} with {frontend} frontend, {norm_statement}. \nseeds = {seed}')
    ax.set(title=f'convnet trained on MNIST, seeds=1-10\n eps = {selected_eps}')
    plt.xticks(wd)
    ax.set(ylim=([0, 1.0]))

    save_name = os.path.join('.', 'figures','weight-decay', f'weight_decay-seeds_{seed}-wd_{wd}-conditional_eps_{selected_eps}.png')
    plt.savefig(save_name, dpi=400, facecolor='white', bbox_inches='tight', transparent=False)

# %%

