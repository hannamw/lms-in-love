#%%
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

#%%
models = ['gpt2',
          'gpt2-medium',
          'gpt2-large',
          'gpt2-xl',
          'opt-2.7b',
          'opt-6.7b',
          'opt-13b',
          'llama-7B',
          'llama-13B',
          'llama-30B',
          ]

for model in models:
    df = pd.read_csv(f'experiment3/{model}.csv')
    animacy_norms = pd.read_csv('experiment3/animacy_norms.csv')
    verb_df = df[df.prompt_type == 'VERB']
    df['kl_diff'] = df['animate_no_context_inanimate_no_context_kl'] - df['animate_no_context_experimental_kl']
    df['emd_diff'] = df['animate_no_context_inanimate_no_context_emd'] - df['animate_no_context_experimental_emd']
    
    
    ###
    mental_animacy = defaultdict(int)
    mental_animacy.update({word:score for word, score in zip(animacy_norms['Word'], animacy_norms['AnimMental'])})
    physical_animacy = defaultdict(int)
    physical_animacy.update({word:score for word, score in zip(animacy_norms['Word'], animacy_norms['AnimPhysical'])})

    ###
    nouns_list = list(set(df['noun']))

    noun_summary_dict = {'noun': nouns_list}
    for column in df.columns:
        if df[column].dtype == np.float64:
            noun_summary_dict[column] = [df[(df.noun == noun) & (~pd.isna(df[column]))][column].mean() for noun in nouns_list]
    noun_summary_df = pd.DataFrame.from_dict(noun_summary_dict)
    noun_summary_df['AnimMental'] = [mental_animacy[noun] if noun in mental_animacy else np.nan for noun in nouns_list]
    noun_summary_df['AnimPhysical'] = [physical_animacy[noun] if noun in mental_animacy else np.nan for noun in nouns_list]

    verbs_list = list(set(df['verb']))
    verbs_list = list(set(df['verb']))
    verb_summary_dict = {'verb': verbs_list}
    for column in df.columns:
        if df[column].dtype == np.float64:
            verb_summary_dict[column] = [df[(df.verb == verb) & (~pd.isna(df[column]))][column].mean() for verb in verbs_list]
    verb_summary_df = pd.DataFrame.from_dict(verb_summary_dict)
    

    ###
    # LOADING VERBS

    verb_info = pd.read_csv('verbs.csv')

    verb_types = {}
    verb_frequencies = {}

    for verb, verb_frequency, verb_type in zip(verb_info['verb'], verb_info['frequency'], verb_info['type']):
        verb_types[verb] = verb_type
        verb_frequencies[verb] = verb_frequency

    verb_summary_df['verb_frequency'] = [verb_frequencies[verb] for verb in verb_summary_df['verb']]
    verb_summary_df['verb_frequency_color'] = ['r' if freq == 'HIGH' else ('y' if freq == 'HIGH-MID' else 'b') for freq in verb_summary_df['verb_frequency']]
    verb_summary_df['verb_category'] = [verb_types[verb] for verb in verb_summary_df['verb']]
    verb_summary_df['verb_category_color'] = ['r' if cat == 'physical' else 'b' for cat in verb_summary_df['verb_category']]
    
    
    ###
    plot_df = df.copy()
    plot_df['Animacy Divergence'] = plot_df['animate_no_context_experimental_kl'] * 10000
    fig, axs_dict = plt.subplot_mosaic('ABC', sharey=True) #plt.subplots(1, 3, sharey=True)
    axs = list(axs_dict.values())
    (ax1, ax2, ax3) = axs
    sns.kdeplot(data=plot_df, x="Animacy Divergence", hue='prompt_type', common_norm=False, ax=ax1, hue_order=['VERB', 'ADJ'])
    sns.kdeplot(data=plot_df, x="Animacy Divergence", hue='verb_category', common_norm=False, ax=ax2, hue_order=['physical', 'psychological'])
    sns.kdeplot(data=plot_df, x="Animacy Divergence", hue='verb_frequency', common_norm=False, ax=ax3, hue_order=['HIGH', 'HIGH-MID', 'MID'])
    for k, ax in axs_dict.items():
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-4, 6))
        ax.get_xaxis().get_label().set_visible(False)
        ax.text(0.05, 0.85, k, transform=ax.transAxes, 
            size=20, weight='bold')
    
    fig.supxlabel("Animacy Divergence (1e-4)")
    fig.set_size_inches(8.5, 3)
    fig.suptitle("Animacy Divergence Distribution by Prompt Type, Verb Category, and Verb-Human Co-occurrence")
    fig.tight_layout()
    fig.show()
    if model == 'llama-7B':
        fig.savefig(f'paper-plots/distributions-{model}.pdf')
    fig.suptitle(f"Animacy Divergence Dist. by Prompt Type, Verb Category, and Verb-Human Co-occ. ({model})")
    #fig.savefig(f'paper-plots-png/distributions-{model}.png')
    fig.savefig(f'exp3-paper-plots/distributions-{model}.pdf')
    
    ###
    col = 'animate_no_context_experimental_kl'
    print(stats.f_oneway(*[df[df.verb_frequency == freq][col] for freq in ['HIGH', 'HIGH-MID', 'MID']]))
    print(stats.ttest_ind(df[df.verb_frequency == 'HIGH'][col], df[df.verb_frequency == 'HIGH-MID'][col]))
    print(stats.ttest_ind(df[df.verb_frequency == 'MID'][col], df[df.verb_frequency == 'HIGH-MID'][col]))
    print(stats.ttest_ind(df[df.verb_frequency == 'HIGH'][col], df[df.verb_frequency == 'MID'][col]))
    print(stats.ttest_ind(df[df.verb_category == 'physical'][col], df[df.verb_category == 'psychological'][col]))
    
    
    ###
    colorcycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axs_dict = plt.subplot_mosaic('D;E', sharex=True)
    axs = list(axs_dict.values())
    (ax1, ax2) = axs

    plot_df = verb_summary_df.copy()
    plot_df['Animacy Divergence'] = plot_df['animate_no_context_experimental_kl'] * 10000
    s1 = ax1.scatter(plot_df["Animacy Divergence"], np.zeros(len(plot_df)), alpha=0.25, c=colorcycle[0])

    plot_df = noun_summary_df.copy()
    plot_df['Animacy Divergence'] = plot_df['animate_no_context_experimental_kl'] * 10000
    s2 = ax2.scatter(plot_df["Animacy Divergence"], np.zeros(len(plot_df)), alpha=0.25, c=colorcycle[1])
    for k, ax in axs_dict.items():
        ax.text(0.02, 0.68, k, transform=ax.transAxes, 
            size=20, weight='bold')
    fig.set_size_inches(5.66, 3)
    fig.supxlabel("Animacy Divergence (1e-4)")
    fig.suptitle("Mean Animacy Divergence per Verb and Noun")
    fig.legend([s1,s2], ['Verbs', 'Nouns'], loc=(0.8, 0.47), framealpha=1.0)
    fig.tight_layout()
    fig.show()
    if model == 'llama-7B':
        fig.savefig(f'paper-plots/scatters-{model}.pdf')
    fig.suptitle(f"Mean Animacy Divergence per Verb and Noun ({model})")
    #fig.savefig(f'paper-plots-png/scatters-{model}.png')
    fig.savefig(f'exp3-paper-plots/scatters-{model}.pdf')


# %%
