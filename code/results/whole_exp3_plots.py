#%%
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

for suffix, title_suffix in [('', ''), ('_large_pool', '(Large Pool)'), ('_matched', '(Matched Frequency)'), ('_cataphor', '(Cataphor Prompt)')]:
    dfs = {}

    results = defaultdict(list)
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
    model_aliases_linebreak = {'gpt2':'GPT-2\nSmall',
            'gpt2-medium':'GPT-2\nMedium',
            'gpt2-large': 'GPT-2\nLarge',
            'gpt2-xl': 'GPT-2\nXL',
            'opt-2.7b': 'OPT\n2.7B',
            'opt-6.7b': 'OPT\n6.7B',
            'opt-13b': 'OPT\n13B',
            'llama-7B': 'LLaMA\n7B',
            'llama-13B': 'LLaMA\n13B',
            'llama-30B': 'LLaMA\n30B',
            }
    for model in models:
        df = pd.read_csv(f'experiment3/{model}{suffix}.csv')
        verb_df = df[df.prompt_type == 'VERB']    
        dfs[model] = df

        results['model'].append(model)
        results['$D_{KL}(A||O)$'].append(df['animate_no_context_experimental_kl'].mean())
        results['$D_{KL}(I||O)$'].append(df['inanimate_no_context_experimental_kl'].mean())
        results['$D_{KL}(A||I)$'].append(df['animate_no_context_inanimate_no_context_kl'].mean())
        
        results['$D_{KL}(A||O)$_std'].append(df['animate_no_context_experimental_kl'].std())
        results['$D_{KL}(I||O)$_std'].append(df['inanimate_no_context_experimental_kl'].std())
        results['$D_{KL}(A||I)$_std'].append(df['animate_no_context_inanimate_no_context_kl'].std())
        

    plt.style.use('ggplot')
    plt.rcParams['text.usetex'] = True
    df = pd.DataFrame.from_dict(results)
    df = df.set_index('model')
    yerr = df[['$D_{KL}(A||O)$_std', '$D_{KL}(I||O)$_std', '$D_{KL}(A||I)$_std']].std(axis=1).to_numpy().T * 2 / 100
    ax = df[['$D_{KL}(A||O)$', '$D_{KL}(I||O)$', '$D_{KL}(A||I)$']].plot.bar(yerr=yerr)

    ax.set_xticklabels([model_aliases_linebreak[model] for model in models])
    ax.xaxis.set_tick_params(rotation=0)
    ax.set_xlabel('Model')
    ax.xaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(axis='y', colors='black')

    ax.set_ylabel('KL Divergence')
    ax.set_title(f'Mean KL Divergences by Model{title_suffix}')
    fig = ax.get_figure()
    fig.tight_layout()
    fig.show()
    fig.savefig(f'paper-plots/exp3_all{suffix}.pdf')
