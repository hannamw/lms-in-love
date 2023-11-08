#%%
from collections import defaultdict
from pathlib import Path 

import jsonlines
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import matplotlib.colors as mc
import colorsys
import plotly.graph_objects as go
import plotly.io as pio   
pio.kaleido.scope.mathjax = None

Path('paper-plots').mkdir(exist_ok=True)
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
model_aliases = {'gpt2':'GPT-2 Small',
          'gpt2-medium':'GPT-2 Medium',
          'gpt2-large': 'GPT-2 Large',
          'gpt2-xl': 'GPT-2 XL',
          'opt-2.7b': 'OPT-2.7B',
          'opt-6.7b': 'OPT-6.7B',
          'opt-13b': 'OPT-13B',
          'llama-7B': 'LLaMA-7B',
          'llama-13B': 'LLaMA-13B',
          'llama-30B': 'LLaMA-30B',
          }
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
def rename_model_index(df):
    return df.index.map(lambda x: model_aliases[x])

def adjust_lightness(color, amount=0.4):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
#%%
plt.style.use('ggplot')
blimp_models = models
blimp_model_set = set(blimp_models)
blimp_df = pd.read_csv('blimp/aggregated.csv')
blimp_df['Model'] = blimp_df['model_name']
blimp_df["Transitive Accuracy"] = blimp_df["transitive_accuracy"]
blimp_df["Passive Accuracy"] = blimp_df["passive_accuracy"]
blimp_df = blimp_df.set_index('Model')
blimp_df = blimp_df[[name in blimp_model_set for name in blimp_df.index]]
blimp_df = blimp_df.reindex(blimp_models)
blimp_df['Human Transitive Accuracy'] = [0.87 for _ in range(len(blimp_df))]
blimp_df['Human Passive Accuracy'] = [0.86 for _ in range(len(blimp_df))]
blimp_df.index = blimp_df.index.map(lambda x: model_aliases[x])
ax = blimp_df.plot(y=["Transitive Accuracy", "Passive Accuracy"], kind="bar", ylabel='Accuracy', ylim=(0.5,1.0))
blimp_df.plot(y=["Human Transitive Accuracy", "Human Passive Accuracy"], ax=ax, linestyle='--')
ax.set_xticklabels([model_aliases_linebreak[model] for model in models])
ax.xaxis.set_tick_params(rotation=0)
ax.xaxis.label.set_color('black')
ax.tick_params(axis='x', colors='black')
ax.set_ylabel("Accuracy", rotation='horizontal', horizontalalignment='left', y=1.02)
ax.yaxis.label.set_color('black')
ax.tick_params(axis='y', colors='black')
ax.legend(loc = "upper center", ncol=2)
ax.set_title("Performance on BLiMP Animacy Datasets")

fig = ax.get_figure()
fig.tight_layout()
fig.show()
fig.savefig('paper-plots/blimp_bar_chart.pdf')


# %%
# PEANUTS EXP 1 INANIMATE
# load in results
results = {}
p1_models = models
for model in p1_models:
    with open(f'peanuts_exp1/{model}.jsonl', 'r') as f:
        with jsonlines.Reader(f) as reader:
            d = reader.read()
        results[model_aliases[model]] = [d[surp] for surp in ['surprisal_t1','surprisal_t3','surprisal_t5']]


p1_df_inanimate = pd.DataFrame(results, index=[1,3,5])
results_inanimate = results 
p1_df_inanimate_inanimate = p1_df_inanimate

# Peanuts Exp 1 Inanimate Bar Chart
p1_df_inanimate = p1_df_inanimate.transpose()
p1_df_inanimate.index.name = 'Model'
p1_df_inanimate = p1_df_inanimate.rename(lambda t: f'surprisal_t{t}', axis='columns')

# plot results
plt.style.use('ggplot')

# %%
# load in results
results = {}
p1_models = models
for model in p1_models:
    with open(f'peanuts_exp1/{model}_animate.jsonl', 'r') as f:
        with jsonlines.Reader(f) as reader:
            d = reader.read()
        results[model_aliases[model]] = [d[surp] for surp in ['surprisal_t1','surprisal_t3','surprisal_t5']]

p1_df_animate = pd.DataFrame(results, index=[1,3,5])

# plot results
plt.style.use('ggplot')

results_animate = results 
p1_df_animate_animate = p1_df_animate

# Peanuts Exp 1 Animate Bar Chart
p1_df_animate = p1_df_animate.transpose()
p1_df_animate.index.name = 'Model'
p1_df_animate = p1_df_animate.rename(lambda t: f'surprisal_t{t}', axis='columns')

# plot results
plt.style.use('ggplot')

#%%
p1_df_inanimate_tp = p1_df_inanimate.rename(lambda t: f'inanimate {t.split("_")[1]}', axis='columns')
ax = p1_df_inanimate_tp.plot.bar(y=[f'inanimate t{t}' for t in [1,3,5]], ylabel='surprisal (bits)', ylim=(0,20))

p1_df_animate_tp = p1_df_animate.rename(lambda t: f'animate {t.split("_")[1]}', axis='columns')
ax = p1_df_animate_tp.plot.bar(y=[f'animate t{t}'for t in [1,3,5]], ylabel='surprisal (bits)', ylim=(0,20), ax=ax, color=[adjust_lightness(color) for color in plt.rcParams['axes.prop_cycle'].by_key()['color'][:3]])
ax.legend(loc = "upper right")
ax.xaxis.set_tick_params(rotation=45)
ax.set_title("Model Surprisal on N&vB Experiment 1")
fig = ax.get_figure()
fig.tight_layout()
fig.show()
fig.savefig('paper-plots/p1_bar_chart_both.pdf')
# %%
fig.clf()
cmap = sns.cm.rocket
# load results
dfs = {}
for model in p1_models:
    dfs[model] = [pd.read_csv(f'peanuts_exp1/{model}.csv'), pd.read_csv(f'peanuts_exp1/{model}_animate.csv')]

# compute_animacy_differences
t1s, t3s, t5s = [], [], []
for model in p1_models:
    in_df, an_df = dfs[model]
    t1, t3, t5 = (in_df[ts] for ts in ['t1_surprisal', 't2_surprisal', 't3_surprisal'])
    t1_ani, t3_ani, t5_ani = (an_df[ts] for ts in ['t1_surprisal', 't2_surprisal', 't3_surprisal'])

    t1s.append(wilcoxon(t1, t1_ani).pvalue)
    t3s.append(wilcoxon(t3, t3_ani).pvalue)
    t5s.append(wilcoxon(t5, t5_ani).pvalue)

d = {'Model': [model_aliases[model] for model in  p1_models], '1': t1s, '3': t3s, '5': t5s}
output_df = pd.DataFrame.from_dict(d)
output_df = output_df.set_index('Model')
fig, ax = plt.subplots()
fig.subplots_adjust(left=0.28)
hm = sns.heatmap(output_df, annot=True, fmt='0.2f', cmap=cmap, cbar_kws={'label': 'p-value'})
hm.set_xlabel('Timestep')
fig = hm.get_figure()
fig.set_size_inches(5, 3.5)
fig.savefig(f'paper-plots/animacy_tests.pdf',bbox_inches="tight")
fig.show()
#%%
d = {'Model': [model_aliases_linebreak[model] for model in  p1_models], '1': t1s, '3': t3s, '5': t5s}
output_df = pd.DataFrame.from_dict(d)
output_df = output_df.set_index('Model')
fig, ax = plt.subplots()
fig.subplots_adjust(left=0.28)
hm = sns.heatmap(output_df.transpose(), annot=True, fmt='0.2f', cmap=cmap, cbar_kws={'label': 'p-value', 'pad':0.02})
hm.set_xlabel('Model')
hm.set_ylabel('Timestep')
hm.tick_params(axis='x', rotation=0, colors='black')
hm.tick_params(axis='y', rotation=0, colors='black')
ax.yaxis.label.set_color('black')
ax.xaxis.label.set_color('black')
fig = hm.get_figure()
hm.set_title('Significance tests for in/animate surprisals')
#fig.tight_layout()
fig.set_size_inches(10, 2)
fig.savefig(f'paper-plots/animacy_tests_horizontal.pdf', bbox_inches="tight")
fig.show()
#%%
ax1 = p1_df_inanimate_tp.plot.bar(y=[f'inanimate t{t}' for t in [1,3,5]], ylabel='surprisal (bits)', ylim=(0,20))
ax = p1_df_animate_tp.plot.bar(y=[f'animate t{t}'for t in [1,3,5]], ylabel='surprisal (bits)', ylim=(0,22), ax=ax1, color=[adjust_lightness(color) for color in plt.rcParams['axes.prop_cycle'].by_key()['color'][:3]])
fig = ax.get_figure()

for p, sig in zip(ax1.patches,t1s + t3s + t5s):
    if sig <= 0.05:
        star = '**' if sig < 0.01 else '*'
        ax.annotate(star, (p.get_x(), p.get_height() * 1.005))

ax.legend(loc = "center right")
ax.xaxis.set_tick_params(rotation=45)
ax.set_title("Model Surprisal on N&vB Experiment 1")
fig.tight_layout()
fig.show()
fig.savefig('paper-plots/p1_bar_chart_both_significance.pdf')

# %%
fig = go.Figure()
alt_model_aliases = {'gpt2':'GPT-2<br>Small',
          'gpt2-medium':'GPT-2<br>Med',
          'gpt2-large': 'GPT-2<br>Large',
          'gpt2-xl': 'GPT-2<br>XL',
          'opt-2.7b': 'OPT<br>2.7B',
          'opt-6.7b': 'OPT<br>6.7B',
          'opt-13b': 'OPT<br>13B',
          'llama-7B': 'LLaMA<br>7B',
          'llama-13B': 'LLaMA<br>13B',
          'llama-30B': 'LLaMA<br>30B',
          }

model_list = [alt_model_aliases[model] for model in  p1_models for _ in range(3)]
timestep_list = [t for _ in p1_models for t in ['t1', 't3', 't5']]

# add inanimate
inanimate_ys = [y for _, row in p1_df_inanimate_tp.iterrows() for y in row.tolist()]
showlegend=True
inanimate_color = '#636EFA'
for i in range(0, len(model_list), 3):
    fig.add_trace(go.Scatter(
    x = [model_list,
        timestep_list],
    y = [None]*i + inanimate_ys[i:i+3] + [None] * (len(model_list) -i -3),
    mode='lines+markers',
    name = "Inanimate Surprisal",
    showlegend=showlegend,
    marker_color=inanimate_color
    ))
    showlegend=False

# add animate
animate_ys = [y for _, row in p1_df_animate_tp.iterrows() for y in row.tolist()]
showlegend=True
animate_color = '#FFA15A'
for i in range(0, len(model_list), 3):
    fig.add_trace(go.Scatter(
    x = [model_list,
        timestep_list],
    y = [None]*i + animate_ys[i:i+3] + [None] * (len(model_list) -i -3),
    mode='lines+markers',
    name = "Animate Surprisal",
    showlegend=showlegend,
    marker_color=animate_color
    ))
    showlegend=False
fig.update_layout(legend=dict(
    orientation='h',
    yanchor="top",
    y=1.12,
    xanchor="left",
    x=0.19,
))

fig.update_layout(xaxis_range=[-0.5,29.5])
fig.update_xaxes(showgrid=True, tickson="labels")
fig.update_xaxes(tickangle=0)

fig.update_layout(
    autosize=True,
    width=550,
    height=400,)

fig.update_layout(
    title={'text':"Repetition Experiment Surprisals", 'x':0.5},
    xaxis_title="Model and Timestep",
    #yaxis={'title':{'text':"Surprisal (bits)", 'standoff':8}}
)
fig.add_annotation(x=0.3, y=1.16, yref='paper', showarrow=False, text="Surprisal<br>(bits)", font=dict(size=14), align='left')


fig.update_layout(
    margin=dict(l=10, r=5, t=60, b=20),
)

fig.write_image('paper-plots/p1_multicategory_line_plot.pdf')
fig.show()


#%%
results = defaultdict(list)

p2_models = models
for model in p2_models:
    with open(f'peanuts_exp2/{model}.jsonl', 'r') as f:
        with jsonlines.Reader(f) as reader:
            d = reader.read()
        results['Inanimate Surprisal'].append(d['inanimate_surprisal'])
        results['Animate Surprisal'].append(d['animate_surprisal'])
        results['Inanimate Baseline'].append(d['baseline_inanimate_surprisal'])
        results['Animate Baseline'].append(d['baseline_animate_surprisal'])
        results['Animate Proportion'].append(d['animate_proportion'])
p2_df = pd.DataFrame(results, index = [model_aliases[model] for model in p2_models])

#%%
ax = p2_df.plot.bar(y=["Animate Surprisal", "Inanimate Surprisal", "Animate Baseline", "Inanimate Baseline"], ylabel='surprisal (bits)', ylim=(0,20))

ax.legend(loc = "upper center", ncol=2)
ax.set_xticklabels([model_aliases_linebreak[model] for model in p2_models])

ax.xaxis.set_tick_params(rotation=0)
ax.set_xlabel('Model')
ax.xaxis.label.set_color('black')
ax.tick_params(axis='x', colors='black')
ax.set_ylabel("Surprisal (bits)", rotation='horizontal', horizontalalignment='left', y=1.03)
ax.yaxis.label.set_color('black')
ax.tick_params(axis='y', colors='black')

ax.set_title("Context Experiment Surprisals")
fig = ax.get_figure()
fig.tight_layout()
fig.show()
fig.savefig('paper-plots/p2_bar_chart.pdf')
