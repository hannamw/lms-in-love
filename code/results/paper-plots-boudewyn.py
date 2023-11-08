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
def rename_model_index(df):
    return df.index.map(lambda x: model_aliases[x])

def adjust_lightness(color, amount=0.4):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
# %%
# BOUDEWYN EXP 1 INANIMATE

# load in results
results = {}
results_reversed = {k:[] for k in ['Model', 'surprisal_t1', 'surprisal_t2']}

p1_models = models
for model in p1_models:
    with open(f'boudewyn_exp1/{model}.jsonl', 'r') as f:
        with jsonlines.Reader(f) as reader:
            d = reader.read()
        results[model_aliases[model]] = [d[surp] for surp in ['surprisal_t1','surprisal_t3']]
        results_reversed['Model'].append(model_aliases[model])
        results_reversed['surprisal_t1'].append(d['surprisal_t1'])
        t2_col = 'surprisal_t2' if 'surprisal_t2' in d else 'surprisal_t3'
        results_reversed['surprisal_t2'].append(d[t2_col])

p1_df = pd.DataFrame(results_reversed)
p1_df = p1_df.set_index('Model')
# plot results
plt.style.use('ggplot')
results_inanimate = results 
p1_df_inanimate = p1_df
# %%
# load in results
results = {}
results_reversed = {k:[] for k in ['Model', 'surprisal_t1', 'surprisal_t2']}

p1_models = models
for model in p1_models:
    with open(f'boudewyn_exp1/{model}_animate.jsonl', 'r') as f:
        with jsonlines.Reader(f) as reader:
            d = reader.read()
        results[model_aliases[model]] = [d[surp] for surp in ['surprisal_t1','surprisal_t3']]
        results_reversed['Model'].append(model_aliases[model])
        results_reversed['surprisal_t1'].append(d['surprisal_t1'])
        t2_col = 'surprisal_t2' if 'surprisal_t2' in d else 'surprisal_t3'
        results_reversed['surprisal_t2'].append(d[t2_col])


p1_df = pd.DataFrame(results_reversed)
p1_df = p1_df.set_index('Model')

# plot results
plt.style.use('ggplot')
results_animate = results 
p1_df_animate = p1_df

#%%
p1_df_inanimate_tp = p1_df_inanimate.rename(lambda t: f'inanimate {t.split("_")[1]}', axis='columns')
ax = p1_df_inanimate_tp.plot.bar(y=[f'inanimate t{t}' for t in [1,2]], ylabel='surprisal (bits)', ylim=(0,20))

p1_df_animate_tp = p1_df_animate.rename(lambda t: f'animate {t.split("_")[1]}', axis='columns')
ax = p1_df_animate_tp.plot.bar(y=[f'animate t{t}'for t in [1,2]], ylabel='surprisal (bits)', ylim=(0,20), ax=ax, color=[adjust_lightness(color) for color in plt.rcParams['axes.prop_cycle'].by_key()['color'][:3]])
ax.legend(loc = "upper right")
ax.xaxis.set_tick_params(rotation=45)
ax.set_title("Model Surprisal on Boudewyn et al. Experiment 1")
fig = ax.get_figure()
fig.tight_layout()
fig.show()
fig.savefig('paper-plots/s1_bar_chart_both.pdf')
# %%
fig.clf()
cmap = sns.cm.rocket
# load results
dfs = {}
for model in p1_models:
    dfs[model] = [pd.read_csv(f'boudewyn_exp1/{model}.csv'), pd.read_csv(f'boudewyn_exp1/{model}_animate.csv')]

# compute_animacy_differences
t1s, t2s = [], []
for model in p1_models:
    in_df, an_df = dfs[model]
    t1, t2, = (in_df[ts] for ts in ['t1_surprisal', 't2_surprisal'])
    t1_ani, t2_ani = (an_df[ts] for ts in ['t1_surprisal', 't2_surprisal'])

    t1s.append(wilcoxon(t1, t1_ani).pvalue)
    t2s.append(wilcoxon(t2, t2_ani).pvalue)

d = {'Model': [model_aliases[model] for model in  p1_models], '1': t1s, '2': t2s, }
output_df = pd.DataFrame.from_dict(d)
output_df = output_df.set_index('Model')
fig, ax = plt.subplots()
fig.subplots_adjust(left=0.28)
hm = sns.heatmap(output_df, annot=True, fmt='0.2f', cmap=cmap, cbar_kws={'label': 'p-value'})
hm.set_xlabel('timestep')
fig = hm.get_figure()
hm.set_title('Significance tests for in/animate surprisals')
#fig.savefig(f'paper-plots/animacy_tests_boudewyn.png')
fig.savefig(f'paper-plots/animacy_tests_boudewyn.pdf')
fig.show()
#%%
fig = go.Figure()

model_list = [alt_model_aliases[model] for model in  p1_models for _ in range(2)]
timestep_list = [t for _ in p1_models for t in ['t1', 't2']]

# add inanimate
inanimate_ys = [y for _, row in p1_df_inanimate_tp.iterrows() for y in row.tolist()]
showlegend=True
inanimate_color = '#636EFA'
for i in range(0, len(model_list), 2):
    fig.add_trace(go.Scatter(
    x = [model_list,
        timestep_list],
    y = [None]*i + inanimate_ys[i:i+2] + [None] * (len(model_list) -i -2),
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
for i in range(0, len(model_list), 2):
    fig.add_trace(go.Scatter(
    x = [model_list,
        timestep_list],
    y = [None]*i + animate_ys[i:i+2] + [None] * (len(model_list) -i -2),
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

fig.update_layout(xaxis_range=[-0.5,19.5], yaxis_range=[-0.5, 15])
fig.update_xaxes(showgrid=True, tickson="labels")
fig.update_xaxes(tickangle=0)

fig.update_layout(
    autosize=False,
    width=550,
    height=400,)

fig.update_layout(
    title={'text':"Adaptation Experiment Surprisals", 'x':0.5},
    xaxis_title="Model and Timestep",
)
fig.add_annotation(x=0.0, y=1.19, yref='paper', showarrow=False, text="Surprisal<br>(bits)", font=dict(size=14), align='left')

fig.update_layout(
    margin=dict(l=10, r=5, t=60, b=20),
)

fig.write_image('paper-plots/s1_multicategory_line_plot.pdf')
fig.show()

#%%
fig = go.Figure()

model_list = [alt_model_aliases[model] for model in  p1_models for _ in range(2)]
timestep_list = [t for _ in p1_models for t in ['t1', 't2']]

# add inanimate
inanimate_ys = [y for _, row in p1_df_inanimate_tp.iterrows() for y in row.tolist()]
showlegend=True
inanimate_color = '#636EFA'
for i in range(0, len(model_list), 2):
    fig.add_trace(go.Scatter(
    x = [model_list,
        timestep_list],
    y = [None]*i + inanimate_ys[i:i+2] + [None] * (len(model_list) -i -2),
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
for i in range(0, len(model_list), 2):
    fig.add_trace(go.Scatter(
    x = [model_list,
        timestep_list],
    y = [None]*i + animate_ys[i:i+2] + [None] * (len(model_list) -i -2),
    mode='lines+markers',
    name = "Animate Surprisal",
    showlegend=showlegend,
    marker_color=animate_color
    ))
    showlegend=False
fig.update_layout(legend=dict(
    orientation='h',
    yanchor="top",
    y=1.2,
    xanchor="left",
    x=0.19,
))

fig.update_layout(xaxis_range=[-0.5,19.5], yaxis_range=[4.5, 15])
fig.update_xaxes(showgrid=True, tickson="labels")
fig.update_xaxes(tickangle=0)

fig.update_layout(
    autosize=False,
    width=550,
    height=300,)

fig.update_layout(
    title={'text':"Adaptation Experiment Surprisals", 'x':0.5},
    xaxis_title="Model and Timestep",
    #yaxis_title="Surprisal (bits)",
)
fig.add_annotation(x=0.0, y=1.35, yref='paper', showarrow=False, text="Surprisal<br>(bits)", font=dict(size=14), align='left')

fig.update_layout(
    margin=dict(l=10, r=5, t=70, b=20),
)

fig.write_image('paper-plots/s1_multicategory_line_plot_short.pdf')
fig.show()

# %%
results = defaultdict(list)

p2_models = models
for model in p2_models:
    with open(f'boudewyn_exp2/{model}.jsonl', 'r') as f:
        with jsonlines.Reader(f) as reader:
            d = reader.read()
        #results['Model'].append(model_aliases[model])
        results['Inanimate Surprisal'].append(d['inanimate_surprisal'])
        results['Animate Surprisal'].append(d['animate_surprisal'])
        results['Inanimate Baseline'].append(d['baseline_inanimate_surprisal'])
        results['Animate Baseline'].append(d['baseline_animate_surprisal'])
        results['Animate Proportion'].append(d['animate_proportion'])
p2_df = pd.DataFrame(results, index = [model_aliases[model] for model in p2_models])

ax = p2_df.plot.line(y=['Inanimate Surprisal', 'Animate Surprisal', 'Inanimate Baseline', 'Animate Baseline'], xlabel='Model', ylabel='Surprisal (bits)')
ax2 = p2_df.plot.line(y=['Animate Proportion'], ax=ax, secondary_y=True,linestyle='--')
fig = ax.get_figure()
ax.set_ylim(10, 22)
ax.grid(True)
ax2.grid(False)
ax2.set_ylabel('Proportion')
ax2.set_ylim(0.7, 1.0)
ax.set_xticks(list(range(len(p2_models))))
ax.set_xticklabels([model_aliases_linebreak[model] for model in p2_models])
ax.xaxis.set_tick_params(rotation=45)
ax.set_title('Boudewyn Experiment 2 Surprisals')
fig.tight_layout()
fig.show()
#fig.savefig('paper-plots/s2_line_chart.png')
fig.savefig('paper-plots/s2_line_chart.pdf')
# %%

ax = p2_df.plot.bar(y=["Animate Surprisal", "Inanimate Surprisal", "Animate Baseline", "Inanimate Baseline"], ylabel='surprisal (bits)', ylim=(0,22))

ax.legend(loc = "lower right")
ax.set_xticklabels([model_aliases_linebreak[model] for model in p2_models])
ax.xaxis.set_tick_params(rotation=0)
ax.set_xlabel('Model')
ax.xaxis.label.set_color('black')
ax.tick_params(axis='x', colors='black')
ax.set_ylabel("Surprisal (bits)", rotation='horizontal', horizontalalignment='left', y=1.02)
ax.yaxis.label.set_color('black')
ax.tick_params(axis='y', colors='black')

ax.set_title("English Context Experiment Surprisals")
fig = ax.get_figure()
fig.tight_layout()
fig.show()
fig.savefig('paper-plots/s2_bar_chart.pdf')
# %%
