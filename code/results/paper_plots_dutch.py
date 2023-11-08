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

Path('paper-plots').mkdir(exist_ok=True)

models = ['gpt2-small-dutch',
          'gpt2-medium-dutch-embeddings',
          'gpt2-medium-dutch',
          'gpt2-large-dutch',
          ]
model_aliases = {'gpt2-small-dutch':'GPT-2 Small',
          'gpt2-medium-dutch-embeddings': 'GPT-2 Medium (Embs)',
          'gpt2-medium-dutch':'GPT-2 Medium',
          'gpt2-large-dutch': 'GPT-2 Large',
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
# PEANUTS EXP 1 INANIMATE
# load in results
results = {}
p1_models = models
for model in p1_models:
    with open(f'peanuts_exp1/{model}_dutch.jsonl', 'r') as f:
        with jsonlines.Reader(f) as reader:
            d = reader.read()
        results[model_aliases[model]] = [d[surp] for surp in ['surprisal_t1','surprisal_t3','surprisal_t5']]


p1_df_inanimate = pd.DataFrame(results, index=[1,3,5])

# plot results
plt.style.use('ggplot')
ax = p1_df_inanimate.plot.line(ylabel='Surprisal (bits)', ylim=(0.0, 20.0), xlabel='Timestep', xticks=[1,3,5])
ax.legend(loc = "upper right")
ax.set_title("Model Surprisal on N&vB Experiment 1 (inanimate)")

fig = ax.get_figure()
fig.tight_layout()
fig.show()
#fig.savefig('paper-plots/p1_line_chart_inanimate.png')
fig.savefig('paper-plots-dutch/p1_line_chart_inanimate.pdf')
results_inanimate = results 
p1_df_inanimate_inanimate = p1_df_inanimate

# Peanuts Exp 1 Inanimate Bar Chart
p1_df_inanimate = p1_df_inanimate.transpose()
p1_df_inanimate.index.name = 'Model'
p1_df_inanimate = p1_df_inanimate.rename(lambda t: f'surprisal_t{t}', axis='columns')

# plot results
plt.style.use('ggplot')
ax = p1_df_inanimate.plot.bar(y=['surprisal_t1', 'surprisal_t3', 'surprisal_t5'], ylabel='surprisal (bits)', ylim=(0,20))
ax.legend(loc = "center right")
ax.xaxis.set_tick_params(rotation=0)
ax.set_title("Model Surprisal on N&vB Experiment 1 (inanimate)")
fig = ax.get_figure()
fig.tight_layout()
fig.show()
#fig.savefig('paper-plots/p1_bar_chart_inanimate.png')
fig.savefig('paper-plots-dutch/p1_bar_chart_inanimate.pdf')

# %%
# load in results
results = {}
p1_models = models
for model in p1_models:
    with open(f'peanuts_exp1/{model}_dutch_animate.jsonl', 'r') as f:
        with jsonlines.Reader(f) as reader:
            d = reader.read()
        results[model_aliases[model]] = [d[surp] for surp in ['surprisal_t1','surprisal_t3','surprisal_t5']]

p1_df_animate = pd.DataFrame(results, index=[1,3,5])

# plot results
plt.style.use('ggplot')
ax = p1_df_animate.plot.line(ylabel='Surprisal (bits)', ylim=(0.0, 20.0), xlabel='Timestep', xticks=[1,3,5])
ax.legend(loc = "upper right")
ax.set_title("Model Surprisal on N&vB Experiment 1 (animate)")

fig = ax.get_figure()
fig.tight_layout()
fig.show()
#fig.savefig('paper-plots/p1_line_chart_animate.png')
fig.savefig('paper-plots-dutch/p1_line_chart_animate.pdf')
results_animate = results 
p1_df_animate_animate = p1_df_animate

# Peanuts Exp 1 Animate Bar Chart
p1_df_animate = p1_df_animate.transpose()
p1_df_animate.index.name = 'Model'
p1_df_animate = p1_df_animate.rename(lambda t: f'surprisal_t{t}', axis='columns')

# plot results
plt.style.use('ggplot')
ax = p1_df_animate.plot.bar(y=['surprisal_t1', 'surprisal_t3', 'surprisal_t5'], ylabel='surprisal (bits)', ylim=(0,20))
ax.legend(loc = "upper right")
ax.xaxis.set_tick_params(rotation=0)
ax.set_title("Model Surprisal on N&vB Experiment 1 (animate)")
fig = ax.get_figure()
fig.tight_layout()
fig.show()
#fig.savefig('paper-plots/p1_bar_chart_animate.png')
fig.savefig('paper-plots-dutch/p1_bar_chart_animate.pdf')

#%%
p1_df_inanimate_tp = p1_df_inanimate.rename(lambda t: f'inanimate {t.split("_")[1]}', axis='columns')
ax = p1_df_inanimate_tp.plot.bar(y=[f'inanimate t{t}' for t in [1,3,5]], ylabel='surprisal (bits)', ylim=(0,20))

p1_df_animate_tp = p1_df_animate.rename(lambda t: f'animate {t.split("_")[1]}', axis='columns')
ax = p1_df_animate_tp.plot.bar(y=[f'animate t{t}'for t in [1,3,5]], ylabel='surprisal (bits)', ylim=(0,20), ax=ax, color=[adjust_lightness(color) for color in plt.rcParams['axes.prop_cycle'].by_key()['color'][:3]])
ax.legend(loc = "upper right")
ax.xaxis.set_tick_params(rotation=0)
ax.set_title("Model Surprisal on N&vB Experiment 1")
fig = ax.get_figure()
fig.tight_layout()
fig.show()
fig.savefig('paper-plots-dutch/p1_bar_chart_both.pdf')
# %%
fig.clf()
cmap = sns.cm.rocket
# load results
dfs = {}
for model in p1_models:
    dfs[model] = [pd.read_csv(f'peanuts_exp1/{model}_dutch.csv'), pd.read_csv(f'peanuts_exp1/{model}_dutch_animate.csv')]

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
hm.set_title('Significance tests for in/animate surprisals')
#fig.savefig(f'paper-plots/animacy_tests.png')
fig.savefig(f'paper-plots-dutch/animacy_tests.pdf')
fig.show()
#%%
ax1 = p1_df_inanimate_tp.plot.bar(y=[f'inanimate t{t}' for t in [1,3,5]], ylabel='surprisal (bits)', ylim=(0,20))
ax = p1_df_animate_tp.plot.bar(y=[f'animate t{t}'for t in [1,3,5]], ylabel='surprisal (bits)', ylim=(0,22), ax=ax1, color=[adjust_lightness(color) for color in plt.rcParams['axes.prop_cycle'].by_key()['color'][:3]])
fig = ax.get_figure()

for p, sig in zip(ax1.patches,t1s + t3s + t5s):#[t for ts in zip(t1s, t3s, t5s) for t in ts]):
    if sig <= 0.05:
        ax.annotate('*', (p.get_x() +0.05 , p.get_height() * 1.005))
        #fig.text(p.get_x() + p.get_width() / 2., p.get_height(), star, ha='center')

ax.legend(loc = "center right")
ax.xaxis.set_tick_params(rotation=0)
ax.set_title("Model Surprisal on N&vB Experiment 1")
fig.tight_layout()
fig.show()
fig.savefig('paper-plots-dutch/p1_bar_chart_both_significance.pdf')
#%%
fig = go.Figure()
alt_model_aliases = {'gpt2-small-dutch':'GPT-2 Small',
          'gpt2-medium-dutch-embeddings':'GPT-2 Medium<br>(Embeddings)',
          'gpt2-medium-dutch':'GPT-2 Medium',
          'gpt2-large-dutch': 'GPT-2 Large',
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
    y=1.14,
    xanchor="left",
    x=0.18
))

fig.update_layout(xaxis_range=[-0.5,11.5])
fig.update_xaxes(showgrid=True, tickson="labels")

fig.update_layout(
    autosize=False,
    width=600,
    height=400,)

fig.update_layout(
    title={'text':"Dutch Repetition Experiment Surprisals", 'x':0.5},
    xaxis_title="Model and Timestep",
    yaxis_title="Surprisal (bits)",
)

fig.update_layout(
    margin=dict(l=30, r=5, t=80, b=20),
)

fig.write_image('paper-plots-dutch/p1_multicategory_line_plot.pdf')
fig.show()

# %%
results = defaultdict(list)
plt.style.use('ggplot')
p2_models = models
for model in p2_models:
    with open(f'peanuts_exp2/{model}_dutch.jsonl', 'r') as f:
        with jsonlines.Reader(f) as reader:
            d = reader.read()
        #results['Model'].append(model_aliases[model])
        results['Inanimate Surprisal'].append(d['inanimate_surprisal'])
        results['Animate Surprisal'].append(d['animate_surprisal'])
        results['Inanimate Baseline'].append(d['baseline_inanimate_surprisal'])
        results['Animate Baseline'].append(d['baseline_animate_surprisal'])
        results['Animate Proportion'].append(d['animate_proportion'])
p2_df = pd.DataFrame(results, index = [model_aliases[model] for model in p2_models])
ax = p2_df.plot.bar(y=['Inanimate Surprisal', 'Animate Surprisal', 'Inanimate Baseline', 'Animate Baseline'], ylabel='surprisal (bits)', ylim=(0,22))

ax.legend(loc = "upper right")
ax.xaxis.set_tick_params(rotation=0)
ax.set_title("Dutch Context Experiment")
fig = ax.get_figure()
fig.tight_layout()
fig.show()
fig.savefig('paper-plots-dutch/p2_bar_chart.pdf')

# %%
