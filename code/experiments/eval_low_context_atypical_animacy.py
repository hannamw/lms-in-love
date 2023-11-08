#%%
import json
from collections import defaultdict
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn.functional as F

from utils import load_model

parser = ArgumentParser()
parser.add_argument('-m', "--model", type=str, required=True)
parser.add_argument('-n', '--n_samples', type=int, default=10000)
parser.add_argument('--multi_gpu', action='store_true')
parser.add_argument('-d', '--split', type=str, default='')
args = parser.parse_args()


experiment_name = args.model.split('/')[-1] + args.split
df = pd.read_csv(f'data/low_context_atypical_animacy{args.split}.csv')
tokenizer, model = load_model(args.model, args.multi_gpu)

def run_sentence(sentence: str) -> torch.Tensor:
    with torch.inference_mode():
        logits = model(**tokenizer(sentence, return_tensors='pt').to('cuda')).logits.squeeze(0)[-1]
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs


d = defaultdict(list)
names = ['animate_no_context', 'inanimate_no_context', 'experimental']

for i, example in tqdm(df.iterrows(), total=df.shape[0]):
    log_probs_animate_no_context = run_sentence(example['animate_no_context'])
    log_probs_inanimate_no_context = run_sentence(example['inanimate_no_context'])
    log_probs_experimental = run_sentence(example['experimental'])

    dists = [log_probs_animate_no_context, log_probs_inanimate_no_context, log_probs_experimental]
    for dist1, dist1_name in zip(dists, names):
        for dist2, dist2_name in zip(dists, names):
            # with torch's F.kl_div, the second argument is the target, i.e. P in KL(P||Q)
            kl = F.kl_div(dist2, dist1, log_target=True).item()
            d[f'{dist1_name}_{dist2_name}_kl'].append(kl)

    probs = log_probs_experimental.exp()
    probs_sorted = probs.sort(descending=True)
    top5_indices, top5_probs = probs_sorted.indices[:5].tolist(), probs_sorted.values[:5].tolist()
    for i, (index, prob) in enumerate(zip(top5_indices, top5_probs)):
        token = tokenizer._convert_id_to_token(index)
        d[f'top{i}_word'].append(token)
        d[f'top{i}_prob'].append(prob)

for k, v in d.items():
    df[k] = v

Path('results/experiment3').mkdir(exist_ok=True, parents=True)
df.to_csv(f'results/experiment3/{experiment_name}.csv')
summary_dict = {column: df[column].mean() for column in df.columns if df[column].dtype == np.float64}

with open(f'results/experiment3/{experiment_name}.json', 'w') as f:
    json.dump(summary_dict, f)
