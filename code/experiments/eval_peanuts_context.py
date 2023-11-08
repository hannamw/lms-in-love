from pathlib import Path 
from argparse import ArgumentParser

import jsonlines
import torch
import pandas as pd
import numpy as np

from utils import trim_sentence, load_model

parser = ArgumentParser()
parser.add_argument('-m', "--model", type=str, required=True)
parser.add_argument('--multi_gpu', action='store_true')
parser.add_argument('--dutch', action='store_true')

args = parser.parse_args()

experiment_name = args.model.split('/')[-1]
if args.dutch:
    experiment_name += '_dutch'


tokenizer, model = load_model(args.model, args.multi_gpu)

if args.dutch:
    df = pd.read_csv('data/stories_exp2_dutch_formatted.csv')
else:
    df = pd.read_csv('data/stories_exp2_formatted.csv')

probs = torch.full([len(df), 2], -1)
baseline_probs = torch.full([len(df), 2], -1)
target_token = '[target]'


for i, (s, w1, w2) in enumerate(zip(df['story'], df['word1'], df['word2'])):
    s = s.strip()
    trimmed_base = trim_sentence(s, target_token)

    for j, w in enumerate([w1,w2]):
        w = w.strip()
        trimmed_with_w = f'{trimmed_base} {w}'
        baseline_sentence = trimmed_with_w.split('.')[-1]

        base_text_tokenized = tokenizer(trimmed_base, return_tensors='pt').to('cuda')
        word_text_tokenized = tokenizer(trimmed_with_w, return_tensors='pt').to('cuda')
        word_length = word_text_tokenized['input_ids'].size(-1) - base_text_tokenized['input_ids'].size(-1)
        idx = torch.arange(word_length).to('cuda')
        word_tokens = word_text_tokenized['input_ids'].squeeze(0)[-word_length:]


        whole_logits = model(**word_text_tokenized).logits
        whole_sentence_probs = whole_logits.softmax(-1).squeeze(0)[-(word_length+1):-1]

        whole_word_prob = torch.prod(whole_sentence_probs[idx, word_tokens])
        probs[i,j] = whole_word_prob


        baseline_text_tokenized = tokenizer(baseline_sentence, return_tensors='pt').to('cuda')
        baseline_logits = model(**baseline_text_tokenized).logits
        baseline_sentence_probs = baseline_logits.softmax(-1).squeeze(0)[-(word_length+1):-1]

        baseline_word_prob = torch.prod(baseline_sentence_probs[idx, word_tokens])
        baseline_probs[i,j] = baseline_word_prob

assert torch.all(probs >= 0)
surprisals = -torch.log2(probs)
baseline_surprisals = -torch.log2(baseline_probs)

d = {'story': [(i + 1) for i in range(len(df))], 'inanimate': probs[:, 0], 'animate': probs[:, 1],
     'inanimate_surprisal': surprisals[:, 0], 'animate_surprisal': surprisals[:, 1],
     'baseline_inanimate_surprisal': baseline_surprisals[:, 0], 'baseline_animate_surprisal': baseline_surprisals[:, 1]}

output_df = pd.DataFrame.from_dict(d)
output_df = output_df.set_index('story')
Path('results/peanuts_exp2').mkdir(exist_ok=True, parents=True)
output_df.to_csv(f'results/peanuts_exp2/{experiment_name}.csv')
inanimate_proportion = np.mean(output_df['inanimate'] > output_df['animate'])
animate_proportion = 1 - inanimate_proportion

print(f"Inanimate: {inanimate_proportion:.2f}, Animate: {animate_proportion:.2f}")
output_dict = {'inanimate_surprisal': surprisals[:,0].mean(), 'animate_surprisal': surprisals[:,1].mean(),
               'inanimate_surprisal_std': surprisals[:,0].std(), 'animate_surprisal_std': surprisals[:,1].std(),
               'inanimate_proportion': inanimate_proportion, 'animate_proportion': animate_proportion,
               'baseline_inanimate_surprisal': baseline_surprisals[:, 0].mean(),
               'baseline_animate_surprisal': baseline_surprisals[:, 1].mean(),
               'baseline_inanimate_surprisal_std': baseline_surprisals[:, 0].std(),
               'baseline_animate_surprisal_std': baseline_surprisals[:, 1].std()
               }
with jsonlines.open(f'results/peanuts_exp2/{experiment_name}.jsonl', mode='w') as writer:
    writer.write(output_dict)