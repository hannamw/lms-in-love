from argparse import ArgumentParser
from pathlib import Path

import jsonlines
import torch
import pandas as pd

from utils import load_model, trim_sentence

parser = ArgumentParser()

parser.add_argument('-m', "--model", type=str, default="gpt2-xl")
parser.add_argument('--multi_gpu', action='store_true')
parser.add_argument('--dutch', action='store_true')

args = parser.parse_args()
experiment_name_base = args.model.split('/')[-1]
if args.dutch:
    experiment_name_base += '_dutch'
    df = pd.read_csv('data/stories_exp1_dutch_formatted.csv')
else:
    df = pd.read_csv('data/stories_exp1_edited.csv')

tokenizer, model = load_model(args.model, args.multi_gpu)

for stories, ws, experiment_name in [(df['story1'], df['word1'], experiment_name_base), (df['story2'], df['word2'], experiment_name_base + "_animate")]:
    probs = torch.full([len(df), 3], -1)
    for i, (story, word) in enumerate(zip(stories, ws)):
        sentences = story.strip().split('.')
        s1, s2, s3, s4, s5, _, _ = sentences

        partial_paragraphs = [[], sentences[:2], sentences[:4]]

        for j, (sent, partial_paragraph) in enumerate(zip([s1, s3, s5], partial_paragraphs)):
            trimmed = trim_sentence(sent, word)
            base_text = '.'.join(partial_paragraph + [trimmed])
            base_text_tokenized = tokenizer(base_text, return_tensors='pt').to('cuda')

            word_text = f'{base_text} {word}'
            word_text_tokenized = tokenizer(word_text, return_tensors='pt').to('cuda')
            word_length = word_text_tokenized['input_ids'].size(-1) - base_text_tokenized['input_ids'].size(-1)
            logits = model(**word_text_tokenized).logits
            sentence_probs = logits.softmax(-1).squeeze(0)[-(word_length+1):-1]

            idx = torch.arange(word_length).to('cuda')
            word_prob = torch.prod(sentence_probs[idx, word_text_tokenized['input_ids'].squeeze(0)[-word_length:]])
            probs[i,j] = word_prob

    assert torch.all(probs >= 0)
    surprisals = -torch.log2(probs)
    d = {'story': [(i + 1) for i in range(len(df))], 't1': probs[:, 0], 't2': probs[:, 1], 't3': probs[:, 2],
        't1_surprisal': surprisals[:,0], 't2_surprisal': surprisals[:, 1], 't3_surprisal': surprisals[:, 2]}

    output_df = pd.DataFrame.from_dict(d)
    output_df = output_df.set_index('story')
    Path('results/peanuts_exp1').mkdir(exist_ok=True, parents=True)
    output_df.to_csv(f'results/peanuts_exp1/{experiment_name}.csv')

    t2t1 = surprisals[:, 1] - surprisals[:, 0]
    t3t1 = surprisals[:, 2] - surprisals[:, 0]
    t3t2 = surprisals[:, 2] - surprisals[:, 1]

    t1s = surprisals[:, 0]
    t3s = surprisals[:, 1]
    t5s = surprisals[:, 2]

    output_dict = {'surprisal_t1': t1s.mean(), 'surprisal_t3': t3s.mean(), 'surprisal_t5': t5s.mean(),
                'surprisal_std_t1': t1s.std(), 'surprisal_std_t3': t3s.std(), 'surprisal_std_t5': t5s.std()}

    with jsonlines.open(f'results/peanuts_exp1/{experiment_name}.jsonl', mode='w') as writer:
        writer.write(output_dict)
