from argparse import ArgumentParser
from pathlib import Path

import jsonlines
import torch
import pandas as pd

from utils import trim_sentence, load_model

parser = ArgumentParser()

parser.add_argument('-m', "--model", type=str, required=True)
parser.add_argument('--multi_gpu', action='store_true')
parser.add_argument('--use_animate', action='store_true')

args = parser.parse_args()

experiment_name_base = args.model.split('/')[-1]
tokenizer, model = load_model(args.model, args.multi_gpu)

df = pd.read_csv('data/stories_boudewyn.csv')

for story_column, experiment_name in [('inanimate_story', experiment_name_base), ('animate_story', experiment_name_base + '_animate')]:
    probs = torch.full([len(df), 2], -1)
    for i, (story, w1, w2) in enumerate(zip(df[story_column], df['action1'], df['action2'])):
        sentences = story.strip().split('.')
        s1, s2, s3, s4, s5 = sentences

        partial_paragraphs = [[], sentences[:1]]

        for j, (sent, partial_paragraph, word) in enumerate(zip([s1, s2], partial_paragraphs, [w1, w2])):
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
    d = {'story': [(i + 1) for i in range(len(df))], 't1': probs[:, 0], 't2': probs[:, 1],
        't1_surprisal': surprisals[:,0], 't2_surprisal': surprisals[:, 1]}

    output_df = pd.DataFrame.from_dict(d)
    output_df = output_df.set_index('story')
    Path('results/boudewyn_exp1').mkdir(exist_ok=True, parents=True)
    output_df.to_csv(f'results/boudewyn_exp1/{experiment_name}.csv')

    t2t1 = surprisals[:, 1] - surprisals[:, 0]

    t1s = surprisals[:, 0]
    t2s = surprisals[:, 1]

    output_dict = {'surprisal_t1': t1s.mean(), 'surprisal_t2': t2s.mean(),
                'surprisal_std_t1': t1s.std(), 'surprisal_std_t2': t2s.std(),}

    with jsonlines.open(f'results/boudewyn_exp1/{experiment_name}.jsonl', mode='w') as writer:
        writer.write(output_dict)
