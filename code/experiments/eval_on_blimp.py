from argparse import ArgumentParser
from pathlib import Path

import jsonlines
import torch
import pandas as pd

from utils import load_model

parser = ArgumentParser()
parser.add_argument('-m', "--model", type=str, required=True)
parser.add_argument('--multi_gpu', action='store_true')

args = parser.parse_args()
model_name = args.model.split('/')[-1]
tokenizer, model = load_model(args.model, args.multi_gpu)

accs = {}
for blimp_split in ["animate_subject_trans", "animate_subject_passive"]:
    with jsonlines.open(f'data/{blimp_split}.jsonl') as reader:
        dataset = [(line['sentence_good'], line['sentence_bad']) for line in reader]

    probs = torch.full([len(dataset), 2], -1)
    with torch.inference_mode():
        for i, (good_sentence, bad_sentence) in enumerate(dataset):

            # note that one might consider inserting the BOS token here for OPT models
            # but we found that this doesn't make a big difference
            good_inputs = tokenizer(good_sentence, return_tensors="pt").to('cuda')
            good_outputs = model(**good_inputs).logits

            good_input_ids = good_inputs['input_ids'].reshape(-1, 1)
            good_probs = good_outputs.softmax(-1).squeeze(0)[:-1, :].gather(1, good_input_ids[1:])
            good_prob = good_probs.prod().cpu().item()
            probs[i, 0] = good_prob

            # as above
            bad_inputs = tokenizer(bad_sentence, return_tensors="pt").to('cuda')
            bad_outputs = model(**bad_inputs).logits

            bad_input_ids = bad_inputs['input_ids'].reshape(-1, 1)
            bad_probs = bad_outputs.softmax(-1).squeeze(0)[:-1, :].gather(1, bad_input_ids[1:])
            bad_prob = bad_probs.prod().cpu().item()
            probs[i, 1] = bad_prob

    assert torch.all(probs >= 0)
    
    corrects = probs[:, 0] > probs[:, 1]
    accuracy = torch.mean(corrects)
    accs[blimp_split] = accuracy
    surprisals = -torch.log2(probs)

    good_sentences, bad_sentences = zip(*dataset)
    d = {'good_sentence': good_sentences, 'bad_sentence': bad_sentences,
         'surprisal_good': surprisals[:, 0], 'surprisal_bad': surprisals[:, 1]}
    output_df = pd.DataFrame.from_dict(d)
    Path('results/blimp').mkdir(exist_ok=True, parents=True)
    output_df.to_csv(f'results/blimp/{blimp_split}_{model_name}.csv')

with jsonlines.open(f'results/blimp/{model_name}.jsonl', mode='w') as writer:
    writer.write(accs)
