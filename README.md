# Code Release for "When Language Models Fall in Love: Animacy Processing in Transformer Language Models"

This folder contains the code for the 2023 EMNLP paper "When Language Models Fall in Love: Animacy Processing in Transformer Language Models".

## Data
In the `datasets/` folder, find the datasets we created for our experiments. They are as follows: 
- Repetition Experiment: `repetition_stories.txt` 
- Context Experiment: `context_stories.txt`
- Low-Context Atypical Animacy Experiment: `low_context_atypical_animacy.txt`
- `appendix/`
    - Low-Context Atypical Animacy Experiment (Large Pool): `low_context_atypical_animacy_large_pool.txt`
    - Low-Context Atypical Animacy Experiment (Matched Frequency): `low_context_atypical_animacy_matched.txt`
    - Low-Context Atypical Animacy Experiment (Cataphor): `low_context_atypical_animacy_cataphor.txt`

The original Dutch stories from Nieuwland and van Berkum (2006) can be found on [ResearchGate](https://www.researchgate.net/publication/6946958_When_Peanuts_Fall_in_Love_N400_Evidence_for_the_Power_of_Discourse).

To view the other two datasets used, please go to the sources of [BLiMP](https://github.com/alexwarstadt/blimp/tree/master/data) and [Boudewyn et al. (2016)](https://swaab.faculty.ucdavis.edu/stimuli/)

## Code
In the `code/` folder, find the implementation of our experiments. This includes an `environment.yml` file that can be used to install a conda environment that is compatible with this codebase. Note that as our experiments include LLaMA, you must request the (private) weights of this model from Meta. You should then specify the path to these weights in the `LLAMA_CHECKPOINT_PATH` environment variable.

To run the experiments, please first retrieve the two BLiMP datasets `animate_subject_trans.jsonl` and `animate_subject_passive.jsonl`, and place them in the `code/data/` folder.

While in the `code/` folder, run every `.py` file in the `experiments/` folder starting with `eval`. Each such file takes in an argument `-m` to specify the name of the model to run it with. If the model needs to be split across multiple GPUs, set the `--multi_gpu` flag. If you want to reproduce the Dutch experiments, run the `peanuts` files in `experiments/` with the `--dutch` argument (only applicable for `peanuts` experiments). If you'd like to replicate the low-context atypical animacy results in the appendix, run `eval_low_context_atypical_animacy.py`, setting the `--split` argument to one of `[_large_pool, _matched, _cataphor]`.

Once you have done this, all results will be in the `results/` folder. To reproduce the plots in the paper, please run the `.py` files in `results`; `paper_plots_dutch.py` is only necessary if you want the Dutch plots. The PDF outputs will be in `results/paper-plots/`.
