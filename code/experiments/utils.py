import os

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer

llama_checkpoint_path = os.environ['LLAMA_CHECKPOINT_PATH']

def trim_sentence(sentence: str, word: str) -> str:
    # find the word in the sentence and remove it and all following text, including the preceding space
    try:
        idx = sentence.index(word)
    except ValueError as e:
        print(f"Couldn't find {word} in {sentence}")
        raise e
    return sentence[:idx - 1]

def load_model(model_name:str, multi_gpu:bool):
    if 'llama' in model_name:
        _, model_size = model_name.split('-')
        tokenizer = LlamaTokenizer.from_pretrained(f"{llama_checkpoint_path}/{model_size}")
        if multi_gpu:
            model = LlamaForCausalLM.from_pretrained(f"{llama_checkpoint_path}/{model_size}", device_map='auto')
            model.eval()
        else:
            model = LlamaForCausalLM.from_pretrained(f"{llama_checkpoint_path}/{model_size}")
            model.eval()
            model.to('cuda')
    else:
        use_fast = 'opt' not in model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)

        if multi_gpu:
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
            model.eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.eval()
            model.to('cuda')
    return tokenizer, model
