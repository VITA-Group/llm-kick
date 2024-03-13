import argparse

import torch
import torch.nn as nn
import quant
import os 
import pickle
import string
import re
import pickle
import numpy as np
import torch
import pandas as pd
from gptq import GPTQ
from utils import find_layers, DEV, set_seed, get_wikitext2, get_ptb, get_c4, get_ptb_new, get_c4_new, get_loaders
import transformers
from transformers import AutoTokenizer
from triviaqa_evaluation import get_ground_truths
from utils.categories import subcategories, categories

from collections import Counter


def get_llama(model):

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model


def load_quant(model, checkpoint, wbits, groupsize=-1, fused_mlp=True, eval=True, warmup_autotune=True):
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig.from_pretrained(model)

    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    if eval:
        model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    quant.make_quant_linear(model, layers, wbits, groupsize)

    del layers

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint), strict=False)
    else:
        model.load_state_dict(torch.load(checkpoint), strict=False)

    if eval:
        quant.make_quant_attn(model)
        quant.make_quant_norm(model)
        if fused_mlp:
            quant.make_fused_mlp(model)
    if warmup_autotune:
        quant.autotune_warmup_linear(model, transpose=not (eval))
        if eval and fused_mlp:
            quant.autotune_warmup_fused(model)
    model.seqlen = 2048
    print('Done.')

    return model




@torch.no_grad()
def llama_eval(model, testenc, dev, dataset, f):
    print(f'Evaluating {dataset}... \t', file=f)

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = quant.Quantizer()
                quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantizer.quantize(W).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"{ppl.item()}\n", file=f)
    f.flush()

    model.config.use_cache = use_cache

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(example, prompt, include_context=True):
    web_evidence, wiki_evidence = None, None

    if include_context == False: return web_evidence, web_evidence, prompt

    if len(example["search_results"]["filename"]) != 0:
        web_evidence = example["search_results"]["search_context"][0]
    if len(example["entity_pages"]["filename"]) != 0:
        wiki_evidence = example["entity_pages"]["wiki_context"][0]
    return web_evidence, wiki_evidence, prompt

def gen_prompt(example, include_context=True):
    if include_context == True:
        prompt = f"Based on these texts, answer these questions:\n\nQ: {example['question']}\nA:" 
    else:
        prompt = f"Answer these questions:\n\nQ: {example['question']}\nA:" 
    return prompt

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(ground_truth) in normalize_answer(prediction) or normalize_answer(prediction) in normalize_answer(ground_truth)

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

@torch.no_grad()
def evaluation(args, model, tokenizer, dataloader, filename, device, mute = False):
    exact_match, f1, total = 0, 0, 0
    from tqdm import tqdm
    for i in tqdm(range(0, 2000)):
        # get prompt and make sure it fits
        prompt_end = gen_prompt(dataloader[i], include_context=args.include_context)
        web_data, wiki_data, prompt_end = format_example(dataloader[i], prompt_end, include_context=args.include_context)

        context = ""
        if wiki_data != None: context += wiki_data
        if web_data != None: context += web_data
        
        prompt = f"{context[:3072]}\n\n{prompt_end}"
        if i == 0: print(prompt)
        input_ids = tokenizer(prompt.strip(), return_tensors="pt").input_ids.cuda()

        generate_ids = model.generate(input_ids, max_length=len(input_ids[0]) + 32)
        output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        prediction = output.split(prompt_end)[1].strip().split("\n")[0]
        ground_truths = get_ground_truths(dataloader[i])

       
        em_for_this_question = metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        if em_for_this_question == 0 and not mute:
            print("em=0:", prediction, ground_truths)
        exact_match += em_for_this_question
        f1_for_this_question = metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)
        f1 += f1_for_this_question
        total += 1

        if total % 50 == 0:
            print(f"Count: {total} \t\t Exact: {100.0 * exact_match / total:.3f} \t\t F1: {100.0 * f1 / total:.3f}", file=filename)
            filename.flush()

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default="lmsys/vicuna-7b-v1.3", type=str, help='llama model to load')
    parser.add_argument('--dataset', default="c4", type=str, choices=['wikitext2', 'ptb', 'c4'], help='Where to extract calibration data from.')
    parser.add_argument('--wbits', type=int, default=8,  help='#bits to use for quantization; use 16 for evaluating base model.') #2, 4, 8
    parser.add_argument('--groupsize', type=int, default=128, help='Groupsize to use for quantization; default uses full row.')
    parser.add_argument('--load', type=str, default='/data/ajay_data/MCI/quantized_llama/vicuna7b-8bit-128g.pt', help='Load quantized model.')

    parser.add_argument('--text', default="Question: In which city where the 1956 Summer Olympic games held? The answer is ", type=str, help='input text')

    parser.add_argument('--min_length', type=int, default=10, help='The minimum length of the sequence to be generated.')

    parser.add_argument('--max_length', type=int, default=50, help='The maximum length of the sequence to be generated.')

    parser.add_argument('--top_p',
                        type=float,
                        default=0.95,
                        help='If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.')

    parser.add_argument('--temperature', type=float, default=0.8, help='The value used to module the next token probabilities.')

    parser.add_argument('--device', type=int, default=-1, help='The device used to load the model when using safetensors. Default device is "cpu" or specify, 0,1,2,3,... for GPU device.')

    # fused mlp is sometimes not working with safetensors, no_fused_mlp is used to set fused_mlp to False, default is true
    parser.add_argument('--fused_mlp', action='store_true')
    parser.add_argument('--no_fused_mlp', dest='fused_mlp', action='store_false')
    parser.set_defaults(fused_mlp=True)

    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nearest', action='store_true', help='Whether to run the RTN baseline.')
    parser.add_argument('--include_context', action="store_true")

    args = parser.parse_args()
    if args.model == "lmsys/vicuna-7b-v1.3":
        file_name = open(f"./runs/vicuna7b/triviaqa_{args.wbits}bits_{args.groupsize}_group_{args.include_context}_context.txt", "a")
    elif args.model == "lmsys/vicuna-13b-v1.3":
        file_name = open(f"./runs/vicuna13b/triviaqa_{args.wbits}bits_{args.groupsize}_group_{args.include_context}_context.txt", "a")
    print(f"{args}\n", file=file_name)
    file_name.flush()

    if type(args.load) is not str:
        args.load = args.load.as_posix()

    # Load the quantized Model
    model = load_quant(args.model, args.load, args.wbits, args.groupsize, fused_mlp=args.fused_mlp)
    model.eval()

    
    model.to(DEV)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    print("Model and Tokenizer Created \n", file = file_name)

    input_ids = tokenizer(args.text, return_tensors="pt").input_ids.to(DEV)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=True,
            min_length=args.min_length,
            max_length=args.max_length,
            top_p=args.top_p,
            temperature=args.temperature,
        )
    # print(tokenizer.decode([el.item() for el in generated_ids[0]]))
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(output)
    import sys
    sys.exit(0)

    ################# Perplexity Evaluation #####################
    # datasets = ['wikitext2', 'ptb', 'c4']
    # datasets = ['wikitext2', 'ptb-new', 'c4-new']
    # for dataset in datasets:
    #     dataloader, testloader = get_loaders(dataset, seed=args.seed, model=args.model, seqlen=model.seqlen)
    #     print(dataset)
    #     llama_eval(model, testloader, DEV, dataset, file_name)

    ################# Quantized Model Loaded #####################

    
    from timeit import default_timer as timer
    from datetime import timedelta
    start_time = timer()
    ####################### Details of the TriviaQA Benchmark ##########################
    
    from datasets import load_dataset
    dataset = load_dataset("trivia_qa", "rc")
    result = evaluation(args, model, tokenizer, dataset["validation"], file_name, DEV)
    print("----------------------------------------", file=file_name)
    print(f"Results: {result}", file=file_name)
    print(f"\n\n\n -------- Total time taken:   {timedelta(seconds=timer() - start_time)}-----------", file=file_name)
    file_name.flush()


    file_name.close()