import os


import argparse
import os 
import pickle
import string
import re
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
from tqdm import tqdm
from collections import Counter
from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, check_sparsity, find_layers
from lib.eval import eval_ppl
from utils.triviaqa_evaluation import get_ground_truths, normalize_answer

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


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
def eval(args, model, tokenizer, dataloader, filename, device, mute = False):
    exact_match, f1, total = 0, 0, 0
    for i in tqdm(range(0, args.num_examples)):
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

def get_llm(model, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = 2048
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random', type=bool, default=False, help='Random pruning')
    parser.add_argument("--model_type", default="vicuna7b" , type=str)
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0.5, help='Sparsity level')
    parser.add_argument("--sparsity_type", default="unstructured", type=str, choices=["unstructured", "4:8", "2:4", "1:2"])
    parser.add_argument("--prune_method", default="magnitude" ,type=str, choices=["magnitude", "wanda", "sparsegpt"])
    parser.add_argument("--cache_dir", default="/data/ajay_data/MCI/llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument("--num_examples", type=int, default=10, help='Number of examples to run the evaluation.')
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument('--include_context', action="store_true")
    args = parser.parse_args()
    device = torch.device(f"cuda:0")

    if args.model_type == "vicuna7b":
        args.model = "lmsys/vicuna-7b-v1.3"
    elif args.model_type == "vicuna13b":
        args.model = "lmsys/vicuna-13b-v1.3"
    elif args.model_type == "vicuna33b":
        args.model = "lmsys/vicuna-33b-v1.3"
    elif args.model_type == "llama7b":
        args.model = "decapoda-research/llama-7b-hf"
    else:
        args.model = "lmsys/vicuna-7b-v1.3"


    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    file_name = open(f"./logs/icrqa_{args.model_type}_{args.prune_method}_{args.sparsity_ratio}_{args.sparsity_type}_{args.include_context}", "a")
    print(f"{args}\n", file=file_name)

    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)


    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            print("Pruning Wanda")
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            print("Pruning Magnitude")
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            print("Pruning SparseGPT")
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"Sparsity sanity check {sparsity_ratio:.4f} ({args.sparsity_type}, {args.prune_method})", file=file_name)
    print("*"*30)
    ################################################################


    from timeit import default_timer as timer
    from datetime import timedelta
    start_time = timer()
    ####################### Details of the TriviaQA Benchmark ##########################
    
    from datasets import load_dataset
    dataset = load_dataset("trivia_qa", "rc", ignore_verifications=True)
    result = eval(args, model, tokenizer, dataset["validation"], file_name, device)
    print("----------------------------------------", file=file_name)
    print(f"Results: {result}", file=file_name)
    print(f"\n\n\n -------- Total time taken:   {timedelta(seconds=timer() - start_time)}-----------", file=file_name)
    file_name.flush()

    file_name.close()

if __name__ == '__main__':
    main()