import os
import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, check_sparsity, find_layers
from lib.eval import eval_ppl
from freebase_qa import FreebaseQA

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


def validate_response(correct_answer, generated_output):
    generated_output = generated_output.strip().replace(" ", "").lower()
    correct_answer = [item.strip().replace(" ", "").lower() for item in correct_answer]
    for ans in correct_answer:
        if ans in generated_output: return 1
    return 0

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
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0.0, help='Sparsity level')
    parser.add_argument("--sparsity_type", default="unstructured", type=str, choices=["unstructured", "4:8", "2:4", "1:2"])
    parser.add_argument("--prune_method", default="magnitude" ,type=str, choices=["magnitude", "wanda", "sparsegpt"])
    parser.add_argument("--cache_dir", default="/data/ajay_data/MCI/llm_weights", type=str )
    parser.add_argument("--prompt", default="None" , type=str)
    parser.add_argument("--model_type", default="vicuna7b" , type=str)
    parser.add_argument("--num_examples", type=int, default=1000, help='Number of examples to run the evaluation.')
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    args = parser.parse_args()

    freebase_qa = FreebaseQA()
    freebase_filepath = "./datasets/freebase/FreebaseQA-train.json"
    exact_match = 0

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

    file_name = open(f"./logs/freebase_{args.model_type}_{args.prune_method}_{args.sparsity_ratio}_{args.sparsity_type}", "a")
    print(f"{args}\n", file=file_name)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
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
    
    num_example = 0
    for question, answers in freebase_qa._generate_examples(freebase_filepath):
        
        if num_example > args.num_examples: break
        lamma_prompt = f"Please give answer to this question: {question}\nThe answer is "
        inputs = tokenizer(lamma_prompt, return_tensors="pt")
        inputs = inputs.to(device)

        generate_ids = model.generate(inputs.input_ids, max_length=inputs["input_ids"].shape[-1] * 3)
        output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        is_match = validate_response(answers, output)
        exact_match += is_match
        print(f"Prompt: {question} || {answers}\nResponse: {output}\nIS_EXACT_MATCH: {is_match}\n", file=file_name)
        file_name.flush()
        num_example += 1
        
    
    print(f"Exact match: {exact_match}/{args.num_examples} || Accuracy : {100 * (exact_match/args.num_examples):.2f}%", file=file_name)
    file_name.close()

if __name__ == '__main__':
    main()