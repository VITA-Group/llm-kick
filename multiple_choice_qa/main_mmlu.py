import os
import argparse
import os 
import pickle
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, check_sparsity, find_layers
from lib.eval import eval_ppl
from categories import subcategories, categories

from timeit import default_timer as timer
from datetime import timedelta

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

choices = ["A", "B", "C", "D"]


def save_dict(item, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(item, handle, protocol=pickle.HIGHEST_PROTOCOL)

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval(args, subject, model, tokenizer, dev_df, test_df, f):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        print(prompt)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        while input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        label = test_df.iloc[i, test_df.shape[1] - 1]

        generate_ids = model.generate(input_ids, max_length=len(input_ids[0]) + 1)
        output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        pred = output[-1:]

        cor = pred == label
        cors.append(cor)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject), file=f)
    f.flush()

    return cors, acc, all_probs


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
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    args = parser.parse_args()


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
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
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

    
    start_time = timer()
    ####################### Details of the MMLU Benchmark ##########################
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs = eval(args, subject, model, tokenizer, dev_df, test_df, file_name)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(args.model)] = cors
        

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat), file=file_name)
        file_name.flush()
    print("----------------------------------------", file=file_name)
    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat), file=file_name)
        file_name.flush()
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("----------------------------------------", file=file_name)
    print("Average accuracy: {:.3f}".format(weighted_acc), file=file_name)
    print(f"\n\n\n -------- Total time taken:   {timedelta(seconds=timer() - start_time)}-----------", file=file_name)
    file_name.flush()


    data_to_save = {}
    data_to_save["category_acc"] = cat_cors
    data_to_save["sub_category_acc"] = subcat_acc
    data_to_save["all_cors"] = all_cors

    file_name.close()

if __name__ == '__main__':
    main()