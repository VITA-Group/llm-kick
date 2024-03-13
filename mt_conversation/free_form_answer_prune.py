import os
os.environ['CUDA_VISIBLE_DEVICES'] = "6,7"
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import ray
from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, check_sparsity, find_layers
from fastchat.model import get_conversation_template


def run_eval(model_path, model_id, question_file, answer_file, num_gpus, args):
    # split question file into num_gpus files
    ques_jsons = []
    with open(os.path.expanduser(question_file), "r") as ques_file:
        for line in ques_file:
            ques_jsons.append(line)

    chunk_size = len(ques_jsons) // num_gpus
    ans_handles = []
    for i in range(0, len(ques_jsons), chunk_size):
        ans_handles.append(
            get_model_answers(
                model_path, model_id, ques_jsons[i : i + chunk_size], args
            )
        )

    ans_jsons = []
    for ans_handle in ans_handles:
        ans_jsons.extend(ans_handle)

    with open(os.path.expanduser(answer_file), "w") as ans_file:
        for line in ans_jsons:
            ans_file.write(json.dumps(line) + "\n")

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


@torch.inference_mode()
def get_model_answers(model_path, model_id, question_jsons, args):

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model = get_llm(model_id, args.cache_dir)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))


    device = torch.device("cuda:0")
    if "30b" in args.model_id or "65b" in args.model_id: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
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
    print(f"Sparsity sanity check {sparsity_ratio:.4f} ({args.sparsity_type}, {args.prune_method})")
    print("*"*30)
    ################################################################

    ans_jsons = []
    for i, line in enumerate(tqdm(question_jsons)):
        ques_json = json.loads(line)
        idx = ques_json["question_id"]
        qs = ques_json["text"]
        conv = get_conversation_template(model_id)
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer([prompt]).input_ids
        output_ids = model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=True,
            temperature=0.7,
            max_new_tokens=1024,
        )
        output_ids = output_ids[0][len(input_ids[0]) :]
        outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        ans_id = shortuuid.uuid()
        ans_jsons.append(
            {
                "question_id": idx,
                "text": outputs,
                "answer_id": ans_id,
                "model_id": model_id,
                "metadata": {},
            }
        )
    return ans_jsons


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--model-id", default = "lmsys/vicuna-7b-v1.3", type=str)
    parser.add_argument("--question-file", type=str, default="./table/question.jsonl")
    parser.add_argument("--answer-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--cache_dir", default="/data/ajay_data/MCI/llm_weights", type=str )
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0.1, help='Sparsity level')
    parser.add_argument("--sparsity_type", default="unstructured", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", default="magnitude" ,type=str, choices=["magnitude", "wanda", "sparsegpt"])
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    args = parser.parse_args()
    import numpy as np
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    
    file_name = f"./runs/vicuna7b_free_form/{args.prune_method}/answer_{args.sparsity_ratio}_{args.sparsity_type}.jsonl"
    # ray.init()
    run_eval(
        args.model_path,
        args.model_id,
        args.question_file,
        file_name,
        args.num_gpus,
        args
    )