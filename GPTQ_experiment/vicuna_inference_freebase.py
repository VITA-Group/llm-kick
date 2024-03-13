import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
import torch
import torch.nn as nn
import quant
import os 
import pickle
import numpy as np
import torch
import pandas as pd
from gptq import GPTQ
from utils import find_layers, DEV, set_seed, get_wikitext2, get_ptb, get_c4, get_ptb_new, get_c4_new, get_loaders
import transformers
from transformers import AutoTokenizer

from utils.categories import subcategories, categories


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
def evaluation(args, subject, model, tokenizer, dev_df, test_df, f):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]
    # print(check_sparsity(model))
    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

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

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0.0
        sub_params = 0.0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f} | {sub_params}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 


def validate_response(correct_answer, generated_output):
    generated_output = generated_output.strip().replace(" ", "").lower()
    correct_answer = [item.strip().replace(" ", "").lower() for item in correct_answer]
    for ans in correct_answer:
        if ans in generated_output: return 1
    return 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default="lmsys/vicuna-7b-v1.3", type=str, help='llama model to load')
    parser.add_argument('--dataset', default="wikitext2", type=str, choices=['wikitext2', 'ptb', 'c4'], help='Where to extract calibration data from.')
    parser.add_argument('--wbits', type=int, default=8,  help='#bits to use for quantization; use 16 for evaluating base model.') #2, 4, 8
    parser.add_argument('--groupsize', type=int, default=128, help='Groupsize to use for quantization; default uses full row.')
    parser.add_argument('--load', type=str, default='/data/ajay_data/MCI/quantized_llama/vicuna7b-2bit-128g.pt', help='Load quantized model.')

    parser.add_argument('--text', default="Please provide answer to the following. Question: In which city where the 1956 Summer Olympic games held? The answer is ", type=str, help='input text')

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


    args = parser.parse_args()
    file_name = open(f"./runs/llama7b/freebase_{args.wbits}bits_{args.groupsize}_group.txt", "a")
    # file_name = open(f"test.txt", "a")
    print(f"{args}\n", file=file_name)
    file_name.flush()

    if type(args.load) is not str:
        args.load = args.load.as_posix()

    # Load the quantized Model
    model = load_quant(args.model, args.load, args.wbits, args.groupsize, fused_mlp=args.fused_mlp)
    model.eval()

    
    model.to(DEV)
    # tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    print("Model and Tokenizer Created \n", file = file_name)

    ################# Freebase evaluation ########
    from freebase_qa import FreebaseQA
    freebase_qa = FreebaseQA()
    freebase_filepath = "/home/aj32632/MCI/wanda/datasets/freebase/FreebaseQA-train.json"
    exact_match = 0

    if  "llama" in args.model:
        tokenizer = AutoTokenizer.from_pretrained("/home/aj32632/MCI/wanda/tokenizer/", use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    num_examples = 1
    for question, answers in freebase_qa._generate_examples(freebase_filepath):
        
        if num_examples > 1000: break
        lamma_prompt = f"Please give answer to this question: {question} The answer is "
        inputs = tokenizer(lamma_prompt, return_tensors="pt")
        inputs = inputs.to(DEV)

        generate_ids = model.generate(inputs.input_ids, max_length=inputs["input_ids"].shape[-1] * 3)
        output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        is_match = validate_response(answers, output)
        exact_match += is_match
        print(f"Prompt: {question} || {answers}\nResponse: {output}\nIS_EXACT_MATCH: {is_match}\n", file=file_name)
        file_name.flush()
        # if num_examples % 10 == 0:  print(f"Exact match: {exact_match}/{num_examples} || Accuracy : {100 * (exact_match/num_examples):.2f}%")
        num_examples += 1
        
    
    print(f"Exact match: {exact_match}/{num_examples} || Accuracy : {100 * (exact_match/num_examples):.2f}%", file=file_name)
    file_name.close()
