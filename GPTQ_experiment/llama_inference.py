import argparse

import torch
import torch.nn as nn
import quant

from gptq import GPTQ
from utils import find_layers, DEV, set_seed, get_wikitext2, get_ptb, get_c4, get_ptb_new, get_c4_new, get_loaders
import transformers
from transformers import AutoTokenizer


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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default="lmsys/vicuna-7b-v1.3", type=str, help='llama model to load')
    parser.add_argument('--dataset', default="c4", type=str, choices=['wikitext2', 'ptb', 'c4'], help='Where to extract calibration data from.')
    parser.add_argument('--wbits', type=int, default = 4, choices=[2, 3, 4, 8, 16], help='#bits to use for quantization; use 16 for evaluating base model.')
    parser.add_argument('--groupsize', type=int, default = 128, help='Groupsize to use for quantization; default uses full row.')
    parser.add_argument('--load', type=str, default='', help='Load quantized model.')

    parser.add_argument('--text', default="Please provide answer to the following. Question: By what name is Allen Konigsberg better known? The answer is ", type=str, help='input text')

    parser.add_argument('--min_length', type=int, default=10, help='The minimum length of the sequence to be generated.')

    parser.add_argument('--max_length', type=int, default=250, help='The maximum length of the sequence to be generated.')

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

    args = parser.parse_args()

    if type(args.load) is not str:
        args.load = args.load.as_posix()

    if args.load:
        model = load_quant(args.model, args.load, args.wbits, args.groupsize, fused_mlp=args.fused_mlp)
    else:
        model = get_llama(args.model)
        model.eval()

    model.to(DEV)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    # input_ids = tokenizer.encode(args.text, return_tensors="pt").to(DEV)
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
