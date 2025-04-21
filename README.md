# [ICLR 2024] Compressing LLMs: The Truth is Rarely Pure and Never Simple
---
```This code is a reproduced (unofficial) version of the work done during the internship at Apple```

> **Authors:** [Ajay Jaiswal](https://ajay1994.github.io/), [Zhe Gan](https://zhegan27.github.io/), [Xianzhi Du](https://www.linkedin.com/in/xianzhi-du-1b128934), [Bowen Zhang](https://zbwglory.github.io/), [Zhangyang Wang](https://vita-group.github.io/index.html), [Yinfei Yang](https://www.linkedin.com/in/yinfeiy)

> **Paper Link:** https://arxiv.org/abs/2310.01382

---  
## Overview
<img width="1090" alt="image" src="https://github.com/Ajay1994/llm_kick/assets/6660499/76b6f0a6-5eef-44cf-8b94-17ef44e58be8">

> [!Note]
> 1.  We introduce **Knowledge-Intensive Compressed LLM BenchmarK (LLM-KICK)**, a collection of carefully curated tasks to re-define the evaluation protocol for compressed LLMs, which have significant alignment with their dense counterparts, and perplexity fail to capture subtle change in their true capabilities.
> 2.  Some of our key observations include: **(a)** all pruning methods suffer significant performance degradation, sometimes at trivial sparsity ratios (e.g., 25-30%), and fail for N:M sparsity on knowledge-intensive tasks; **(b)** current quantization methods are
more successful than pruning; **(c)** pruned LLMs even at ≥ 50% sparsity are robust in-context retrieval and summarization systems.
> 3.  LLM-KICK is designed to holistically access compressed LLMs’ ability for language understanding, reasoning, generation, in-context retrieval, in-context summarization.
    
## Update
- [x] (02.06.2024) We released the code for LLM-KICK - Supports Vicuna 7B, 13B, 30B, 65B.
- [x] (02.08.2024) We provided support for [GPTQ-related](https://github.com/Ajay1994/llm_kick/tree/main/GPTQ_experiment) experiments.


## Installation 
--- 
Step 1: Clone this repository and navigate to `llm_kick` folder

```
git clone https://github.com/apple/llm_kick
cd llm_kick
```

Step 2: Create the conda environment:

```
conda create -n llm_kick python=3.9
conda activate llm_kick
```

Step 3: Install relevant packages:

```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install transformers==4.28.0 datasets==2.11.0 wandb sentencepiece
pip install accelerate==0.18.0
pip install shortuuid tqdm
```

## Usage

--- 
We provide a quick overview of the important arguments:  
- `--model`: The identifier for the Vicuna/LLaMa model on the Hugging Face model hub.
- `--cache_dir`: Directory for loading or storing LLM weights. The default is `llm_weights`.
- `--prune_method`: Pruning methods, namely [`magnitude`, `wanda`, `sparsegpt`].
- `--sparsity_ratio`: Denotes the percentage of weights to be pruned.
- `--sparsity_type`: Denotes the sparsity type: structured/unstructured [`unstructured`, `4:8`, `2:4`, `1:2`]
- `--num_examples`: Specifies the number of examples you want to conduct the evaluation.
- `--nsamples` : Specifies the number of calibration samples to use during pruning with Wanda and SparseGPT.
- `--ntrain`: Denotes the number of in-context training examples.
- `--include_context`:  Denotes if we want to use context knowledge in ICR-QA setting.


--- 
### Script example of Factoid QA experiments

```
CUDA_VISIBLE_DEVICES=0 python main_freebase.py \
    --model_type vicuna7b \
    --prune_method magnitude \
    --sparsity_ratio 0.20 \
    --sparsity_type unstructured \
    --num_examples 10

CUDA_VISIBLE_DEVICES=0 python main_freebase.py \
    --model_type vicuna7b \
    --prune_method wanda \
    --sparsity_ratio 0.20 \
    --sparsity_type 2:4 \
    --num_examples 10 
```


--- 
### Script example of MCR-QA experiments

```
CUDA_VISIBLE_DEVICES=0 python main_mmlu.py \
    --model_type vicuna7b \
    --prune_method wanda \
    --sparsity_ratio 0.20 \
    --sparsity_type unstructured 
```

--- 
### Script example of ICR-QA experiments

- Script for Closed-Book QA

```
CUDA_VISIBLE_DEVICES=0 python main_icrqa.py \
    --model_type vicuna7b \
    --prune_method magnitude \
    --sparsity_ratio 0.20 \
    --sparsity_type unstructured \
    --num_examples 10 
```

- Script for Open-Book QA

```
CUDA_VISIBLE_DEVICES=0 python main_icrqa.py \
    --model_type vicuna7b \
    --prune_method magnitude \
    --sparsity_ratio 0.20 \
    --sparsity_type unstructured \
    --include_context \
    --num_examples 10
```

--- 
### Script example of Text Summarization experiments

1. Add your `OPEN_AI KEY` at line [1](https://github.com/Ajay1994/llm_kick/blob/5b296458db1dc05d367ab9b8ab4d02ca1be5f3d7/in-context_summarization/gpt35_openai.py#L10) and [2](https://github.com/Ajay1994/llm_kick/blob/5b296458db1dc05d367ab9b8ab4d02ca1be5f3d7/in-context_summarization/gpt4_judge.py#L7) to get the GPT-3.5 reference summary and initiate GPT-4 Judge.
   
2. Script for generating the GPT 3.5 Summary
   ```
   python gpt35_openai.py --question ./json_utils/new_question.jsonl --output ./gpt35_answer.jsonl
   ```

3. Script for generating the Compressed Model Summary
   ```
   CUDA_VISIBLE_DEVICES=0 python compressed_model.py \
    --model_id lmsys/vicuna-7b-v1.3 \
    --prune_method magnitude \
    --sparsity_ratio 0.20 \
    --sparsity_type unstructured \
    --question-file ./table/new_question.jsonl \
    --nsamples 128
   ```
   
4. Script for running GPT-4 Judge
   ```
   CUDA_VISIBLE_DEVICES=0 python gpt4_judge.py --prune_method magnitude --sparsity_ratio 0.20
   ```

--- 
### Script example of Instruction Following experiments

1. Add your `OPEN_AI KEY` at line [1](https://github.com/Ajay1994/llm_kick/blob/d3cb0b2fe14a1b1da3312f1ca449e7689953763a/mt_conversation/gpt35_openai.py#L10) and [2](https://github.com/Ajay1994/llm_kick/blob/d3cb0b2fe14a1b1da3312f1ca449e7689953763a/mt_conversation/gpt4_judge.py#L7) to get the GPT-3.5 reference response to multi-turn conversation questions and initiate GPT-4 Judge respectively.

2. Script for generating the GPT 3.5 Responses.
   ```
   python gpt35_openai.py --question ./json_utils/question.jsonl --output ./answer.jsonl
   ```

3. Script for generating the Compressed Model Responses.
   ```
   CUDA_VISIBLE_DEVICES=0 python free_form_answer_prune.py \
    --model_id lmsys/vicuna-7b-v1.3 \
    --prune_method magnitude \
    --sparsity_ratio 0.20 \
    --sparsity_type unstructured \
    --question-file ./table/question.jsonl \
    --nsamples 128
   ```
   
4. Script for running GPT-4 Judge.
   ```
   CUDA_VISIBLE_DEVICES=0 python gpt4_judge.py --prune_method magnitude --sparsity_ratio 0.20
   ```

---
### Script example for Quantization Experiments

Script to generate the quantized models:

```
[1] Vicuna 7B:   python llama.py --wbits 8 --save <SAVE_LOCATION>/vicuna7b-8bit-128g.pt --model lmsys/vicuna-7b-v1.3
[2] Vicuna 13B:  python llama.py --wbits 8 --save <SAVE_LOCATION>/vicuna13b-8bit-128g.pt --model lmsys/vicuna-13b-v1.3
```

Script to test the quantized model on Factoid-QA:

```
CUDA_VISIBLE_DEVICES=0 python vicuna_inference_freebase.py --wbits 8 --load <SAVE_LOCATION>/vicuna13b-8bit-128g.pt --model lmsys/vicuna-13b-v1.3
```

### Acknowledgement
This repository is built upon the [Wanda](https://github.com/locuslab/wanda), [SparseGPT](https://github.com/IST-DASLab/sparsegpt), and [GPTQ](https://github.com/qwopqwop200/GPTQ-for-LLaMa) repositories.

**More details coming soon!**

## Citation
If you find this repository is helpful, please cite:

```
@article{jaiswal2023compressing,
  title={Compressing llms: The truth is rarely pure and never simple},
  author={Jaiswal, Ajay and Gan, Zhe and Du, Xianzhi and Zhang, Bowen and Wang, Zhangyang and Yang, Yinfei},
  journal={arXiv preprint arXiv:2310.01382},
  year={2023}
}
```
