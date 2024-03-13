conda activate mci
CUDA_VISIBLE_DEVICES=0 python main_mmlu.py \
    --model_type vicuna7b \
    --prune_method wanda \
    --sparsity_ratio 0.20 \
    --sparsity_type unstructured 