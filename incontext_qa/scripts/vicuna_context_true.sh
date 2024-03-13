conda activate mci
CUDA_VISIBLE_DEVICES=0 python main_icrqa.py \
    --model_type vicuna7b \
    --prune_method magnitude \
    --sparsity_ratio 0.20 \
    --sparsity_type unstructured \
    --include_context \
    --num_examples 10 