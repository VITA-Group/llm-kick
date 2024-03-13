conda activate mci
CUDA_VISIBLE_DEVICES=0 python main_freebase.py \
    --model_type vicuna7b \
    --prune_method wanda \
    --sparsity_ratio 0.20 \
    --sparsity_type 2:4 \
    --num_examples 10 
