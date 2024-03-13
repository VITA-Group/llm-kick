conda activate mci

# CUDA_VISIBLE_DEVICES=7 python vicuna_inference_summarization.py --wbits 8 --load /data/ajay_data/MCI/quantized_llama/vicuna7b-8bit-128g.pt --model lmsys/vicuna-7b-v1.3 
# CUDA_VISIBLE_DEVICES=7 python vicuna_inference_summarization.py --wbits 2 --load /data/ajay_data/MCI/quantized_llama/vicuna7b-2bit-128g.pt --model lmsys/vicuna-7b-v1.3 
# CUDA_VISIBLE_DEVICES=7 python vicuna_inference_summarization.py --wbits 4 --load /data/ajay_data/MCI/quantized_llama/vicuna7b-4bit-128g.pt --model lmsys/vicuna-7b-v1.3 

CUDA_VISIBLE_DEVICES=6 python vicuna_inference_mtbench.py --wbits 8 --load /data/ajay_data/MCI/quantized_llama/vicuna7b-8bit-128g.pt --model lmsys/vicuna-7b-v1.3 
CUDA_VISIBLE_DEVICES=6 python vicuna_inference_mtbench.py --wbits 2 --load /data/ajay_data/MCI/quantized_llama/vicuna7b-2bit-128g.pt --model lmsys/vicuna-7b-v1.3 
CUDA_VISIBLE_DEVICES=6 python vicuna_inference_mtbench.py --wbits 4 --load /data/ajay_data/MCI/quantized_llama/vicuna7b-4bit-128g.pt --model lmsys/vicuna-7b-v1.3 

# CUDA_VISIBLE_DEVICES=7 python vicuna_inference_freebase.py --wbits 8 --load /data/ajay_data/MCI/quantized_llama/llama7b-8bit-128g.pt --model decapoda-research/llama-7b-hf
# CUDA_VISIBLE_DEVICES=7 python vicuna_inference_freebase.py --wbits 2 --load /data/ajay_data/MCI/quantized_llama/llama7b-2bit-128g.pt --model decapoda-research/llama-7b-hf
# CUDA_VISIBLE_DEVICES=7 python vicuna_inference_freebase.py --wbits 4 --load /data/ajay_data/MCI/quantized_llama/llama7b-4bit-128g.pt --model decapoda-research/llama-7b-hf

