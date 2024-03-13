conda activate mci
# CUDA_VISIBLE_DEVICES=0 python vicuna_inference_trivia.py --wbits 2 --load /data/ajay_data/MCI/quantized_llama/vicuna7b-2bit-128g.pt --model lmsys/vicuna-7b-v1.3 --include_context
# CUDA_VISIBLE_DEVICES=0 python vicuna_inference_trivia.py --wbits 4 --load /data/ajay_data/MCI/quantized_llama/vicuna7b-4bit-128g.pt --model lmsys/vicuna-7b-v1.3 --include_context
# CUDA_VISIBLE_DEVICES=0 python vicuna_inference_trivia.py --wbits 8 --load /data/ajay_data/MCI/quantized_llama/vicuna7b-8bit-128g.pt --model lmsys/vicuna-7b-v1.3 --include_context
# CUDA_VISIBLE_DEVICES=0 python vicuna_inference_trivia.py --wbits 2 --load /data/ajay_data/MCI/quantized_llama/vicuna7b-2bit-128g.pt --model lmsys/vicuna-7b-v1.3
# CUDA_VISIBLE_DEVICES=0 python vicuna_inference_trivia.py --wbits 4 --load /data/ajay_data/MCI/quantized_llama/vicuna7b-4bit-128g.pt --model lmsys/vicuna-7b-v1.3
# CUDA_VISIBLE_DEVICES=0 python vicuna_inference_trivia.py --wbits 8 --load /data/ajay_data/MCI/quantized_llama/vicuna7b-8bit-128g.pt --model lmsys/vicuna-7b-v1.3

CUDA_VISIBLE_DEVICES=1 python vicuna_inference_trivia.py --wbits 8 --load /data/ajay_data/MCI/quantized_llama/vicuna13b-8bit-128g.pt --model lmsys/vicuna-13b-v1.3 --include_context
CUDA_VISIBLE_DEVICES=1 python vicuna_inference_trivia.py --wbits 2 --load /data/ajay_data/MCI/quantized_llama/vicuna13b-2bit-128g.pt --model lmsys/vicuna-13b-v1.3 --include_context
CUDA_VISIBLE_DEVICES=1 python vicuna_inference_trivia.py --wbits 4 --load /data/ajay_data/MCI/quantized_llama/vicuna13b-4bit-128g.pt --model lmsys/vicuna-13b-v1.3 --include_context
CUDA_VISIBLE_DEVICES=1 python vicuna_inference_trivia.py --wbits 8 --load /data/ajay_data/MCI/quantized_llama/vicuna13b-8bit-128g.pt --model lmsys/vicuna-13b-v1.3
CUDA_VISIBLE_DEVICES=1 python vicuna_inference_trivia.py --wbits 2 --load /data/ajay_data/MCI/quantized_llama/vicuna13b-2bit-128g.pt --model lmsys/vicuna-13b-v1.3
CUDA_VISIBLE_DEVICES=1 python vicuna_inference_trivia.py --wbits 4 --load /data/ajay_data/MCI/quantized_llama/vicuna13b-4bit-128g.pt --model lmsys/vicuna-13b-v1.3


