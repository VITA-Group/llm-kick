conda activate mci
# python llama.py --wbits 2 --save /data/ajay_data/MCI/quantized_llama/vicuna7b-2bit-128g.pt
# CUDA_VISIBLE_DEVICES=1 python llama.py --wbits 3 --save /data/ajay_data/MCI/quantized_llama/vicuna7b-3bit-128g.pt
# CUDA_VISIBLE_DEVICES=1 python llama.py --wbits 4 --save /data/ajay_data/MCI/quantized_llama/vicuna7b-4bit-128g.pt
# CUDA_VISIBLE_DEVICES=1 python llama.py --wbits 6 --save /data/ajay_data/MCI/quantized_llama/vicuna7b-6bit-128g.pt
# # python llama.py --wbits 8 --save /data/ajay_data/MCI/quantized_llama/vicuna7b-8bit-128g.pt
# CUDA_VISIBLE_DEVICES=1 python llama.py --wbits 10 --save /data/ajay_data/MCI/quantized_llama/vicuna7b-10bit-128g.pt
# CUDA_VISIBLE_DEVICES=1 python llama.py --wbits 12 --save /data/ajay_data/MCI/quantized_llama/vicuna7b-12bit-128g.pt
# CUDA_VISIBLE_DEVICES=1 python llama.py --wbits 14 --save /data/ajay_data/MCI/quantized_llama/vicuna7b-14bit-128g.pt
# CUDA_VISIBLE_DEVICES=1 python llama.py --wbits 16 --save /data/ajay_data/MCI/quantized_llama/vicuna7b-16bit-128g.pt


python llama.py --wbits 2 --save /data/ajay_data/MCI/quantized_llama/llama7b-2bit-128g.pt --model decapoda-research/llama-7b-hf

python llama.py --wbits 4 --save /data/ajay_data/MCI/quantized_llama/llama7b-4bit-128g.pt --model decapoda-research/llama-7b-hf

python llama.py --wbits 8 --save /data/ajay_data/MCI/quantized_llama/llama7b-8bit-128g.pt --model decapoda-research/llama-7b-hf