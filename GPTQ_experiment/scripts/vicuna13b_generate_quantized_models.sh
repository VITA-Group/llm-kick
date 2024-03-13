conda activate mci
# python llama.py --wbits 2 --save /data/ajay_data/MCI/quantized_llama/vicuna13b-2bit-128g.pt --model lmsys/vicuna-13b-v1.3
python llama.py --wbits 3 --save /data/ajay_data/MCI/quantized_llama/vicuna13b-3bit-128g.pt --model lmsys/vicuna-13b-v1.3
# python llama.py --wbits 4 --save /data/ajay_data/MCI/quantized_llama/vicuna13b-4bit-128g.pt --model lmsys/vicuna-13b-v1.3
python llama.py --wbits 6 --save /data/ajay_data/MCI/quantized_llama/vicuna13b-6bit-128g.pt --model lmsys/vicuna-13b-v1.3
# python llama.py --wbits 8 --save /data/ajay_data/MCI/quantized_llama/vicuna13b-8bit-128g.pt --model lmsys/vicuna-13b-v1.3
python llama.py --wbits 10 --save /data/ajay_data/MCI/quantized_llama/vicuna13b-10bit-128g.pt --model lmsys/vicuna-13b-v1.3
python llama.py --wbits 12 --save /data/ajay_data/MCI/quantized_llama/vicuna13b-12bit-128g.pt --model lmsys/vicuna-13b-v1.3
python llama.py --wbits 14 --save /data/ajay_data/MCI/quantized_llama/vicuna13b-14bit-128g.pt --model lmsys/vicuna-13b-v1.3
python llama.py --wbits 16 --save /data/ajay_data/MCI/quantized_llama/vicuna13b-16bit-128g.pt --model lmsys/vicuna-13b-v1.3