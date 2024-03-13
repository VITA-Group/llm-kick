conda activate mci
CUDA_VISIBLE_DEVICES=3 python gpt4_judge.py --output-review-file ./compressed_answers/vicuna7b/answer_2bits_128_group
CUDA_VISIBLE_DEVICES=3 python gpt4_judge.py --output-review-file ./compressed_answers/vicuna7b/answer_4bits_128_group
CUDA_VISIBLE_DEVICES=3 python gpt4_judge.py --output-review-file ./compressed_answers/vicuna7b/answer_8bits_128_group

CUDA_VISIBLE_DEVICES=3 python gpt4_judge.py --output-review-file ./compressed_answers/vicuna7b/magnitude_answer_0.5_2:4
CUDA_VISIBLE_DEVICES=3 python gpt4_judge.py --output-review-file ./compressed_answers/vicuna7b/sparsegpt_answer_0.5_2:4
CUDA_VISIBLE_DEVICES=3 python gpt4_judge.py --output-review-file ./compressed_answers/vicuna7b/wanda_answer_0.5_2:4