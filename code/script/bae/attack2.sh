CUDA_VISIBLE_DEVICES=1 \
deepspeed code2/main.py --mode attack --dataset_name sst2 --attack_method bae --training_type sparse --attack_numbers 200 \
--sparse_mask_rate 0.2 \
--predict_ensemble 10 \
--batch_size 128 \
--predictor alpaca_sst2 \
--denoise_method roberta \
--mask_word "<mask>" \




# CUDA_VISIBLE_DEVICES=1 \
# python code2/main.py --mode train --dataset_name sst2 --training_type sparse --sparse_mask_rate 0.3