CUDA_VISIBLE_DEVICES=6 \
deepspeed code2/main.py --mode attack --dataset_name agnews --attack_method bae --training_type sparse --attack_numbers 200 \
--sparse_mask_rate 0.05 \
--predict_ensemble 5 \
--batch_size 4 \
--predictor alpaca_agnews \
--denoise_method alpaca \
--mask_word "###" \



# CUDA_VISIBLE_DEVICES=1 \
# python code2/main.py --mode train --dataset_name sst2 --training_type sparse --sparse_mask_rate 0.3