CUDA_VISIBLE_DEVICES=7 \
deepspeed code2/main.py --mode attack --dataset_name sst2 --attack_method textbugger --training_type sparse --attack_numbers 200 \
--sparse_mask_rate 0.05 \
--predict_ensemble 10 \
--batch_size 16 \
--predictor alpaca_sst2 \
--denoise_method alpaca \
--mask_word "###" \




# CUDA_VISIBLE_DEVICES=1 \
# python code2/main.py --mode train --dataset_name sst2 --training_type sparse --sparse_mask_rate 0.3