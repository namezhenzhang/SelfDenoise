CUDA_VISIBLE_DEVICES=6 \
deepspeed code2/main.py --mode attack --dataset_name sst2 --attack_method textbugger --training_type None --attack_numbers 200 \
--sparse_mask_rate 0.05 \
--predict_ensemble 1 \
--batch_size 256 \
--predictor alpaca_sst2 \
--denoise_method None \




# CUDA_VISIBLE_DEVICES=1 \
# python code2/main.py --mode train --dataset_name sst2 --training_type sparse --sparse_mask_rate 0.3