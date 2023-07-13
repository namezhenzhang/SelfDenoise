CUDA_VISIBLE_DEVICES=2 \
deepspeed code2/main.py --mode attack --dataset_name sst2 --attack_method bae --training_type safer --attack_numbers 200 \
--sparse_mask_rate 0.20 \
--predict_ensemble 50 \
--batch_size 256 \
--predictor alpaca_sst2 \
--denoise_method None \




# CUDA_VISIBLE_DEVICES=1 \
# python code2/main.py --mode train --dataset_name sst2 --training_type sparse --sparse_mask_rate 0.3