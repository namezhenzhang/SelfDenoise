CUDA_VISIBLE_DEVICES=5 \
deepspeed code2/main.py --mode attack --dataset_name agnews --attack_method bae --training_type safer --attack_numbers 200 \
--sparse_mask_rate 0.05 \
--predict_ensemble 5 \
--batch_size 64 \
--predictor alpaca_agnews \
--denoise_method None \




# CUDA_VISIBLE_DEVICES=1 \
# python code2/main.py --mode train --dataset_name sst2 --training_type sparse --sparse_mask_rate 0.3