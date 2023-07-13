CUDA_VISIBLE_DEVICES=1 \
python code2/main.py --mode attack --dataset_name sst2 --attack_method bae --training_type sparse --attack_numbers 200 \
--sparse_mask_rate 0.05 \
--predict_ensemble 50 \
--batch_size 200 \
--predictor alpaca_sst2 \





# CUDA_VISIBLE_DEVICES=1 \
# python code2/main.py --mode train --dataset_name sst2 --training_type sparse --sparse_mask_rate 0.3