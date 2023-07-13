CUDA_VISIBLE_DEVICES=1 \
python code2/main.py --mode attack --dataset_name agnews --attack_method textbugger --training_type sparse --attack_numbers 200 \
--sparse_mask_rate 0.05 \
--predict_ensemble 10 \
--batch_size 64 \
--predictor alpaca_agnews \





# CUDA_VISIBLE_DEVICES=1 \
# python code2/main.py --mode train --dataset_name sst2 --training_type sparse --sparse_mask_rate 0.3