CUDA_VISIBLE_DEVICES=6 \
python code2/main.py --mode attack --dataset_name sst2 --attack_method deepwordbug --training_type None --attack_numbers 200 \
--sparse_mask_rate 0.05 \
--predict_ensemble 50 \
--batch_size 256 \
--predictor alpaca_sst2 \
--denoise_method None \

