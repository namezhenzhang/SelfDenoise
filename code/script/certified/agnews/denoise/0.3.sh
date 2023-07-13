#!/bin/bash
deepspeed code/main.py \
--mode certify \
--dataset_name agnews \
--training_type sparse \
--sparse_mask_rate 0.3 \
--certify_numbers 100 \
--predict_ensemble 50 \
--ceritfy_ensemble 500 \
--denoise_method alpaca \
--predictor alpaca_agnews \
--alpaca_batchsize 3 \
--world_size 1 \
--mask_word "<mask>" \
