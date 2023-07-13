mkdir output

# export CUDA_VISIBLE_DEVICES=0
# deepspeed code2/main.py \
# --mode certify \
# --dataset_name agnews \
# --training_type sparse \
# --sparse_mask_rate 0.1 \
# --certify_numbers 100 \
# --predict_ensemble 50 \
# --ceritfy_ensemble 500 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method alpaca \
# --predictor alpaca_agnews \
# --alpaca_batchsize 2 \
# --world_size 1 \
# --mask_word "###" \

export CUDA_VISIBLE_DEVICES=0
deepspeed code2/main.py \
--mode certify \
--dataset_name agnews \
--training_type sparse \
--sparse_mask_rate 0.1 \
--certify_numbers 100 \
--predict_ensemble 50 \
--ceritfy_ensemble 500 \
--saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
--denoise_method remove_mask \
--predictor alpaca_agnews \
--alpaca_batchsize 2 \
--world_size 1 \
--mask_word "<mask>" \

# export CUDA_VISIBLE_DEVICES=0
# deepspeed code2/main.py \
# --mode certify \
# --dataset_name agnews \
# --training_type sparse \
# --sparse_mask_rate 0.5 \
# --certify_numbers 100 \
# --predict_ensemble 50 \
# --ceritfy_ensemble 500 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method alpaca \
# --predictor alpaca_agnews \
# --alpaca_batchsize 2 \
# --world_size 1 \
# --mask_word "###" \

export CUDA_VISIBLE_DEVICES=0
deepspeed code2/main.py \
--mode certify \
--dataset_name agnews \
--training_type sparse \
--sparse_mask_rate 0.5 \
--certify_numbers 100 \
--predict_ensemble 50 \
--ceritfy_ensemble 500 \
--saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
--denoise_method remove_mask \
--predictor alpaca_agnews \
--alpaca_batchsize 2 \
--world_size 1 \
--mask_word "<mask>" \

# export CUDA_VISIBLE_DEVICES=0
# deepspeed code2/main.py \
# --mode certify \
# --dataset_name agnews \
# --training_type sparse \
# --sparse_mask_rate 0.9 \
# --certify_numbers 100 \
# --predict_ensemble 50 \
# --ceritfy_ensemble 500 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method alpaca \
# --predictor alpaca_agnews \
# --alpaca_batchsize 2 \
# --world_size 1 \
# --mask_word "###" \

export CUDA_VISIBLE_DEVICES=0
deepspeed code2/main.py \
--mode certify \
--dataset_name agnews \
--training_type sparse \
--sparse_mask_rate 0.9 \
--certify_numbers 100 \
--predict_ensemble 50 \
--ceritfy_ensemble 500 \
--saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
--denoise_method remove_mask \
--predictor alpaca_agnews \
--alpaca_batchsize 2 \
--world_size 1 \
--mask_word "<mask>" \