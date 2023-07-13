mkdir output

export CUDA_VISIBLE_DEVICES=6
deepspeed code2/main.py \
--mode certify \
--dataset_name sst2 \
--training_type sparse \
--sparse_mask_rate 0.1 \
--certify_numbers 100 \
--predict_ensemble 50 \
--ceritfy_ensemble 500 \
--saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
--denoise_method remove_mask \
--predictor alpaca_sst2 \
--alpaca_batchsize 32 \
--world_size 1 \
--mask_word "<mask>" \

export CUDA_VISIBLE_DEVICES=1
deepspeed code2/main.py \
--mode certify \
--dataset_name sst2 \
--training_type sparse \
--sparse_mask_rate 0.2 \
--certify_numbers 100 \
--predict_ensemble 50 \
--ceritfy_ensemble 500 \
--saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
--denoise_method remove_mask \
--predictor alpaca_sst2 \
--alpaca_batchsize 32 \
--world_size 1 \
--mask_word "<mask>" \

export CUDA_VISIBLE_DEVICES=1
deepspeed code2/main.py \
--mode certify \
--dataset_name sst2 \
--training_type sparse \
--sparse_mask_rate 0.3 \
--certify_numbers 100 \
--predict_ensemble 50 \
--ceritfy_ensemble 500 \
--saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
--denoise_method remove_mask \
--predictor alpaca_sst2 \
--alpaca_batchsize 32 \
--world_size 1 \
--mask_word "<mask>" \

export CUDA_VISIBLE_DEVICES=1
deepspeed code2/main.py \
--mode certify \
--dataset_name sst2 \
--training_type sparse \
--sparse_mask_rate 0.4 \
--certify_numbers 100 \
--predict_ensemble 50 \
--ceritfy_ensemble 500 \
--saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
--denoise_method remove_mask \
--predictor alpaca_sst2 \
--alpaca_batchsize 32 \
--world_size 1 \
--mask_word "<mask>" \

export CUDA_VISIBLE_DEVICES=1
deepspeed code2/main.py \
--mode certify \
--dataset_name sst2 \
--training_type sparse \
--sparse_mask_rate 0.5 \
--certify_numbers 100 \
--predict_ensemble 50 \
--ceritfy_ensemble 500 \
--saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
--denoise_method remove_mask \
--predictor alpaca_sst2 \
--alpaca_batchsize 32 \
--world_size 1 \
--mask_word "<mask>" \

export CUDA_VISIBLE_DEVICES=1
deepspeed code2/main.py \
--mode certify \
--dataset_name sst2 \
--training_type sparse \
--sparse_mask_rate 0.6 \
--certify_numbers 100 \
--predict_ensemble 50 \
--ceritfy_ensemble 500 \
--saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
--denoise_method remove_mask \
--predictor alpaca_sst2 \
--alpaca_batchsize 32 \
--world_size 1 \
--mask_word "<mask>" \

export CUDA_VISIBLE_DEVICES=1
deepspeed code2/main.py \
--mode certify \
--dataset_name sst2 \
--training_type sparse \
--sparse_mask_rate 0.7 \
--certify_numbers 100 \
--predict_ensemble 50 \
--ceritfy_ensemble 500 \
--saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
--denoise_method remove_mask \
--predictor alpaca_sst2 \
--alpaca_batchsize 32 \
--world_size 1 \
--mask_word "<mask>" \

export CUDA_VISIBLE_DEVICES=1
deepspeed code2/main.py \
--mode certify \
--dataset_name sst2 \
--training_type sparse \
--sparse_mask_rate 0.8 \
--certify_numbers 100 \
--predict_ensemble 50 \
--ceritfy_ensemble 500 \
--saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
--denoise_method remove_mask \
--predictor alpaca_sst2 \
--alpaca_batchsize 32 \
--world_size 1 \
--mask_word "<mask>" \

export CUDA_VISIBLE_DEVICES=1
deepspeed code2/main.py \
--mode certify \
--dataset_name sst2 \
--training_type sparse \
--sparse_mask_rate 0.9 \
--certify_numbers 100 \
--predict_ensemble 50 \
--ceritfy_ensemble 500 \
--saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
--denoise_method remove_mask \
--predictor alpaca_sst2 \
--alpaca_batchsize 32 \
--world_size 1 \
--mask_word "<mask>" \