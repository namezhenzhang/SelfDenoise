mkdir output


export CUDA_VISIBLE_DEVICES=7
python code2/main.py \
--mode certify \
--dataset_name agnews \
--training_type sparse \
--sparse_mask_rate 0.1 \
--certify_numbers 100 \
--predict_ensemble 50 \
--ceritfy_ensemble 500 \
--saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
--denoise_method roberta \
--predictor alpaca_agnews \
--alpaca_batchsize 32 \
--world_size 1 \
--mask_word "<mask>" \

export CUDA_VISIBLE_DEVICES=7
python code2/main.py \
--mode certify \
--dataset_name agnews \
--training_type sparse \
--sparse_mask_rate 0.3 \
--certify_numbers 100 \
--predict_ensemble 50 \
--ceritfy_ensemble 500 \
--saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
--denoise_method roberta \
--predictor alpaca_agnews \
--alpaca_batchsize 32 \
--world_size 1 \
--mask_word "<mask>" \

export CUDA_VISIBLE_DEVICES=7
python code2/main.py \
--mode certify \
--dataset_name agnews \
--training_type sparse \
--sparse_mask_rate 0.5 \
--certify_numbers 100 \
--predict_ensemble 50 \
--ceritfy_ensemble 500 \
--saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
--denoise_method roberta \
--predictor alpaca_agnews \
--alpaca_batchsize 32 \
--world_size 1 \
--mask_word "<mask>" \

export CUDA_VISIBLE_DEVICES=7
python code2/main.py \
--mode certify \
--dataset_name agnews \
--training_type sparse \
--sparse_mask_rate 0.7 \
--certify_numbers 100 \
--predict_ensemble 50 \
--ceritfy_ensemble 500 \
--saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
--denoise_method roberta \
--predictor alpaca_agnews \
--alpaca_batchsize 32 \
--world_size 1 \
--mask_word "<mask>" \

export CUDA_VISIBLE_DEVICES=7
python code2/main.py \
--mode certify \
--dataset_name agnews \
--training_type sparse \
--sparse_mask_rate 0.9 \
--certify_numbers 100 \
--predict_ensemble 50 \
--ceritfy_ensemble 500 \
--saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
--denoise_method roberta \
--predictor alpaca_agnews \
--alpaca_batchsize 32 \
--world_size 1 \
--mask_word "<mask>" \
