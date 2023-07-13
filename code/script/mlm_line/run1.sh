# sleep 5h
mkdir output


export CUDA_VISIBLE_DEVICES=5
python code2/main.py \
--mode certify \
--dataset_name sst2 \
--training_type sparse \
--sparse_mask_rate 0.2 \
--certify_numbers 100 \
--predict_ensemble 50 \
--ceritfy_ensemble 500 \
--saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
--denoise_method roberta \
--predictor alpaca_sst2 \
--alpaca_batchsize 128 \
--world_size 1 \
--mask_word "<mask>" \


export CUDA_VISIBLE_DEVICES=5
python code2/main.py \
--mode certify \
--dataset_name sst2 \
--training_type sparse \
--sparse_mask_rate 0.4 \
--certify_numbers 100 \
--predict_ensemble 50 \
--ceritfy_ensemble 500 \
--saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
--denoise_method roberta \
--predictor alpaca_sst2 \
--alpaca_batchsize 128 \
--world_size 1 \
--mask_word "<mask>" \

export CUDA_VISIBLE_DEVICES=5
python code2/main.py \
--mode certify \
--dataset_name sst2 \
--training_type sparse \
--sparse_mask_rate 0.6 \
--certify_numbers 100 \
--predict_ensemble 50 \
--ceritfy_ensemble 500 \
--saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
--denoise_method roberta \
--predictor alpaca_sst2 \
--alpaca_batchsize 128 \
--world_size 1 \
--mask_word "<mask>" \


export CUDA_VISIBLE_DEVICES=5
python code2/main.py \
--mode certify \
--dataset_name sst2 \
--training_type sparse \
--sparse_mask_rate 0.8 \
--certify_numbers 100 \
--predict_ensemble 50 \
--ceritfy_ensemble 500 \
--saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
--denoise_method roberta \
--predictor alpaca_sst2 \
--alpaca_batchsize 128 \
--world_size 1 \
--mask_word "<mask>" \

# export CUDA_VISIBLE_DEVICES=5
# python code/main.py \
# --mode certify \
# --dataset_name sst2 \
# --training_type sparse \
# --sparse_mask_rate 0.2 \
# --certify_numbers 100 \
# --predict_ensemble 50 \
# --ceritfy_ensemble 500 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method alpaca \
# --predictor alpaca_sst2 \
# --alpaca_batchsize 3 \
# --world_size 1 \
# --mask_word "###" \

# export CUDA_VISIBLE_DEVICES=5
# python code/main.py \
# --mode certify \
# --dataset_name sst2 \
# --training_type sparse \
# --sparse_mask_rate 0.2 \
# --certify_numbers 100 \
# --predict_ensemble 50 \
# --ceritfy_ensemble 500 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method None \
# --predictor alpaca_sst2 \
# --alpaca_batchsize 3 \
# --world_size 1 \
# --mask_word "###" \

# export CUDA_VISIBLE_DEVICES=5
# python code/main.py \
# --mode certify \
# --dataset_name sst2 \
# --training_type sparse \
# --sparse_mask_rate 0.6 \
# --certify_numbers 100 \
# --predict_ensemble 50 \
# --ceritfy_ensemble 500 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method alpaca \
# --predictor alpaca_sst2 \
# --alpaca_batchsize 3 \
# --world_size 1 \
# --mask_word "###" \

# export CUDA_VISIBLE_DEVICES=5
# python code/main.py \
# --mode certify \
# --dataset_name sst2 \
# --training_type sparse \
# --sparse_mask_rate 0.6 \
# --certify_numbers 100 \
# --predict_ensemble 50 \
# --ceritfy_ensemble 500 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method None \
# --predictor alpaca_sst2 \
# --alpaca_batchsize 3 \
# --world_size 1 \
# --mask_word "<mask>" \

# export CUDA_VISIBLE_DEVICES=5
# python code/main.py \
# --mode certify \
# --dataset_name sst2 \
# --training_type sparse \
# --sparse_mask_rate 0.95 \
# --certify_numbers 100 \
# --predict_ensemble 50 \
# --ceritfy_ensemble 500 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method None \
# --predictor alpaca_sst2 \
# --alpaca_batchsize 3 \
# --world_size 1 \
# --mask_word "<mask>" \

# > output/cls_alpaca_sst2_None_maskrate_0.7_num_100_ensmb_10_ceritfy_ensemble_50_mw_###.txt 

# export CUDA_VISIBLE_DEVICES=0
# python /mnt/data/zhenzhang/dir1/ranmask/main.py \
# --mode certify \
# --dataset_name sst2 \
# --training_type sparse \
# --sparse_mask_rate 0.7 \
# --certify_numbers 100 \
# --predict_ensemble 10 \
# --ceritfy_ensemble 50 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method alpaca \
# --predictor alpaca_sst2 \
# --mask_word "###" \
# > output/cls_alpaca_sst2_alpaca_maskrate_0.7_num_100_ensmb_10_ceritfy_ensemble_50_mw_###.txt 


# export CUDA_VISIBLE_DEVICES=0
# python /mnt/data/zhenzhang/dir1/ranmask/main.py \
# --mode certify \
# --dataset_name sst2 \
# --training_type sparse \
# --sparse_mask_rate 0.8 \
# --certify_numbers 100 \
# --predict_ensemble 10 \
# --ceritfy_ensemble 50 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method None \
# --predictor alpaca_sst2 \
# --mask_word "###" \
# > output/cls_alpaca_sst2_None_maskrate_0.8_num_100_ensmb_10_ceritfy_ensemble_50_mw_###.txt 

# export CUDA_VISIBLE_DEVICES=0
# python /mnt/data/zhenzhang/dir1/ranmask/main.py \
# --mode certify \
# --dataset_name sst2 \
# --training_type sparse \
# --sparse_mask_rate 0.8 \
# --certify_numbers 100 \
# --predict_ensemble 10 \
# --ceritfy_ensemble 50 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method alpaca \
# --predictor alpaca_sst2 \
# --mask_word "###" \
# > output/cls_alpaca_sst2_alpaca_maskrate_0.8_num_100_ensmb_10_ceritfy_ensemble_50_mw_###.txt 

# export CUDA_VISIBLE_DEVICES=0
# python /mnt/data/zhenzhang/dir1/ranmask/main.py \
# --mode certify \
# --dataset_name sst2 \
# --training_type sparse \
# --sparse_mask_rate 0.9 \
# --certify_numbers 100 \
# --predict_ensemble 10 \
# --ceritfy_ensemble 50 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method None \
# --predictor alpaca_sst2 \
# --mask_word "###" \
# > output/cls_alpaca_sst2_None_maskrate_0.9_num_100_ensmb_10_ceritfy_ensemble_50_mw_###.txt 

# export CUDA_VISIBLE_DEVICES=0
# python /mnt/data/zhenzhang/dir1/ranmask/main.py \
# --mode certify \
# --dataset_name sst2 \
# --training_type sparse \
# --sparse_mask_rate 0.9 \
# --certify_numbers 100 \
# --predict_ensemble 10 \
# --ceritfy_ensemble 50 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method alpaca \
# --predictor alpaca_sst2 \
# --mask_word "###" \
# > output/cls_alpaca_sst2_alpaca_maskrate_0.9_num_100_ensmb_10_ceritfy_ensemble_50_mw_###.txt 
# export CUDA_VISIBLE_DEVICES=0
# python /mnt/data/zhenzhang/dir1/ranmask/main.py \
# --mode certify \
# --dataset_name agnews \
# --training_type sparse \
# --sparse_mask_rate 0.9 \
# --certify_numbers 100 \
# --predict_ensemble 40 \
# --ceritfy_ensemble 100 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method alpaca \
# --predictor alpaca_agnews \
# --mask_word "###" \
# > output/cls_alpaca_agnews_alpaca_maskrate_0.9_num_100_ensmb_40_ceritfy_ensemble_100_mw_###.txt 

# export CUDA_VISIBLE_DEVICES=0
# python /mnt/data/zhenzhang/dir1/ranmask/main.py \
# --mode certify \
# --dataset_name agnews \
# --training_type sparse \
# --sparse_mask_rate 0.8 \
# --certify_numbers 100 \
# --predict_ensemble 40 \
# --ceritfy_ensemble 100 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method alpaca \
# --predictor alpaca_agnews \
# --mask_word "###" \
# > output/cls_alpaca_agnews_alpaca_maskrate_0.8_num_100_ensmb_40_ceritfy_ensemble_100_mw_###.txt 

# export CUDA_VISIBLE_DEVICES=0
# python /mnt/data/zhenzhang/dir1/ranmask/main.py \
# --mode certify \
# --dataset_name agnews \
# --training_type sparse \
# --sparse_mask_rate 0.7 \
# --certify_numbers 100 \
# --predict_ensemble 40 \
# --ceritfy_ensemble 100 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method alpaca \
# --predictor alpaca_agnews \
# --mask_word "###" \
# > output/cls_alpaca_agnews_alpaca_maskrate_0.7_num_100_ensmb_40_ceritfy_ensemble_100_mw_###.txt 

# export CUDA_VISIBLE_DEVICES=0
# python /mnt/data/zhenzhang/dir1/ranmask/main.py \
# --mode certify \
# --dataset_name agnews \
# --training_type sparse \
# --sparse_mask_rate 0.95 \
# --certify_numbers 100 \
# --predict_ensemble 40 \
# --ceritfy_ensemble 100 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method alpaca \
# --predictor alpaca_agnews \
# --mask_word "###" \
# > output/cls_alpaca_agnews_alpaca_maskrate_0.95_num_100_ensmb_40_ceritfy_ensemble_100_mw_###.txt 

# export CUDA_VISIBLE_DEVICES=0
# python /mnt/data/zhenzhang/dir1/ranmask/main.py \
# --mode certify \
# --dataset_name agnews \
# --training_type sparse \
# --sparse_mask_rate 0.85 \
# --certify_numbers 100 \
# --predict_ensemble 40 \
# --ceritfy_ensemble 100 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method alpaca \
# --predictor alpaca_agnews \
# --mask_word "###" \
# > output/cls_alpaca_agnews_alpaca_maskrate_0.85_num_100_ensmb_40_ceritfy_ensemble_100_mw_###.txt 

# export CUDA_VISIBLE_DEVICES=0
# python /mnt/data/zhenzhang/dir1/ranmask/main.py \
# --mode certify \
# --dataset_name agnews \
# --training_type sparse \
# --sparse_mask_rate 0.75 \
# --certify_numbers 100 \
# --predict_ensemble 40 \
# --ceritfy_ensemble 100 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method alpaca \
# --predictor alpaca_agnews \
# --mask_word "###" \
# > output/cls_alpaca_agnews_alpaca_maskrate_0.75_num_100_ensmb_40_ceritfy_ensemble_100_mw_###.txt 


# export CUDA_VISIBLE_DEVICES=0
# python /mnt/data/zhenzhang/dir1/ranmask/main.py \
# --mode certify \
# --dataset_name agnews \
# --training_type sparse \
# --sparse_mask_rate 0.6 \
# --certify_numbers 100 \
# --predict_ensemble 40 \
# --ceritfy_ensemble 100 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method alpaca \
# --predictor alpaca_agnews \
# --mask_word "###" \
# > output/cls_alpaca_agnews_alpaca_maskrate_0.6_num_100_ensmb_40_ceritfy_ensemble_100_mw_###.txt 



# export CUDA_VISIBLE_DEVICES=0
# python /mnt/data/zhenzhang/dir1/ranmask/main.py \
# --mode certify \
# --dataset_name sst2 \
# --training_type sparse \
# --sparse_mask_rate 0.3 \
# --certify_numbers 100 \
# --predict_ensemble 10 \
# --ceritfy_ensemble 50 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method alpaca \
# --predictor alpaca_sst2 \
# --mask_word "###" \
# > output/cls_alpaca_sst2_alpaca_maskrate_0.3_num_100_ensmb_10_ceritfy_ensemble_50_mw_###.txt 

# export CUDA_VISIBLE_DEVICES=0
# python /mnt/data/zhenzhang/dir1/ranmask/main.py \
# --mode certify \
# --dataset_name sst2 \
# --training_type sparse \
# --sparse_mask_rate 0.4 \
# --certify_numbers 100 \
# --predict_ensemble 10 \
# --ceritfy_ensemble 50 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method alpaca \
# --predictor alpaca_sst2 \
# --mask_word "###" \
# > output/cls_alpaca_sst2_alpaca_maskrate_0.4_num_100_ensmb_10_ceritfy_ensemble_50_mw_###.txt 

# export CUDA_VISIBLE_DEVICES=0
# python /mnt/data/zhenzhang/dir1/ranmask/main.py \
# --mode certify \
# --dataset_name sst2 \
# --training_type sparse \
# --sparse_mask_rate 0.5 \
# --certify_numbers 100 \
# --predict_ensemble 10 \
# --ceritfy_ensemble 50 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method alpaca \
# --predictor alpaca_sst2 \
# --mask_word "###" \
# > output/cls_alpaca_sst2_alpaca_maskrate_0.5_num_100_ensmb_10_ceritfy_ensemble_50_mw_###.txt 


# export CUDA_VISIBLE_DEVICES=0
# python /mnt/data/zhenzhang/dir1/ranmask/main.py \
# --mode certify \
# --dataset_name sst2 \
# --training_type sparse \
# --sparse_mask_rate 0.6 \
# --certify_numbers 100 \
# --predict_ensemble 10 \
# --ceritfy_ensemble 50 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method alpaca \
# --predictor alpaca_sst2 \
# --mask_word "###" \
# > output/cls_alpaca_sst2_alpaca_maskrate_0.6_num_100_ensmb_10_ceritfy_ensemble_50_mw_###.txt 
# export CUDA_VISIBLE_DEVICES=0
# python /mnt/data/zhenzhang/dir1/ranmask/main.py \
# --mode certify \
# --dataset_name sst2 \
# --training_type sparse \
# --sparse_mask_rate 0.1 \
# --certify_numbers 100 \
# --predict_ensemble 10 \
# --ceritfy_ensemble 500 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method None \
# --predictor alpaca_sst2 \
# --mask_word "###" \
# > output/cls_alpaca_sst2_None_maskrate_0.1_num_100_ensmb_10_ceritfy_ensemble_500_mw_###.txt 

# export CUDA_VISIBLE_DEVICES=0
# python /mnt/data/zhenzhang/dir1/ranmask/main.py \
# --mode certify \
# --dataset_name sst2 \
# --training_type sparse \
# --sparse_mask_rate 0.1 \
# --certify_numbers 100 \
# --predict_ensemble 10 \
# --ceritfy_ensemble 500 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method alpaca \
# --predictor alpaca_sst2 \
# --mask_word "###" \
# > output/cls_alpaca_sst2_alpaca_maskrate_0.1_num_100_ensmb_10_ceritfy_ensemble_500_mw_###.txt 

# export CUDA_VISIBLE_DEVICES=0
# python /mnt/data/zhenzhang/dir1/ranmask/main.py \
# --mode certify \
# --dataset_name sst2 \
# --training_type sparse \
# --sparse_mask_rate 0.4 \
# --certify_numbers 100 \
# --predict_ensemble 10 \
# --ceritfy_ensemble 500 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method None \
# --predictor alpaca_sst2 \
# --mask_word "###" \
# > output/cls_alpaca_sst2_None_maskrate_0.4_num_100_ensmb_10_ceritfy_ensemble_500_mw_###.txt 

# export CUDA_VISIBLE_DEVICES=0
# python /mnt/data/zhenzhang/dir1/ranmask/main.py \
# --mode certify \
# --dataset_name sst2 \
# --training_type sparse \
# --sparse_mask_rate 0.4 \
# --certify_numbers 100 \
# --predict_ensemble 10 \
# --ceritfy_ensemble 500 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method alpaca \
# --predictor alpaca_sst2 \
# --mask_word "###" \
# > output/cls_alpaca_sst2_alpaca_maskrate_0.4_num_100_ensmb_10_ceritfy_ensemble_500_mw_###.txt 

# export CUDA_VISIBLE_DEVICES=0
# python /mnt/data/zhenzhang/dir1/ranmask/main.py \
# --mode certify \
# --dataset_name sst2 \
# --training_type sparse \
# --sparse_mask_rate 0.7 \
# --certify_numbers 100 \
# --predict_ensemble 10 \
# --ceritfy_ensemble 500 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method None \
# --predictor alpaca_sst2 \
# --mask_word "###" \
# > output/cls_alpaca_sst2_None_maskrate_0.7_num_100_ensmb_10_ceritfy_ensemble_500_mw_###.txt 

# export CUDA_VISIBLE_DEVICES=0
# python /mnt/data/zhenzhang/dir1/ranmask/main.py \
# --mode certify \
# --dataset_name sst2 \
# --training_type sparse \
# --sparse_mask_rate 0.7 \
# --certify_numbers 100 \
# --predict_ensemble 10 \
# --ceritfy_ensemble 500 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method alpaca \
# --predictor alpaca_sst2 \
# --mask_word "###" \
# > output/cls_alpaca_sst2_alpaca_maskrate_0.7_num_100_ensmb_10_ceritfy_ensemble_500_mw_###.txt 

# export CUDA_VISIBLE_DEVICES=0
# python /mnt/data/zhenzhang/dir1/ranmask/main.py \
# --mode certify \
# --dataset_name sst2 \
# --training_type sparse \
# --sparse_mask_rate 0.8 \
# --certify_numbers 100 \
# --predict_ensemble 10 \
# --ceritfy_ensemble 500 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method None \
# --predictor alpaca_sst2 \
# --mask_word "###" \
# > output/cls_alpaca_sst2_None_maskrate_0.8_num_100_ensmb_10_ceritfy_ensemble_500_mw_###.txt 

# export CUDA_VISIBLE_DEVICES=0
# python /mnt/data/zhenzhang/dir1/ranmask/main.py \
# --mode certify \
# --dataset_name sst2 \
# --training_type sparse \
# --sparse_mask_rate 0.8 \
# --certify_numbers 100 \
# --predict_ensemble 10 \
# --ceritfy_ensemble 500 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method alpaca \
# --predictor alpaca_sst2 \
# --mask_word "###" \
# > output/cls_alpaca_sst2_alpaca_maskrate_0.8_num_100_ensmb_10_ceritfy_ensemble_500_mw_###.txt 

# export CUDA_VISIBLE_DEVICES=5
# python /mnt/data/zhenzhang/dir1/ranmask/main.py \
# --mode certify \
# --dataset_name sst2 \
# --training_type sparse \
# --sparse_mask_rate 0.2 \
# --certify_numbers 100 \
# --predict_ensemble 10 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method chatgpt_batch \
# > output/sst2_chatgpt_maskrate_0.2_num_100_ensmb_10.txt 



# export CUDA_VISIBLE_DEVICES=5
# python /mnt/data/zhenzhang/dir1/ranmask/main.py \
# --mode certify \
# --dataset_name sst2 \
# --training_type sparse \
# --sparse_mask_rate 0.2 \
# --certify_numbers 100 \
# --predict_ensemble 10 \
# --saving_dir /mnt/data/zhenzhang/dir1/ranmask/save_fake_models \
# --denoise_method None \
# > output/sst2_None_maskrate_0.2_num_100_ensmb_10.txt 


# export CUDA_VISIBLE_DEVICES=5
# python main.py \
# --mode certify \
# --dataset_name agnews \
# --training_type sparse \
# --batch_size 32 \
# --sparse_mask_rate 0.75 \
# --certify_numbers 200 \
# --predict_ensemble 50 > output_0_75.txt &


# export CUDA_VISIBLE_DEVICES=5
# python main.py \
# --mode certify \
# --dataset_name agnews \
# --training_type sparse \
# --batch_size 32 \
# --sparse_mask_rate 0.85 \
# --certify_numbers 200 \
# --predict_ensemble 50 > output_0_85.txt &

# export CUDA_VISIBLE_DEVICES=5
# python main.py \
# --mode certify \
# --dataset_name agnews \
# --training_type sparse \
# --batch_size 32 \
# --sparse_mask_rate 0.95 \
# --certify_numbers 200 \
# --predict_ensemble 50 > output_0_95.txt &

# wait

# export CUDA_VISIBLE_DEVICES=5
# python main.py \
# --mode certify \
# --dataset_name agnews \
# --training_type sparse \
# --batch_size 32 \
# --sparse_mask_rate 0.4 \
# --certify_numbers 200 \
# --predict_ensemble 50 > output_0_4.txt &

# export CUDA_VISIBLE_DEVICES=5
# python main.py \
# --mode certify \
# --dataset_name agnews \
# --training_type sparse \
# --batch_size 32 \
# --sparse_mask_rate 0.5 \
# --certify_numbers 200 \
# --predict_ensemble 50 > output_0_5.txt &

# export CUDA_VISIBLE_DEVICES=5
# python main.py \
# --mode certify \
# --dataset_name agnews \
# --training_type sparse \
# --batch_size 32 \
# --sparse_mask_rate 0.6 \
# --certify_numbers 200 \
# --predict_ensemble 50 > output_0_6.txt &

# wait

# export CUDA_VISIBLE_DEVICES=5
# python main.py \
# --mode certify \
# --dataset_name agnews \
# --training_type sparse \
# --batch_size 32 \
# --sparse_mask_rate 0.7 \
# --certify_numbers 200 \
# --predict_ensemble 50 > output_0_7.txt &

# export CUDA_VISIBLE_DEVICES=5
# python main.py \
# --mode certify \
# --dataset_name agnews \
# --training_type sparse \
# --batch_size 32 \
# --sparse_mask_rate 0.8 \
# --certify_numbers 200 \
# --predict_ensemble 50 > output_0_8.txt &

# export CUDA_VISIBLE_DEVICES=5
# python main.py \
# --mode certify \
# --dataset_name agnews \
# --training_type sparse \
# --batch_size 32 \
# --sparse_mask_rate 0.9 \
# --certify_numbers 200 \
# --predict_ensemble 50 > output_0_9.txt &

# wait 

# export CUDA_VISIBLE_DEVICES=5
# python main.py \
# --mode certify \
# --dataset_name agnews \
# --training_type sparse \
# --batch_size 32 \
# --sparse_mask_rate 0.65 \    `
# --certify_numbers 200 \
# --predict_ensemble 50 > output_0_65.txt &

# export CUDA_VISIBLE_DEVICES=5
# python main.py \
# --mode certify \
# --dataset_name agnews \
# --training_type sparse \
# --batch_size 32 \
# --sparse_mask_rate 0.75 \    `
# --certify_numbers 200 \
# --predict_ensemble 50 > output_0_75.txt &

# export CUDA_VISIBLE_DEVICES=5
# python main.py \
# --mode certify \
# --dataset_name agnews \
# --training_type sparse \
# --batch_size 32 \
# --sparse_mask_rate 0.85 \    `
# --certify_numbers 200 \
# --predict_ensemble 50 > output_0_85.txt &


# export CUDA_VISIBLE_DEVICES=5
# python main.py \
# --mode certify \
# --dataset_name agnews \
# --training_type sparse \
# --batch_size 32 \
# --sparse_mask_rate 0.95 \    `
# --certify_numbers 200 \
# --predict_ensemble 50 > output_0_95.txt &

# wait

# export CUDA_VISIBLE_DEVICES=5
# python main.py \
# --mode train \
# --dataset_name agnews \
# --training_type sparse \
# --sparse_mask_rate 0.4 \
# --batch_size 32 \
# --seed 100 &

# export CUDA_VISIBLE_DEVICES=5
# python main.py \
# --mode train \
# --dataset_name agnews \
# --training_type sparse \
# --sparse_mask_rate 0.5 \
# --batch_size 32 \
# --seed 100 &

# export CUDA_VISIBLE_DEVICES=5
# python main.py \
# --mode train \
# --dataset_name agnews \
# --training_type sparse \
# --sparse_mask_rate 0.6 \
# --batch_size 32 \
# --seed 100 &


# wait


# export CUDA_VISIBLE_DEVICES=0
# python main.py \
# --mode train \
# --dataset_name agnews \
# --training_type sparse \
# --sparse_mask_rate 0.95 \
# --batch_size 32 \
# --seed 100 

# wait












# export CUDA_VISIBLE_DEVICES=7 
# python main.py \
# --mode certify \
# --dataset_name agnews \
# --training_type sparse \
# --sparse_mask_rate 0.9 \
# --certify_numbers 200 \
# --predict_ensemble 50

# export CUDA_VISIBLE_DEVICES=7 
# python main.py \
# --mode train \
# --dataset_name agnews \
# --training_type sparse \
# --sparse_mask_rate 0.4 \
# --seed 100

# export OPENAI_API_KEY=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
# python /data/private/zhangzhen/dir3/RanMASK/denoiser.py

# /data/private/zhangzhen/dir3/RanMASK/clash/clash-linux-amd64-v1.10.0 -f glados.yaml -d .