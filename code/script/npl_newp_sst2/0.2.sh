#!/bin/bash
#SBATCH --job-name=de_1
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/1.out

# Your script goes here
PYTHON_VIRTUAL_ENVIRONMENT=ran
CONDA_ROOT=/gpfs/u/home/DFLM/DFLMshcg/scratch/miniconda3-x86
## Activate the virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited
export PYTHONHASHSEED=0
#YOURCOMMAND
deepspeed code2/main.py \
--mode certify \
--dataset_name agnews \
--training_type sparse \
--sparse_mask_rate 0.2 \
--certify_numbers 100 \
--predict_ensemble 50 \
--ceritfy_ensemble 500 \
--denoise_method alpaca \
--predictor alpaca_agnews \
--alpaca_batchsize 1 \
--world_size 1 \
--mask_word "<mask>" \