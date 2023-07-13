#!/bin/bash
#SBATCH --job-name=de_29
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/29.out

# Your script goes here
PYTHON_VIRTUAL_ENVIRONMENT=ran
CONDA_ROOT=/gpfs/u/home/DFLM/DFLMshcg/scratch/miniconda3-x86
## Activate the virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited
export PYTHONHASHSEED=0
#YOURCOMMAND
deepspeed  code2/main.py --mode attack --dataset_name sst2 --attack_method textfooler --training_type sparse --attack_numbers 2 --sparse_mask_rate 0.3 \
--predict_ensemble 50 \
--batch_size 1 \
--predictor alpaca_sst2 \
--denoise_method alpaca \
--seed 29 \