#!/bin/bash

for ((i=1; i<=100; i++))
do
    sbatch code2/script/slurm-sst2-0.3/$i.sh
done