#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -t 240
#SBATCH -c 80
#SBATCH --gres=gpu:8

srun -n 1 nohup ~/.conda/envs/hpg_gpu_env/bin/python train.py \
    --config "../../../config/main.conf" \
    > log/log_`date +"%H_%m_%d_%Y"`.txt&