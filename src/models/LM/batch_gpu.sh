#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -t 480
#SBATCH -c 80
#SBATCH --gres=gpu:8

if [[ ! -d "log" ]]
then
    mkdir log
fi

srun -n 1 nohup ~/.conda/envs/hpg_gpu_env/bin/python train.py \
    --config "../../../config/main.conf" \
    > log/log_`date +"%H_%m_%d_%Y"`.txt&
