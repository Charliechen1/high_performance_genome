#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -t 240
#SBATCH -c 80
#SBATCH --gres=gpu:8

srun -n 1 nohup ~/.conda/envs/hpg_gpu_env/bin/python train.py \
    --sample_rate 1 \
    --emb_dim 80 \
    --epoches 5 \
    --hid_dim 200 \
    --num_of_folds 10 \
    --padding_size 2000 \
    --batch_size 256 \
    --config "../../../config/main.conf" \
    --gpu \
    --debug \
    >> log/log_`date +"%H_%m_%d_%Y"`.txt &