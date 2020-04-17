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

if [[ ! -d "model" ]]
then
    mkdir model
fi

srun -n 1 nohup ~/.conda/envs/hpg_gcn_env/bin/python train.py
