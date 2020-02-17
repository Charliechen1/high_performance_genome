#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -t 480
#SBATCH -c 10
#SBATCH --gres=gpu:1

srun -n 1 nohup ~/.conda/envs/hpg_gpu_env/bin/python train.py -s 1 -e 30 -H 20 -b 64 -p 300 -c "../../../config/main.conf" -g -d > log_`date +"%m-%d-%Y"`.txt