#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -t 480
#SBATCH -c 80
#SBATCH --gres=gpu:8

srun -n 1 `run script`
