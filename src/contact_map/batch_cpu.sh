#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -t 04:00:00

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


#run the application:
srun -n 1 -c 64 --cpu_bind=cores nohup ~/.conda/envs/contact_map_env/bin/python contactmap.py > log_`date +"%m-%d-%Y"`.txt&