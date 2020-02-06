module load esslurm
salloc -C gpu -N 1 -t 60 -c 10 --gres=gpu:1