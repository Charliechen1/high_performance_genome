#module load python cuda
#module load pytorch/v1.1.0-gpu
if [[ ! -d "log" ]]
then
    mkdir log
fi

srun -n 1 nohup ~/.conda/envs/hpg_gpu_env/bin/python train.py \
    --config "../../../config/main.conf" \
    > log/log_`date +"%H_%m_%d_%Y"`.txt&