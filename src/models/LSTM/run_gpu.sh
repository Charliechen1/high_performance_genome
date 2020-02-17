#module load python cuda
#module load pytorch/v1.1.0-gpu
if [[ ! -d "log" ]]
then
    mkdir log
fi

nohup srun -n 1 ~/.conda/envs/hpg_gpu_env/bin/python train.py -s 1 -e 80 -E 8 -H 80 -f 10 -p 500 -P 200 -c "../../../config/main.conf" -g -d >> log/log_`date +"%m-%d-%Y"`.txt &