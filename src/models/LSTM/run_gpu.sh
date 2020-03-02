#module load python cuda
#module load pytorch/v1.1.0-gpu
if [[ ! -d "log" ]]
then
    mkdir log
fi

srun -n 1 nohup ~/.conda/envs/hpg_gpu_env/bin/python train.py \
    --sample_rate 1 \
    --emb_dim 80 \
    --epoches 20 \
    --hid_dim 200 \
    --num_of_folds 10 \
    --padding_size 2000 \
    --batch_size 400 \
    --config "../../../config/main.conf" \
    --gpu \
    --debug \
    >> log/log_`date +"%H_%m_%d_%Y"`.txt&