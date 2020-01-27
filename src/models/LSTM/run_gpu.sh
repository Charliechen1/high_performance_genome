module load python cuda
module load pytorch/v1.1.0-gpu
srun -n 1 nohup python train.py -s 1 -e 30 -H 20 -b 64 -p 300 -c "../../../config/main.conf" -g -d > log_`date +"%m-%d-%Y"`.txt