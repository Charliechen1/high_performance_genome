#module load esslurm
#salloc -C gpu -N 1 -t 60 -c 10 --gres=gpu:1
CONF="hpg_gpu_env.yml"
if [ -e $CONF ]; then
    #echo "yml file found"
    ENV=$(head -n 1 $CONF | cut -f2 -d ' ')
    # Check if you are already in the environment
    if [[ $PATH != *$ENV* ]]; then
        # check whether the environment has been installed
        $EXIST=`conda info --envs | grep $ENV | wc -l`
        if [ $EXIST != "1" ]; then
            conda env create -f $CONF
        fi
    fi
    source activate $ENV
fi

module load python cuda
module load pytorch/v1.1.0-gpu
srun -n 1 nohup python train.py -s 1 -e 30 -H 20 -b 64 -p 300 -c "../../../config/main.conf" -g -d > log_`date +"%m-%d-%Y"`.txt