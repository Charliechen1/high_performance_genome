CONF="hpg_gpu_env.yml"

#Check that config file exists and grab environment name
if [ -e $CONF ]
then
    ENV=$(head -n 1 $CONF | cut -f2 -d ' ')
else
    echo "Environment config not found"
    exit 1
fi

#Check if environment is already installed and install if needed
EXIST=`conda info --envs | grep $ENV | wc -l`
if [ $EXIST == "1" ]
then
    echo "Environment already installed"
else
    echo "Installing environment"
    conda env create -f $CONF
fi

echo "Launching GPU interactive session"
module load esslurm
salloc -C gpu -N 1 -t 60 -c 10 --gres=gpu:1