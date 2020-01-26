if [[ $1 == "--gpu" ]] || [[ $1 == "-g" ]]
then
    # gpu version execution
    nohup /usr/common/software/pytorch/v1.2.0-gpu/bin/python train.py -s 1 -e 30 -H 20 -b 64 -p 300 -c -g "../../../config/main.conf" -d > log_`date +"%m-%d-%Y"`.txt &
elif [[ $1 == "--cpu" ]] || [[ $1 == "-c" ]]
then
    # cpu version execution
    nohup /usr/common/software/pytorch/v1.2.0/bin/python train.py -s 1 -e 30 -H 20 -b 64 -p 300 -c "../../../config/main.conf" -d > log_`date +"%m-%d-%Y"`.txt &
else
    echo "unknown command"
fi
