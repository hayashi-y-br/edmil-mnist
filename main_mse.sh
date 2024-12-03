#!/bin/sh
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=24:00:00
#$ -t 1-5

eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate mil

python main.py -m dataset='012' seed=$(($SGE_TASK_ID-1)) model.activation=relu,exp settings.loss=mse settings.annealing_step=10,20,50,100 settings.lr=0.001,0.0005,0.0001,0.00005,0.00001 settings.reg=0