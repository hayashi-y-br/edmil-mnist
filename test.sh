#!/bin/sh
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=24:00:00

eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate mil

python test.py -m dataset='012' model=EABMIL seed=0 path='/home/6/uf02776/edmil-mnist/MNIST-EABMIL.pth' model.activation=exp settings.loss=nll
python test.py -m dataset='012' model=EAdditiveMIL seed=0 path='/home/6/uf02776/edmil-mnist/MNIST-EAdditiveMIL.pth' model.activation=exp settings.loss=mse