#!/bin/bash
#SBATCH --job-name=logsigma
#SBATCH --partition=gpu4090
#SBATCH --qos=4gpus
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --output=logsigma.out
#SBATCH --error=logsigma.err

source ~/.bashrc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dimabsa
export PYTHONNOUSERSITE=1

which python
python --version
python -u main_logsigma.py