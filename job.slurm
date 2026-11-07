#!/bin/bash
#SBATCH --job-name=logsigma
#SBATCH --partition=gpu4090
#SBATCH --qos=4gpus
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --output=logsigma_marker.out
#SBATCH --error=logsigma_marker.err

module load cuda/12.1
conda activate dimabsa
python3 main_logsigma_marker.py