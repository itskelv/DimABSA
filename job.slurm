#!/bin/bash
#SBATCH --job-name=notebook_job
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

source myenv/bin/activate
python SentimentAnalysis.py