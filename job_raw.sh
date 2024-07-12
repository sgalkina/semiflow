#!/bin/bash
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --job-name=moflow
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4 --mem=250000M
#SBATCH --time=24:00:00

./train_raw.sh