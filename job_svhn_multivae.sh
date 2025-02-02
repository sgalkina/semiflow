#!/bin/bash
#SBATCH -p gpu --gres=gpu:a40:1
#SBATCH --job-name=moflow
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4 --mem=250000M
#SBATCH --time=12:00:00

./train_svhn_multivae.sh