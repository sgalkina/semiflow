module load cuda/11.8
module load gcc/11.2.0
module load anaconda3/2023.03-py3.10

source ~/.bashrc
conda activate semiflow

/home/nmb127/.conda/envs/semiflow/bin/python train-flow-ssl.py --config config.yaml