#!/bin/bash
#SBATCH --job-name=test.job
#SBATCH --output=test.out
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4g
echo LEGGOO
module list
nvidia-smi
source ~/eecs542_final/env542/bin/activate
python ~/eecs542_final/training_pipeline.py --n_epochs 100 --model_name densenet --opt_name Adam --lr 1e-3 --dataset mnist --wandb_expt_name densenetMNIST --batch_size 511 --num_workers 16 --corr 0.9