#!/bin/bash
#SBATCH --job-name=slurm_logs/mnist_densenet_baseline.job
#SBATCH --output=slurm_logs/mnist_densenet_baseline.out
#SBATCH --time=1-0
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G

module list
nvidia-smi

pip install --user -r ../requirements.txt
python ../training_pipeline.py --n_epochs 100 --model_name densenet --opt_name Adam --lr 1e-3 --dataset mnist --wandb_expt_name mnist_densenet --corr 0.9 --batch_size 512 --num_workers 16 # sync afterwards

