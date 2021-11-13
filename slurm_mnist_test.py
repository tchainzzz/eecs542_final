#!/bin/bash
#SBATCH --job-name=test.job
#SBATCH --output=.slurm/test.out
#SBATCH --error=.slurm/test.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2g
pip install -r requirements.txt
python training_pipeline.py --n_epochs 1 --model_name resnet --opt_name Adam --lr 1e-3 --dataset mnist --wandb_expt_name ctrenton-gl-test --limit_batches 8

