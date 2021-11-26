#!/bin/bash
#SBATCH --job-name=slurm_logs/iwildcam_baseline.job
#SBATCH --output=slurm_logs/iwildcam_baseline.out
#SBATCH --time=5-0
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G

echo "Starting experiment"
module list
nvidia-smi

pip install --user -r requirements.txt
python training_pipeline.py --n_epochs 12 --model_name resnet --opt_name Adam --lr 3e-5 --dataset iwildcam --batch_size 16 --num_workers 16 --corr 0.9 --wandb_expt_name iwildcam_baseline_corr0.9
