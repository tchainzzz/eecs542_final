#!/bin/bash
#SBATCH --job-name=slurm_out/camelyon17_densenet_baseline.job
#SBATCH --output=slurm_out/camelyon17_densenet_baseline.out
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G

echo "Starting experiment"
module list
nvidia-smi

pip install --user -r ../requirements.txt
python ../training_pipeline.py --n_epochs 5 --model_name densenet --opt_name SGD --opt_kwargs momentum=0.9 weight_decay=0.01 --lr 1e-3 --dataset camelyon17 --batch_size 32 --num_workers 16 --corr 0.9 --wandb_expt_name camelyon_densenet
