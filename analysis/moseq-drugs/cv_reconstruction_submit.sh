#!/bin/bash
#SBATCH --job-name=dtd-moseq-cv-recon
#SBATCH --partition swl1,stat,owners
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=24G
#SBATCH --time=02:00:00
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --output=/scratch/users/eyz/tmp/slurm/moseq-%j.out
#SBATCH --error=/scratch/users/eyz/tmp/slurm/moseq-%j.out
#SBATCH --array=[1-15]

echo SLURM_ARRAY_JOB_ID $SLURM_ARRAY_JOB_ID
echo SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID
echo SLURM_JOB_ID $SLURM_JOB_ID

# INSTRUCTIONS
# 1. Update sweep file, cv_reconstruction_config_grid.yaml
# 2. Initialize sweep and node the sweep ID
#       > wandb sweep cv_reconstruction_config_grid.yaml
# 3. Update this bash file, particularly the following
#       - WANDB_SWEEP_ID, environment variable
#       - WANDB_PROJECT, environment variable, must match
# 4. Submit this job script
#       > sbatch cv_reconstruction_submit.sh
export WANDB_SWEEP_ID=d30x2ewe
export WANDB_PROJECT=moseq-dtd-sweep-20250225

echo
echo WANDB_SWEEP_ID $WANDB_SWEEP_ID

export WANDB_DIR=/scratch/users/eyz/tmp/

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

# Activate environment
eval "$(conda shell.bash hook)"
conda activate dtd
echo -n "Using "
which python

# # Launch job
cd /home/groups/swl1/eyz/dirichlet-tucker/analysis/moseq-drugs
wandb agent eyz/$WANDB_PROJECT/$WANDB_SWEEP_ID

# For debugging
# python -m pdb /home/groups/swl1/eyz/dirichlet-tucker/analysis/moseq-drugs/cv_reconstruction.py \
#     --data_path="/home/groups/swl1/eyz/data/moseq-drugs/syllable_binned_1min.npz" \
#     --k1 10 --k2 2 --k3 20 --mask_frac 0.2 --mask_block_shape 1 1 --mask_buffer_size 0 0 \
#     --wandb_project "($WANDB_PROJECT)" \
#     --output_dir "/scratch/users/eyz/" \
#     --wandb_debug
