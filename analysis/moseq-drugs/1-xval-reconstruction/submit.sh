#!/bin/bash
#SBATCH --job-name=dtd-moseq-cv-recon
#SBATCH --partition swl1,stat,owners
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=8G
#SBATCH --time=02:00:00
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --output=/scratch/users/eyz/tmp/slurm/moseq-%j.out
#SBATCH --error=/scratch/users/eyz/tmp/slurm/moseq-%j.out
#SBATCH --array=[1-5]

echo SLURM_ARRAY_JOB_ID $SLURM_ARRAY_JOB_ID
echo SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID
echo SLURM_JOB_ID $SLURM_JOB_ID
echo

# INSTRUCTIONS
# 1. Update sweep file, config.yaml
# 2. Initialize sweep and note the sweep id
#       > wandb sweep config.yaml
# 3. Submit job script with WANDB_SWEEP_ID <sweep_id>
#       > sbatch --export=WANDB_SWEEP_ID=<sweep_id> submit.sh

echo WANDB_SWEEP_ID $WANDB_SWEEP_ID

export WANDB_DIR=/scratch/users/eyz/tmp/
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

# Activate environment
eval "$(conda shell.bash hook)"
conda activate dtd
echo -n "Using "
which python

# Launch job
export WANDB_PROJECT=moseq-dtd-sweep-20250225
wandb agent eyz/$WANDB_PROJECT/$WANDB_SWEEP_ID

# For debugging
# python -m pdb /home/groups/swl1/eyz/dirichlet-tucker/analysis/moseq-drugs/crossval_reconstruction/run.py \
#     --data_path="/home/groups/swl1/eyz/data/moseq-drugs/syllable_binned_1min.npz" \
#     --k1 10 --k2 2 --k3 20 --mask_frac 0.2 --mask_block_shape 1 1 --mask_buffer_size 0 0 \
#     --wandb_project "($WANDB_PROJECT)" \
#     --output_dir "/scratch/users/eyz/" \
#     # --wandb_debug