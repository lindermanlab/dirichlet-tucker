#!/bin/bash
#SBATCH --job-name=moseq-predict
#SBATCH --partition swl1,stat,owners
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16G
#SBATCH --time=02:00:00
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --output=/scratch/users/eyz/tmp/slurm/moseq-predict-%j.out
#SBATCH --error=/scratch/users/eyz/tmp/slurm/moseq-predict-%j.out
#SBATCH --array=[1-3]

echo SLURM_ARRAY_JOB_ID $SLURM_ARRAY_JOB_ID
echo SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID
echo SLURM_JOB_ID $SLURM_JOB_ID
echo

# INSTRUCTIONS
# 1. Update sweep file, config.yaml
# 2. Initialize sweep and note the sweep id
#       > wandb sweep config.yaml
# 3. Submit job script with WANDB_SWEEP_ID <sweep_id>
#       > sbatch --export=WANDB_SWEEP_ID=<sweep_id> sweep.sh

echo WANDB_SWEEP_ID $WANDB_SWEEP_ID  # Expected to passed in

source .env  # Set WANDB_ENTITY, WANDB_PROJECT variables
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

# Activate environment
eval "$(conda shell.bash hook)"
conda activate dtd
echo -n "Using "
which python

# Launch job
wandb agent $WANDB_ENTITY/$WANDB_PROJECT/$WANDB_SWEEP_ID