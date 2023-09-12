#!/bin/bash
#SBATCH --job-name=kf-epoch
#SBATCH --partition swl1
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-gpu=2
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=8G
#SBATCH --time=03:00:00
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --output=/scratch/groups/swl1/killifish/tmp/kf-one-%A-%a.out
#SBATCH --error=/scratch/groups/swl1/killifish/tmp/kf-one-%A-%a.out
#SBATCH --array=[1-2]

# 12 epochs * (15 min/epoch @ M=3000) = 3 hours

echo SLURM_ARRAY_JOB_ID $SLURM_ARRAY_JOB_ID
echo SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID
echo SLURM_JOB_ID $SLURM_JOB_ID

source ~/.envs/dtd/bin/activate
echo
echo 'Activated venv:' $(eval "echo -e "\${PATH//:/'\\n'}" | sed -n 1p")

echo
echo SWEEP_ID $SWEEP_ID

export DATADIR=/scratch/groups/swl1/killifish/p3_20220608-20230208/q2-aligned_bin-20230413/aligned_counts_10min
export OUTDIR=/scratch/groups/swl1/killifish/tmp

export WANDB_DIR="$OUTDIR"

wandb agent --count 1 eyz/kf-dtd/"$SWEEP_ID"
