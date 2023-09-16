#!/bin/bash
#SBATCH --job-name=kf-sweep
#SBATCH --partition swl1
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=5G
#SBATCH --time=24:00:00
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --output=/scratch/groups/swl1/killifish/tmp/kf-sweep-swl1-%A-%a.out
#SBATCH --error=/scratch/groups/swl1/killifish/tmp/kf-sweep-swl1-%A-%a.out
#SBATCH --array=[1-4]

# noble-sun-288: 48 min to run 100 epochs at M=16384, on all data

echo SLURM_ARRAY_JOB_ID $SLURM_ARRAY_JOB_ID
echo SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID
echo SLURM_JOB_ID $SLURM_JOB_ID

source ~/.envs/dtd/bin/activate
echo
echo 'Activated venv:' $(eval "echo -e "\${PATH//:/'\\n'}" | sed -n 1p")

echo
echo SWEEP_ID $SWEEP_ID

export DATADIR=/scratch/groups/swl1/killifish/p3_20230726-20230915/q2-aligned_10min
export OUTDIR=/scratch/groups/swl1/killifish/tmp

export WANDB_DIR="$OUTDIR"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

wandb agent --count 1 eyz/kf-dtd-230726/"$SWEEP_ID"
