#!/bin/bash
#SBATCH --job-name=kf-sweep
#SBATCH --partition=owners
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint=GPU_MEM:16GB|GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:40GB|GPU_MEM:48GB|GPU_MEM:80GB
#SBATCH --mem=5G
#SBATCH --time=12:00:00
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --output=/scratch/groups/swl1/killifish/tmp/kf-sweep-small-%A-%a.out
#SBATCH --error=/scratch/groups/swl1/killifish/tmp/kf-sweep-small-%A-%a.out
#SBATCH --array=[1-12]

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

wandb agent --count 1 eyz/kf-dtd-230726/"$SWEEP_ID"
