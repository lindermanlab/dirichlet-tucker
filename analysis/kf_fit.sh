#!/bin/bash
#SBATCH --job-name=kf-one-64
#SBATCH --partition swl1
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-gpu=2
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=8G
#SBATCH --time=05:00:00
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --output=/scratch/groups/swl1/killifish/tmp/kf-one-%j.out
#SBATCH --error=/scratch/groups/swl1/killifish/tmp/kf-one-%j.out


# 12 epochs * (15 min/epoch @ M=3000) = 3 hours

echo SLURM_JOB_ID $SLURM_JOB_ID

source ~/.envs/dtd/bin/activate
echo 'Activated venv:' $(eval "echo -e "\${PATH//:/'\\n'}" | sed -n 1p")

export DATADIR=/scratch/groups/swl1/killifish/p3_20220608-20230208/q2-aligned_bin-20230413/aligned_counts_10min
export OUTDIR=/scratch/groups/swl1/killifish/tmp
export WANDB_DIR="$OUTDIR"

python killifish.py \
    "$DATADIR" --k1 30 --k2 5 --k3 30 --alpha 1.1 \
    --seed 1230 \
    --epoch 500 --minibatch 256 \
    --method stochastic --max-samples 200 \
    --outdir "$OUTDIR" --wandb
