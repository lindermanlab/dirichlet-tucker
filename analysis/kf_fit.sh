#!/bin/bash
#SBATCH --job-name=kf-one
#SBATCH --partition swl1
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-gpu=1
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=5G
#SBATCH --time=24:00:00
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --output=/scratch/groups/swl1/killifish/tmp/kf-one-%j.out
#SBATCH --error=/scratch/groups/swl1/killifish/tmp/kf-one-%j.out


# 12 epochs * (15 min/epoch @ M=3000) = 3 hours

echo SLURM_JOB_ID $SLURM_JOB_ID

source ~/.envs/dtd/bin/activate
echo
echo 'Activated venv:' $(eval "echo -e "\${PATH//:/'\\n'}" | sed -n 1p")

export DATADIR=/scratch/groups/swl1/killifish/p3_20230726-20230915/q2-aligned_10min
export OUTDIR=/scratch/groups/swl1/killifish/tmp
export WANDB_DIR="$OUTDIR"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

# Must manually set project name in killifish.py script
python killifish.py \
    "$DATADIR" --k1 40 --k2 20 --k3 30 --alpha 1.1 \
    --epoch 250 --minibatch_size 16085 \
    --method full --max_samples -1 \
    --outdir "$OUTDIR" \
    --drop_last --wandb
