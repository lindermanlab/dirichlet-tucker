#!/bin/bash
#SBATCH --job-name=kf-one
#SBATCH --partition swl1
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-gpu=1
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=5G
#SBATCH --time=00:30:00
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

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.98

# Remember to MANUALLY set project name in top of killifish.py script
python -m pdb killifish.py \
    "$DATADIR" --k1 20 --k2 4 --k3 30 --alpha 1.1 \
    --data_seed 123 --init_seed 246 --val_frac 0.20 --test_frac 0.2 \
    --epoch 500 \
    --n_splits 1 \
    --method full \
    --out_dir "$OUTDIR" \
    --drop_last
