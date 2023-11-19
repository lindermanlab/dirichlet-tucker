#!/bin/bash
#SBATCH --job-name=kf-lofo
#SBATCH --partition swl1
#SBATCH --gpus=1
#SBATCH --mem=256GB
#SBATCH --time=6:00:00
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH --output=/scratch/groups/swl1/killifish/tmp/kf-lofo-%j.out
#SBATCH --error=/scratch/groups/swl1/killifish/tmp/kf-lofo-%j.err

# Adding a comment
source ~/.envs/dtd/bin/activate
echo
echo 'Activated venv:' $(eval "echo -e "\${PATH//:/'\\n'}" | sed -n 1p")

echo
echo SLURM_JOB_ID $SLURM_JOB_ID

python kf_lofo.py