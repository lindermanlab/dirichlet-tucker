#!/bin/bash
#SBATCH --job-name=serotonin-tucker-decomp-4d
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-gpu=1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=12gb
#SBATCH --time=23:59:00
#SBATCH -p swl1

echo ${SLURM_ARRAY_TASK_ID}
source ~/bin/load_modules.sh
cd /home/groups/swl1/swl1/dirichlet-tucker/analysis
python serotonin4d.py --km_min=2 --km_max=4 --kn_min=2 --kn_max=8 --kp_min=2 --kp_max=8 --ks_min=2 --ks_max=20 --k_step=2 --num_restarts=5 --num_iters=2000 --tol=0.0001