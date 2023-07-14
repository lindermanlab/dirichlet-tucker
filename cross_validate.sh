#!/bin/bash
#SBATCH --job-name=serotonin-tucker-decomp
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-gpu=1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=12gb
#SBATCH --time=23:59:00
#SBATCH -p swl1

echo ${SLURM_ARRAY_TASK_ID}
# which python3.9
# which pip3.9
# pip3.9 list
# nvidia-smi

seed=0;
for km in {2..20};
    do for kn in {2..12};
        do for km in {2..20};
            ((seed=seed+1))
            do python3.9 tucker.py --km=$km --kn=$kn --kp=$kp --seed=$seed
        done;
    done;
done;
