#!/bin/bash

# Runs classification for a single model-dimension configuration, over all targets
source .env
seed=13  # SHARED so that all models run with same data splits

model=sum
dim=90
target_names=(drug dose class)
for target in "${target_names[@]}"; do
    python run.py \
        --file_path "$DATA_DIR"/syllable_binned_1min.npz \
        --output_dir "$OUTPUT_DIR" \
        --seed "$seed" \
        --target_name "$target" \
        --model_name "$model" \
        --model_dim "$dim" \
        --use_wandb
done