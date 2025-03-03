# Cross-validating reconstruction error

Cross-validate model dimensions by evaluating the reconstruction loss (negative log likelihood) of fitted model on held-out entries.

Entries are held-out via **speckled sequence masking**, with 
sequence masks of approximately 3 time bins, `block_shape=(1,3)`, and buffer by 1 time bin on both ends, `buffer_size=(0,1)`.

WandB Project: [moseq-dtd-sweep-20250225](https://wandb.ai/eyz/moseq-dtd-sweep-20250225)

## Workflow
1. **Run cross-validation**
    1. Update `config_grid.yaml` with `k1`, `k2`, `k3` values to sweep over.
    2. Launch `N` sweeps for the given set of configurations. *(see note)*.
        ```shell
        wandb sweep config_grid.yaml

        # Copy the returned <sweep_id>
        sbatch --export=WANDB_SWEEP_ID=<sweep_id> submit.sh
        ```

        *NB: The `run.py` script supports iterating through `N` splits of the data and launching a unique Run instance for split. However, when the script is run via the WandB split, it is unable to do so. Hence, the current workaround is to run the script with single split and launch `N` sweeps instead.*

2. **Analyze cross-validation results** via `Analyze_cross_validation_results.ipynb`.

## Directory structure
```bash
- .env              # Contains WNB_ENTITY and WNB_PROJECT variables (Not uploaded)
- Analyze_cross_validation_results.ipynb
- config_grid.yaml  # Wandb sweep configuration file
- README.md         # This file
- results.csv       # Pandas DataFrame CSV of cross-validation results 
- run.py            # Python script for running N folds for given config
- submit.sh         # Slurm job script for launching wandb agents
```

Some file-specific notes:
- `results.csv`. This is a local copy of the WandB project results.
    - Column headers: `run_id`, `run_name`, `k1`, `k2`, `k3`, `avg_lp.min`, `avg_test_ll`
    - This file is generated/read by `Analyze_cross_validation_results.ipynb`.