# Dirichlet Tucker Decomposition
This repository implements a _dynamic latent Dirichlet allocation_, or _dynamic topic model_, of discretized behavioral sequence motifs.
This model is equivalently a Tucker decomposition under certain constraints and Dirichlet prior assumptions.
Thus, we refer to this model as a **Dirichlet Tucker decomposition (DTD)**.

## Installation
```shell
# -----------------------------------------------
# Create virtual environment with python >= 3.10
# -----------------------------------------------
conda create -n dtd python=3.10
conda activate dtd

# ---------------------
# Install dependencies
# ---------------------
pip install -U "jax[cuda12]"  # CPU-install available; see: https://jax.readthedocs.io/en/latest/installation.html
pip install torch --index-url https://download.pytorch.org/whl/cpu  # for dataloading
pip install pathlib tqdm wandb

# ------------------
# Install this repo
# ------------------
git clone git@github.com:lindermanlab/dirichlet-tucker.git
pip install [-e] dirichlet-tucker  # Optional [-e] flag creates an editable installation
```

The full package list with exact package versions used are found in `requirements.txt`

## Repository structure
The repository is organized as follows
```
- dirichlet-tucker/
| - analysis/           # Dataset-specific scripts and analyses
| | - killifish/
| | - serotonin/
| - dtd/                # Source code
```

The following branches (in varying states of active development) currently exist:
- `main`: (currently also contains serotonin / 4d analysis)
- `killifish`: Killifish scripts and analyses
- `poisson-tucker`: Model comparisons against Poisson Tucker variations
    - Currently kept in a separate branch due to active development; will be merged into the `main` upon completion
These branches will be eventually merged in to the `main` branch.

The source code (`./dtd/`) is manually kept in-sync with the main branch.
Selective merging into a target branch (whether updating the main branch or
porting new changes into a working branch) is executed with the following:
```
git checkout BRANCH_TO_MERGE_INTO
git checkout --patch BRANCH_TO_MERGE_FROM FOLDER_OR_PATH_TO_MERGE
```