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

 ## Dataset structure
 We apply DTD to several datasets to demonstrate its broad applicability and various extensions. 

To maintain a common interface, the provided dataloader expects data to be structured as follows:
```
- <dataset>/
| - <session_1>.npz
| - <session_2>.npz
| - ...
```
where within in each archive, there `N+1` arrays, where `N` is the number of modes (or more colloquially, "dimensions") of the tensor:
- `tensor` : The `N`-mode data tensor with shape `(d1, d2, d3,...)`
- `mode_0` : Labels or ids for the first mode, shape `(d1,)`
- `mode_1` : Labels or ids for the second mode, shape `(d2,)`
- `mode_2` : Labels or ids for the third mode, shape `(d3,)`
- and so on for additional modes