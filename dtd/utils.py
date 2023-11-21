from __future__ import annotations
from typing import Optional, Union
from jaxtyping import Array

import itertools
from pathlib import Path
import shutil

import jax.numpy as jnp
import jax.random as jr
import numpy as onp

import pandas as pd
import wandb

def calculate_minibatch_size(d1,d2,d3,k1,k2,k3,mem_gb,mem_frac=0.75):
    """Calculate minibatch size that maximizes available memory.
    
    Assumes 3D model with batch_dims (0,1).

    If setting mem_frac > 0.75 (the default fraction of GPU that JAX allocates
    on start up), make sure to adjust the environmental variable appropriately:
        XLA_PYTHON_CLIENT_MEM_FRACTION=<mem_frac>

    Note that the resulting minibatch size may not minimize the size of the last
    incomplete batch. To then identify a better minibatch size, you could sweep
    through minibatch sizes up to this returned value. For example:
        import plotly.express as px
        mods = [(m, (d1,d2) % m) for m in range(m_start,m)]
        mod_m, mod_val = list(zip(*mods))
        px.line(x=mod_m, y=mod_val)
    """
    m = (mem_gb*mem_frac) * (1024**3) / 4
    m /= jnp.max(jnp.array([k1,k2,k3])) * d3
    return int(m)

def calculate_memory(d1,d2,d3,k1,k2,k3,minibatch_size):
    """Calculate memory needed for a given minibatch size.
    
    Assumes 3D model with batch_dims (0,1).

    Note that this does NOT account for the memory needed to calculate the 
    incomplete batch (if drop_last=False), which would use _additional_ memory.
    """
    mem_gb = jnp.max(jnp.array([k1,k2,k3])) * d3 * minibatch_size
    # m += d1*d2*d3
    mem_gb *= 4 / (1024**3)
    return mem_gb

class ShuffleIndicesIterator():
    """Custom Iterator that produces minibatches of shuffled indices.
    
    Parameters
        key: PRNGKey
            Used to shuffle the indices at each iteration.
        batch_shape: tuple
            Shape of target array's batch dimensions
        minibatch_size: int
            Number of indices per minibatch
    
    Outputs
        batched_indices:   shape (n_batches, minibatch_size, batch_ndims)
        remaining_indices: shape (n_remaining, batch_ndims)
    """

    def __init__(self, key, batch_shape, minibatch_size):
        self.key = key

        # Sequentially enumerated indices, shape (n_samples, batch_ndims)
        # where n_samples = prod(*batch_shape) and batch_ndims = len(batch_shape)
        self._indices = jnp.indices(batch_shape).reshape(len(batch_shape), -1).T

        self.minibatch_size = minibatch_size
        self.n_complete = len(self._indices) // self.minibatch_size
        self.incomplete_size = len(self._indices) % self.minibatch_size
        self.has_incomplete = self.incomplete_size > 0

    def __iter__(self):
        return self
    
    def __next__(self):
        # Shuffle indices
        _, self.key = jr.split(self.key)
        indices = jr.permutation(self.key, self._indices, axis=0)

        # Split indices into complete minibatch sets and an incomplete set
        batched_indices, remaining_indices \
            = jnp.split(indices, (self.n_complete*self.minibatch_size,))
        batched_indices \
            = batched_indices.reshape(self.n_complete, self.minibatch_size, -1)

        return batched_indices, remaining_indices
    
def get_wnb_project_df(entity: str,
                       project: str,
                       config_keys: list[str],
                       summary_keys: list[str]
                       ) -> pd.DataFrame:
    """Retrieve WandB <entity/project> configs and results.

    
    Parameters
        entity: WandB entity (user) name
        project: WandB project name
        config_keys: List of run configuration (e.g. hyperparameter) to load
        summary_keys: List of run summary (e.g. results) to load
    
    Returns
        pd.DataFrame, with config and summaries, plus
            - id: unique id, used for download files associated wiht run
            - name: human readable name, used in wandb gui
    """

    runs = wandb.Api().runs(f"{entity}/{project}")

    run_results = {
        key: [] for key in itertools.chain(['id', 'name'], config_keys, summary_keys)
    }
    for run in runs:
        # run.summary contains the output keys/values for metrics like accuracy.
        # use `get`` to catch NaN runs (e.g. due to OOM)
        for key in summary_keys:
            run_results[key].append(run.summary.get(key,))

        # run.config contains the hyperparameters.
        for key in config_keys:
            run_results[key].append(run.config[key])

        # 'id' is the unique identifier
        # 'name' is the human-readable name
        run_results['id'].append(run._attrs['name'])
        run_results['name'].append(run.name)

    return pd.DataFrame(run_results)

def download_wnb_params(entity: str,
                        project: str,
                        run_id: str,
                        params_path: Optional[Path]=None,
                        ) -> tuple[Array]:
    """Download WandB <entity/project/run_id> fitted parameters.

    Parameters are saved at the specified PARAMS_PATH. Note that the WandB
    file download api (run.file(FILE).download(root=RELPATH) only handles
    downloading to a path RELPATH specified relative to the current working
    directory. We need to ensure that current directory (e.g. where the repo is
    located) has sufficient space to handle this transfer.
    TODO Open an issue with WandB about this functionality

    If PARAMS_PATH is not specified, save in '../temp/'
    
    Parameters
        entity: WandB entity (user) name
        project: WandB project name
        run_id: WandB unique identifier, not the human-readable name
        params_path: parent directory to save params file to, i.e.
            PARAMS_PATH / RUN_ID / params.npz

    Returns
        params: dict of fitted parameters and associated data
    """

    if params_path is None:
        true_path = Path(f'../temp/{run_id}')
    else:
        true_path = params_path / run_id
    true_path.mkdir(parents=True, exist_ok=True)

    # If the file already exists at PARAMS_PATH, do not re-download
    if not (true_path.exists() and (true_path / 'params.npz').is_file()):
        # Download to local temporary folder
        temp_path = f'../temp/{run_id}'

        wandb.Api().run(f"{entity}/{project}/{run_id}") \
                   .file("params.npz") \
                   .download(root=temp_path, exist_ok=True)

        # If PARAMS_PATH is specified, this means that TEMP_PATH is truly a
        # temporary path, and we should manually move file and delete this folder.
        # However, if PARAMS_PATH is None, then TRUE_PATH == TEMP_PATH, so do nothing
        if params_path is not None:
            shutil.move(Path(temp_path)/'params.npz', true_path)
            shutil.rmtree(temp_path)

    # Load the parameters, retrieve the seed, and return
    params = onp.load(true_path/'params.npz', allow_pickle=True)

    return {k: params[k] for k in params.files}