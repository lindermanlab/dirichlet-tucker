from __future__ import annotations
from typing import Optional, Union
from jaxtyping import Array

from math import prod
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
    """Retrieve configs and results of all runs associated with a WandB project.

    Unique run id and human-readable name are automatically downloaded.
    
    Parameters
        entity (str): WandB entity (user) name
        project (str): WandB project name
        config_keys (list[str]): List of run configuration (e.g. hyperparameter) to load
        summary_keys (list[str]): List of run summary (e.g. results) to load
    
    Returns
        pd.DataFrame, with config and summaries, plus
            - id: unique id, used for download files associated with run
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
                        update: bool=False
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
        update: If True, re-download and overwrite existing params

    Returns
        params: dict of fitted parameters and associated data
    """

    if params_path is None:
        true_path = Path(f'../temp/{run_id}')
    else:
        true_path = params_path / run_id
    
    # If `update==True`, delete path at PARAMS_PATH if it exists to trigger re-download
    if update and true_path.exists():
        shutil.rmtree(true_path)

    true_path.mkdir(parents=True, exist_ok=True)

    # If file does not yet exist at PARAMS_PATH, download
    if not (true_path / 'params.npz').is_file():
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


def create_block_speckled_mask(
    rng: onp.random.Generator,
    batch_shape: Sequence[int],
    block_shape: int | Sequence[int]=1,
    buffer_size: int | Sequence[int]=0,
    n_blocks: int=None,
    frac_mask: float=None,
    frac_include_buffer: bool=True,
    exact: bool=False,
):
    """Create block speckled mask and buffer.

    Block speckling allows for masking consecutive subblocks of entries, and buffering
    reduces correlation between held-in and held-out data. This is useful for more
    accurately evaluating model performance on data with spatial or temporal correlations.

    Overlapping blocks are merged together and a buffer is created for the merged blocks.
    
    This function is not jit compatible and is implemented using numpy random generation.
    It uses in-place array assignment, for-loop iterations, and list comprehension.

    Default parameters of block_shape=1 and buffer_size=0 recovers "speckled" masking;
    see [Wold, 1978] and [Williams, 2018].

    Start index randomization and ND-slicing based on `get_block_mask` implementation by
    Cayco-Gajic lab accompanying [Pellegrino et al. 2023].
        https://github.com/caycogajiclab/sliceTCA_paper/blob/62b0ca1267fe9d615111a193080577eda601687e/run_sliceTCA/core/utilities.py#L56
    This implementation includes explicit creation of a buffer mask.

    Parameters
        rng (onp.random.Generator): Numpy random Generator.
        batch_shape (Sequence[int]): Shape to create mask for.
        block_shape (Sequence[int] | int): Block shape to mask out. If int, value is
            repeated across all dimensions. Default: 1, single element.
        buffer_size (Sequence[int] | int)): Amount to mask out for buffering on _each_
            side of the block. If int, value is repeated across all dimensions.
            Default: 0, no buffer.
        n_blocks (int): Number of blocks to create. Mutually exclusive with `frac_mask`;
            only one of the two may be specified.
        frac_mask (float): Fraction of `batch_shape` to mask out. Mutually exculsive with
            `n_blocks`; only one of the two may be specified.
        frac_include_buffer (bool): Whether to include buffered elements when calculating
            number of blocks from `frac_mask`. If True (default); include buffer; this
            results in a smaller effective test split. Else, do not include buffer; this
            results in a smaller effective training split.
        exact (bool): If True, create exactly the number of blocks specified by
            `n_blocks` or `frac_mask`; this approach may be slower for large batch shapes.
            Typically guaranteed to create the same number of blocks each time. If False,
            create _approximately_ the number of blocks specified by `n_blocks` or `frac_mask`.
            Not guaranteed to create the same number of blocks each time. Default: False

    Returns
        mask (onp.array): shape (*batch_shape), bool dtype
            Boolean array indicating which entries to hold out for validation.
        buffer (onp.array): shape (*batch_shape), bool dtype
            Boolean array indicating which entires to discard from validation.

    References
        [Pellegrino et al. 2024] "Dimensionality reduction beyond neural subspaces with
            slice tensor component analysis." Nature Neuroscience, 27, 1199-1210.
        [Williams, 2018.] "How to cross-validate PCA, clustering, and matrix decomposition
            models." Personal blog, 2018 Feb 26, updated 2019 Sept 17.
            https://alexhwilliams.info/itsneuronalblog/2018/02/26/crossval/
        [Wold, 1978.] "Cross-Validatory Estimation of the Number of Components in Factor
        and Principal Components Models." Technometrics, 20, 397-405.
        
    """

    assert not ((n_blocks is not None) and (frac_mask is not None)), \
        f"Expected one of `n_blocks` or `frac_mask` to be specified, got {n_blocks=} and {frac_mask=}."
    assert not ((n_blocks is None) and (frac_mask is None)), \
        f"Only one of `n_blocks` or `frac_mask` may be specified, got {n_blocks=} and {frac_mask=}."

    # Standardize block_shape and buffer_size into sequences
    ndim = len(batch_shape)
    if isinstance(block_shape, int):
        block_shape = (block_shape,) * ndim
    if isinstance(buffer_size, int):
        buffer_size = (buffer_size,) * ndim
    
    # Convert frac_mask into n_blocks
    if frac_mask is not None:
        block_volume = onp.array(block_shape)
        block_volume += 2*onp.array(buffer_size) if frac_include_buffer else 0
        block_volume = prod(block_volume)

        n_blocks = int(frac_mask * prod(batch_shape) / block_volume)

    # Calculate max indices to avoid index out of range
    max_indices = onp.array(batch_shape) - (onp.array(block_shape) + 2*onp.array(buffer_size)) + 1
    flat_max_index = prod(max_indices)

    # Get start indices of blocks
    if exact:
        is_start_index = onp.concatenate(
            [onp.ones(n_blocks, dtype=bool), onp.zeros(flat_max_index-n_blocks, dtype=bool)]
        )
        is_start_index = rng.permutation(is_start_index)
        is_start_index = is_start_index.reshape(*max_indices)
    
    else:
        p = n_blocks / flat_max_index
        is_start_index = rng.binomial(1, p, size=max_indices)  # = Bernoulli distr

    # Create masks
    mask = onp.zeros(batch_shape, dtype=bool)
    buffer = onp.zeros(batch_shape, dtype=bool)
    for start_indices in zip(*onp.nonzero(is_start_index)):
        start_indices = onp.asarray(start_indices) + buffer_size  # account for ante-buffer

        # Construct multi-dimensional slices to mask out block
        ndslices = [
            slice(start_indices[d], start_indices[d]+block_shape[d]) for d in range(ndim)
        ]
        mask[*ndslices] = True

        # Construct multi-dimensional slices to mask out block + buffer
        # "Fill in" the masked area as well; we will unmask areas later
        ndslices = [
            slice(start_indices[d]-buffer_size[d], start_indices[d]+block_shape[d]+buffer_size[d])
            for d in range(ndim)
        ]
        buffer[*ndslices] = True

    # Remove buffer indicator where mask indicator is True
    buffer = onp.where((mask == 1) & (buffer == 1), 0, buffer)

    return mask, buffer