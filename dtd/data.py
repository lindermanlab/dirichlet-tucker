from jax import Array
from jax.typing import ArrayLike
from typing import Callable, Literal, Optional, Sequence

import jax.numpy as jnp
import jax.random as jr
import numpy as onp
from pathlib import Path
from sklearn.model_selection import GroupKFold
import torch
import torch.utils.data as torchdata

Axis = int | Sequence[int]
KeyArray = Array
PathLike = Path | str
ShapeLike = tuple[int]


def make_speckled_mask(key: KeyArray, mask_frac: float, shape: ShapeLike):
    """Randomly select elements of arr to holdout.
    
    Parameters
    ----------
    key: JAX PRNG key
    mask_frac: float
        Fraction of data to mask out (set to 0).
    shape: ShapeLike
        Shape of resulting mask

    Returns
    -------
    mask: bool array, with shape ``shape``

    """
    
    return jr.bernoulli(key, 1.-mask_frac, shape=shape)


def minmax_scaled_frequency_transform(arr: ArrayLike, axis: Axis=(0,1)) -> Array:
    """Scale data to fall in [0,1] range, then restandardize so last axis sums to 1.

    Parameters
    ----------
    arr: ArrayLike
        Array to calculate min and max values from
    axis: Axis, default=0.
        Axis or axes to compute min and max values along

    Returns
    -------
    Array

    """

    min_val = jnp.min(arr, axis=axis, keepdims=True)
    max_val = jnp.max(arr, axis=axis, keepdims=True)
    scaled_counts = (arr - min_val) / (max_val - min_val)

    return scaled_counts / scaled_counts.sum(axis=-1, keepdims=True)


def frequency_transform(arr: ArrayLike) -> Array:
    """Transform count data so last axis sums to 1.

    Parameters
    ----------
    arr: ArrayLike
        Not used; this transform is independent of any array statistics.

    Returns
    -------
    Array

    """

    return arr / arr.sum(axis=-1, keepdims=True)


def concatenate_sessions(
    data_dir: PathLike,
    *,
    transform_method: Optional[Literal["minmax", "frequency"]]=None,
    transform_kwargs: Optional[dict]=None,
):
    """Construct binned killifish data tensor with subjects concatenated along age axis.

    The expected tensor shape of the `killifish-10min-20230726` dataset is with
    no masking (i.e. ``mask_frac=0``) should be ``(14411, 144, 100)``.

    Parameters
    ----------
    data_dir: PathLike
        Path to directory containing data to load. Each file corresponds to a separate subject.
        Each data file contains the following arrays:
            - "tensor": uint32 ndarray, shape (n_days_recorded, n_bins_per_day, n_syllables).
            - "mode_0": uint16 vector, length (n_days_recorded,).
                Subject age, in days, on the day of recording. The last day of recording
                is the age at death; this value is also referred to as the subject's lifespan.
            - "mode_1": uint16 vector, length (n_bins_per_day,).
                Time bin index, rannging from 0 to ``n_bins_per_day-1``.
            - "mode_2": str vector, length (n_syllables,).
                Label for each syllable, using the ``<cluster_id>-<syllable_id>``,
                e.g. ``['c0-1', 'c0-2', ..., 'c11-99']``. Cluster IDs are obtained by
                hierarchically clustering syllables by feature similarity for visualization
                purposes and are not behaviorally meaningful.
    
    transform_method: {"minmax", "frequency"} or None. default=None.
        Transform method for transforming data tensor:
        - "minmax": Scale data to fall within (0,1) along specifed axis,
          ``(X - X.max(axis=axis)) / (X.max(axis=axis) - X.min(axis=axis))``.
          Note that this method does not reduce the effect of outliers, but it linearly
          scales them down into a fixed range.
        - "frequency": Standardize data to sum to 1 along specified axis, 
          ``X / X.sum(axis=axis)``
        - If None, do nothing.
    
    transform_kwargs: dict. default=None.
        Keywork arguments for transform method. See docstring for specified method.
    
    Returns
    -------
    dict, with items
        - "data": float ndarray, shape (n_total_days_recorded, n_bins_per_day, n_syllables)
            Transformed data tensor, concatenated along the age axis (mode 0)
        - "mode_0": list, length (n_total_days_recorded,).
            Subject-age labels, consisting of tuple ``(<subject_name>, <subject_age>)``.
        - "mode_1": uint16 vector, length (n_bins_per_day,).
            Time bin index, rannging from 0 to ``n_bins_per_day-1``.
        - "mode_2": str vector, length (n_syllables,).
            Syllable labels; see description in ``data_dir`` input parameter.

    """

    print(f"Loading data from...{str(data_dir)}")

    if transform_kwargs is None:
        transform_kwargs = {}

    filepaths = sorted([f for f in Path(data_dir).rglob("*.npz")])

    # Load tensor data
    tensor, mode_0, mode_0_ids = [], [], []
    for fpath in filepaths:
        with onp.load(fpath) as f:
            tensor.append(f['tensor'])
            mode_0.append(f['mode_0'])
            mode_0_ids.extend([fpath.stem]*len(f['mode_0']))

            # Since modes 1 and 2 are already aligned across all files,
            # we will just make use of last ones loaded
            mode_1, mode_2 = f['mode_1'], f['mode_2']

    tensor = jnp.concatenate(tensor, dtype=float)
    mode_0 = jnp.concatenate(mode_0, dtype=int)

    # Transform data tensor
    if transform_method == "minmax":
        tensor = minmax_scaled_frequency_transform(tensor, **transform_kwargs)
    elif transform_method == "frequency":
        tensor = frequency_transform(tensor, **transform_kwargs)
    elif transform_method is None:
        pass

    return {
        'tensor': tensor,
        'mode_0': mode_0,
        'mode_1': mode_1,
        'mode_2': mode_2,
        'mode_0_ids': mode_0_ids
    }


def generate_cross_validation_masks(
    mask_shape: ShapeLike,
    *,
    val_frac: float,
    test_frac: float,
    groups: Optional[Sequence]=None,
    mask_method: Literal["speckle"]="speckle",
    mask_kwargs: Optional[dict]=None,
    n_folds: Optional[int]=None,
    seed: int,
):
    """Generate cross-validation masks.
    
    Parameters
    ----------
    seed: int
        Random seed for PRNGs
    val_frac: float
        Fraction of (held-in) data to mask
    test_frac: float
        Fraction of held-out data to mask.
    shape: ShapeLike
        Shape of mask, (d1, ...)
    groups: sequence or None, length (d1,).
        Labels to split data into non-overlapping groups, i.e. individual subjects.
    mask_method: Literal["speckle"]. default="speckle"
        mask method for held-in data
        - "speckle": Randomly select tensor entries to mask.
    mask_kwargs: dict or None. default=None.
        Keyword arguments for mask method. See docstring for specified method.
    
        
    Yields
    ------
    dict, of bool arrays with keys:
        - 'val'
        - 'buffer'
        - 'test'
    """

    jax_key = jr.key(seed)

    if mask_method == "speckle":
        mask_fn = make_speckled_mask
    else:
        raise ValueError("Expected 'mask_method' to be one of \{'speckle'\}, " + f"but got {mask_method}")

    if mask_kwargs is None:
        mask_kwargs = {}

    if n_folds is None:
        n_folds = int(onp.round(1/test_frac))

    cv = GroupKFold(n_folds)
    
    for i_fold, (train_idxs, test_idxs) in enumerate(cv.split(X=onp.empty(shape), groups=groups)):
        
        # Validation mask: Randomly mask data from held-in subjects 
        val_mask = onp.ones(shape, dtype=bool)
        val_mask[train_idxs] = mask_fn(
                jr.fold_in(jax_key, i_fold),
                val_frac,
                (len(train_idxs),) + shape[1:],
                **mask_kwargs
            )

        # Buffer mask: Currently not used. This will eventuall be returned by `mask_fn`
        buffer_mask = onp.ones(shape, dtype=bool)

        # Test mask: Mask out all rows associated with test subjects
        test_mask = onp.ones(shape, dtype=bool)
        test_mask[test_idxs] = 0

        yield {
            "val": jnp.asarray(val_mask, dtype=bool),
            "buffer": jnp.asarray(buffer_mask, dtype=bool),
            "test": jnp.asarray(test_mask, dtype=bool),
        }