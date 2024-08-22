from jax import Array
from jax.typing import ArrayLike
from typing import Callable, Literal, Optional, Sequence

import jax.numpy as jnp
import jax.random as jr
import numpy as onp
from pathlib import Path
import torch
import torch.utils.data as torchdata

Axis = int | Sequence[int]
KeyArray = Array
PathLike = Path | str


def make_speckled_mask(key: KeyArray, arr: ArrayLike, mask_frac: float):
    """Randomly select elements of arr to holdout.
    
    Parameters
    ----------
    key: JAX PRNG key
    arr: ndarray
    mask_frac: float
        Fraction of data to mask out (set to 0).

    Returns
    -------
    mask: bool ndarray, same shape as ``arr``

    """
    return jr.bernoulli(key, 1-mask_frac, shape=jnp.asarray(arr).shape)


def make_minmax_transform(arr: ArrayLike, axis: Axis=0) -> Callable[[ArrayLike], Array]:
    """Scale data to fall within (0,1) range.

    Parameters
    ----------
    arr: ndarray
        Array to calculate min and max values from
    axis: axis, default=0.
        Axis or axes to compute min and max values along

    Returns
    -------
    Callable[[ArrayLike], Array],
        Function that applies transform to input array.
        Array must be broadcastable to the non-``axis`` axes of ``arr``.

    """

    min_val = jnp.min(arr, axis=axis, keepdims=True)
    max_val = jnp.max(arr, axis=axis, keepdims=True)

    def transform(x: ArrayLike) -> Array:
        return (x - min_val) / (max_val - min_val)

    return transform


def make_frequency_transform(arr: ArrayLike, axis: Axis=-1) -> Callable[[ArrayLike], Array]:
    """Transform count data to sum to 1.

    Parameters
    ----------
    arr: ndarray
        Not used; this transform is independent of any array statistics.
    axis: axis, default=-1.
        Axis or axes to divide data by such that resulting values sum to 1.

    Returns
    -------
    Callable[[ArrayLike], Array],
        Function that applies transform to input array.

    """

    def transform(x: ArrayLike) -> Array:
        return x / x.sum(axis=axis, keepdims=True)

    return transform


def construct_subject_concatenated_kf(
    data_dir: PathLike,
    *,
    holdin_mask_frac: float,
    holdout_subjects_frac: Optional[float]=None,
    holdout_subjects_names: Optional[Sequence[str]]=None,
    masking_method: Literal["speckle"]="speckle",
    masking_kwargs: Optional[dict]=None,
    transform_method: Optional[Literal["minmax", "frequency"]]="minmax",
    transform_kwargs: Optional[dict]=None,
    key: KeyArray,
):
    """Construct binned killifish data tensor with subjects concatenated along age axis.

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
    
    holdin_mask_frac: float
        Fraction of held-in data to mask.

    holdout_subjects_frac: float or None. default=None.
        Fraction of subjects to holdout. These will be selected randomly.
        Only one of ``holdout_subjects_frac`` and ``holdout_subjects_names`` may be specified.

    holdout_subjects_names: sequence of strings or None. default=None.
        Sequence of subjects to holdout, by filename stem. These will be fixed.
        Only one of ``holdout_subjects_frac`` and ``holdout_subjects_names`` may be specified.

    masking_method: {"speckle"}. default="speckle"
        Masking method for held-in data.:
        - "speckle": Randomly select tensor entries to mask.

    masking_kwargs: dict or None. default=None.
        Keyword arguments for masking method. See docstring for specified method.

    transform_method: {"minmax", "frequency"} or None. default="minmax"
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
    
    key: KeyArray
        JAX PRNG key.

    Returns
    -------
    dict, with items
        - "data": float ndarray, shape (n_total_days_recorded, n_bins_per_day, n_syllables)
            Transformed data tensor, concatenated along the age axis (mode 0)
        - "mask": bool ndarray, shape (n_total_days_recorded, n_bins_per_day, n_syllables)
            Boolean array indicating whether a tensor entry should be included in
            the fitting loss (1) or excluded (0).
        - "mode_0": list, length (n_total_days_recorded,).
            Subject-age labels, consisting of tuple ``(<subject_name>, <subject_age>)``.
        - "mode_1": uint16 vector, length (n_bins_per_day,).
            Time bin index, rannging from 0 to ``n_bins_per_day-1``.
        - "mode_2": str vector, length (n_syllables,).
            Syllable labels; see description in ``data_dir`` input parameter.

    """

    holdout_key, masking_key = jr.split(key)

    if all([holdout_subjects_frac, holdout_subjects_names]) or not any([holdout_subjects_frac, holdout_subjects_names]):
        raise ValueError(
            "Expected exactly one of 'holdout_subjects_frac` and `holdout_subjects_names "
            + f"to be not None, but got {holdout_subjects_frac=}, {holdout_subjects_names=}."
        )

    if masking_kwargs is None:
        masking_kwargs = {}

    if transform_kwargs is None:
        transform_kwargs = {}

    # Split subjects into a hold-in set and and hold-out set 
    filepaths = sorted([f for f in Path(data_dir).rglob("*.npz")])

    if holdout_subjects_frac:
        n_holdout_subjects = int(holdout_subjects_frac * len(filepaths))
        holdout_indices = jr.permutation(holdout_key, len(filepaths))[:n_holdout_subjects]
    else:
        holdout_indices = [
            i for i, fpath in enumerate(filepaths) if fpath.stem in holdout_subjects_names
        ]
    
    # Sort indices and pop in reverse to avoid corrupting list when pop
    # After popping, `filepaths` will only contain file paths associated with held-in subjects
    holdout_indices.sort()
    holdout_filepaths = sorted([filepaths.pop(i) for i in holdout_indices[::-1]])

    n_holdin, n_holdout = len(filepaths), len(holdout_filepaths)

    # Load held-in data
    holdin_data, holdin_mode_0 = [], []
    for fpath in filepaths:
        with onp.load(fpath) as f:
            holdin_data.append(f['tensor'])
            holdin_mode_0.append(
                list(zip([fpath.stem]*len(f['mode_0']), f['mode_0'].tolist()))
            )  # mode_0 labels are now (subject_name, subject_age) tuples
    holdin_data = jnp.concatenate(holdin_data, dtype=float)

    # Construct held-in mask
    if masking_method == "speckle":
        holdin_mask = make_speckled_mask(masking_key, holdin_data, holdin_mask_frac, **masking_kwargs)

    # Load held-out data, create a mask of all 0's
    holdout_data, holdout_mode_0 = [], []
    for fpath in filepaths:
        with onp.load(fpath) as f:
            holdout_data.append(f['tensor'])
            holdout_mode_0.append(
                list(zip([fpath.stem]*len(f['mode_0']), f['mode_0'].tolist()))
            )  # mode_0 labels are now (subject_name, subject_age) tuples
            
            # Since data is already aligned across files, mode 1 and 2 labels are shared
            # across all data files. So, we just use the last ones loaded
            mode_1 = f['mode_1']
            mode_2 = f['mode_2']

    holdout_data = jnp.concatenate(holdout_data, dtype=float)
    holdout_mask = jnp.zeros_like(holdout_data, dtype=bool)  # mask out all ooutputf heldout data

    # Combine held-in and held-out data
    data = jnp.concatenate([holdin_data, holdout_data], axis=0)
    mask = jnp.concatenate([holdin_mask, holdout_mask], axis=0)
    mode_0 = holdin_mode_0 + holdout_mode_0

    # Transform data tensor based on held-in statistics
    if transform_method == "minmax":
        transform = make_minmax_transform(holdin_data, **transform_kwargs)
    elif transform_method == "frequency":
        transform = make_frequency_transform(holdin_data, **transform_kwargs)
    elif transform_method is None:
        transform = lambda arr: jnp.asarray(arr, dtype=float)
    data = transform(data)

    return {'data': data, 'mask': mask, 'mode_0': mode_0, 'mode_1': mode_1, 'mode_2': mode_2}
    