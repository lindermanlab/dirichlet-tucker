"""Convert binned syllable usage frequencies (Pandas .csv) into a tensor of counts (.npz).

This script is valid for `df_reformat_10_20230726.csv` shared by the Brunet/Deisseroth labs.
The data is stored as syllable usage frequencies in a Pandas dataframe (.csv extension).
This script converts the frequencies into  tensor of counts and saves these counts,
plus auxiliary data (i.e. age, time-of-day), into a zipped numpy format.

Input format:
    Usage frequency (scaled by bin size) is stored in tidy-format as a pd.DataFrame.
    This data frame contains the following columns:
        - full_fish_name: Name, consisting of {table_id}_cohort_{cohort_id}
        - genotype: Genotype label
        - status: Experiment status
        - age_days: Age in days for each bin
        - minutes: Time-of-day label
        - frames: Number of frames used to compute frequency usage
        - {state_0 ... state_N}

Output format:
    Each subjects data is saved in a seperate file, which can be opened by calling
        out = onp.load(path/to/output/file.npz, allow_pickle=True)
    Thes files consists of four onp.ndarrays:
        - tensor: (n_days_recorded, n_timebins, n_syllables), uint32 dtype
        - mode_0: (n_days_recorded,), uint16 dtype.
        - mode_1: (n_timebins,), uint16 dtype.
        - mode_2: (n_syllables,), str dtype. ['c0-1', 'c0-2', ..., 'c11-99']
            Syllable labels, <cluster_id>-<syllable_id>. Cluster ID is based on
            agglomorative clustering of symmetric KL-divergences of syllable
            feature, for easier visualization. Clusters are not equivalent to topics.

Usage:
    python 00_convert_freq_to_counts
"""


from typing import Callable, Union
from jax import Array

import argparse
from collections import OrderedDict
import itertools
import os
from pathlib import Path

import jax
import jax.random as jr
import numpy as onp
import pandas as pd
from tqdm.auto import tqdm


Dtype = Union[type, onp.dtype, str]
PRNGKey = Array

# Syllable clusters based on agglomerative clustering of symmetric KL-divergences
# of syllable parameters. These clusterings are purely based on feature similarities
# and used for visualization; they do not represent meaningful topics.
SYLLABLE_PERM_DICT = OrderedDict([
    ('c0', [86, 16, 91, 26, 35, 27, 55, 57, 74,  0, 90,  3, 47]),
    ('c1', [67, 87, 63, 32, 54,  5, 69,  4, 49,  2, 80, 83, 18, 38, 11,  8, 66]),
    ('c2', [43, 78, 10, 20,31]),
    ('c3', [15, 76]),
    ('c4', [77, 84]),
    ('c5', [73, 24,  1, 44]),
    ('c6', [75, 62, 85]),
    ('c7', [41, 60, 88, 25, 53, 37, 82, 29, 50]),
    ('c8', [23, 79, 46, 81, 51, 59, 58, 94, 96, 65, 97, 56, 36,14, 17, 89]),
    ('c9', [30, 45, 71, 40, 93, 39, 52]),
    ('c10', [22, 70, 19, 92,  7, 13, 28, 64, 42, 61, 21, 99,  9, 33]),
    ('c11', [12, 34, 68,  6, 48, 95, 72, 98])
])
SYLLABLE_PERM = list(itertools.chain.from_iterable(SYLLABLE_PERM_DICT.values()))


def get_valid_ilocs_by_fish(src_path: Path) -> list[tuple[str, int, int]]:
    """Get name and data rows of all wild-type, status-confirmed subjects from master file.

    Read all rows of data but only selected columns.
    Return a list of tuples consisting of (full_fish_name, start_row_index, n_rows).
    """

    # Only read columns of dataframe needed to identifying valid individuals
    # `usecols` arguments converts all data indices to 0-index (default: 1-index)
    cols = ['full_fish_name', 'genotype', 'status']
    df = pd.read_csv(src_path, usecols=cols)

    # Filter fish
    is_valid = (df['genotype'] == 'wt') & (df['status'] == 'd')
    df = df.loc[is_valid]

    # Record fish name, starting index, and number of rows
    name_start_nrows = []
    for name in df['full_fish_name'].unique():
        
        idxs = df[df['full_fish_name']==name].index

        # Check that an individual fish's data is stored contiguously
        d_idxs = (idxs[1:] - idxs[:-1])
        if not onp.all(d_idxs == 1):
            print(f'[WARNING] {name}: Expected data to be a contiguous chunk, but found discontinuities at {onp.nonzero(d_idxs>1)[0]}. Skipping.')
            continue

        # Add information to output list. Convert start index back to 1-index convention.
        name_start_nrows.append((name, idxs[0]+1, len(idxs)))
    
    del df
    return name_start_nrows

def get_subdf(src_path: Path, start_row: int, n_rows: int, n_syllables: int) -> pd.DataFrame:
    """Extract data corresponding to specific subject from source file.
    
    Read selected columns of specified rows.
    Return a DataFrame containing the subject-specific data.
    """

    # Specify the columns to read and the order that we want them in
    state_col_names = [f'state_{i}' for i in range(n_syllables)]
    col_names = [
        'age_days',                 # Age in days
        'minutes',                  # Minute of day identifier, values {0, 10, 20, ..., 1430}
        'frames',                   # Number of frames used in the bin
        ] + state_col_names

    # Since we skip the rows as the beginning of the file, we cannot rely on using
    # column names. So, explicitly identify the column _indices_ to read.
    tmp = pd.read_csv(src_path, nrows=0)
    col_indices = [tmp.columns.get_loc(col_name) for col_name in col_names]
    del tmp

    # Read the specified columns and rows. Rename columns with key-based names
    df = pd.read_csv(src_path, header=None, usecols=col_indices, skiprows=start_row, nrows=n_rows)[col_indices]
    df = df.set_axis(col_names, axis='columns')

    return df

def make_counts_tensor(df: pd.DataFrame,
                       n_minutes_per_bin: float,
                       frame_rate: int,
                       rtol_frames_per_bin: float=0.05,
                       atol_bins_per_day: int=4,
                       dtype: Dtype='uint32',
                       ) -> tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    """Convert fraction-of-bin-minutes tidy-form data into tensor of counts.

    Returned tensor of counts sum to AROUND `n_frames_per_bin`, not not necessarily
    exactly. This tolerance is specified by `rtol_frames_per_bin`.
    
    DataFrame consists of columns:
        age_days (int): Age in days
        minutes (int): Bin minute identifier. For example, if bin_minutes = 10,
                       then values consist of {0, 10, ..., 1430}
        frames (int): Number of frames per bin
        state_0 (float64): Fraction usage in bin, scaled by bin_minutes for state 0
        ...
        state_N (float64)
    """

    n_frames_per_bin = int(n_minutes_per_bin * 60 * frame_rate)
    n_bins_per_day = int(24 * 60 / n_minutes_per_bin)

    # Convert scaled frequency usage to counts
    df.iloc[:,3:] = (df.iloc[:, 3:] / n_minutes_per_bin * n_frames_per_bin).round()

    # Replace bins with out-of-tolerance number of frames bins with NaNs
    min_frames = n_frames_per_bin * (1 - rtol_frames_per_bin)
    max_frames = n_frames_per_bin * (1 + rtol_frames_per_bin)
    idx_valid = (df['frames'] >= min_frames) \
                 & (df['frames'] <= max_frames) \
                 & df['age_days'].notna()
    df.iloc[~idx_valid, 3:] = onp.nan

    # Drop the frames column
    df.drop('frames', axis=1, inplace=True)

    # Pivot dataframe; organized by age (rows) and [states, minutes] (multi-index columns)
    df = df.pivot(index='age_days', columns='minutes')
    n_syllables = df.columns.levshape[0]

    # Remove days that are missing too many bins. Only need to evaluate for a
    # single state, because if a bin is missing, it's missing for all states.
    df = df[df['state_0'].count(axis=1) > (n_bins_per_day - atol_bins_per_day)]

    # Fill in remaining missing bins, by propagating last valid time bin value
    df = df.fillna(method='ffill', axis=1)
    df = df.fillna(method='bfill', axis=1)

    assert onp.all(df['state_0'].count(axis=1) == n_bins_per_day), 'Expected all bins per day to be valid'

    # Convert counts data to a numpy array (yay!)
    counts_tensor = df.to_numpy(dtype=dtype)\
                      .reshape(-1, n_syllables, n_bins_per_day).transpose((0,2,1))
    counts_tensor = counts_tensor[:,:,SYLLABLE_PERM]  # permute syllables

    # Construct ids for each mode
    age_ids = df.index.to_numpy(dtype='uint16')

    # bin_ids = onp.asarray(
    #     pd.date_range('00:00:00', "23:59:59", freq=f'{n_minutes_per_bin}min').time,
    #     dtype=object)
    bin_ids = onp.arange(n_bins_per_day, dtype='uint16')
    
    syllable_ids = []
    for cluster_name, cluster_syllables in SYLLABLE_PERM_DICT.items():
        syllable_ids += [
            f'{cluster_name}-{len(syllable_ids)+i}' for i in range(len(cluster_syllables))
        ]
    syllable_ids = onp.asarray(syllable_ids)

    return counts_tensor, (age_ids, bin_ids, syllable_ids)

def normalize_counts_tensor(counts_tensor: onp.ndarray,
                            target_counts: int,
                            seed: PRNGKey,
                            dtype: Dtype='uint32') -> onp.ndarray:
    """Normalize counts tensor (in-place) so that each bin of data has exactly `target_counts`."""
    
    _, _, n_states = counts_tensor.shape

    # Get signed difference from target. If difference is positive, the bin is
    # missing frames. If difference is negative, the bin has too many frames.
    signed_diffs = target_counts - counts_tensor.sum(axis=-1).astype('int16')

    for i, j in zip(*signed_diffs.nonzero()):
        s_diff = signed_diffs[i,j]
        sgn, diff = onp.sign(s_diff), int(onp.abs(s_diff))

        # Choose from among the nonzero states to add or subtract difference
        _valid_states = (counts_tensor[i,j] >= diff).nonzero()[0]

        _seed = jr.fold_in(jr.fold_in(seed, j), i)
        chosen_states = jr.choice(_seed, _valid_states, (diff,))

        # Convert chosen states into a one-hot vector and standardize counts tensor
        counts_tensor[i,j] += (sgn * onp.eye(n_states)[chosen_states,:].sum(axis=0)).astype(dtype)

    assert onp.all(counts_tensor.sum(axis=-1) == target_counts), f'Expected counts to be normalized to exactly `target_counts`={target_counts}.'

    return

def main(args):
    src_path = Path(args.src_path)
    assert src_path.is_file(), f'Expected {src_path} to be a file.'
    assert src_path.suffix == '.csv', f'Expected {src_path} to be a .csv file'

    out_dir = Path(args.out_dir) if args.out_dir is not None else src_path.parent
    
    # Make output directory if does not already exist
    try: out_dir.mkdir(parents=True)
    except FileExistsError:
        if ~(os.listdir(out_dir) == []):
            print("[WARNING] Directory already exists and contains files. Overwriting.")
        pass
    
    print(f'Loading data from:\t{src_path}')
    print(f"Saving binned data to:\t{out_dir}")

    # ----------------------------------------------------------------------

    seed = jr.PRNGKey(args.seed)

    n_minutes_per_bin = args.n_minutes_per_bin
    frame_rate = args.frame_rate
    n_syllables = args.n_syllables

    n_frames_per_bin = n_minutes_per_bin * frame_rate * 60
    
    # Find minimal counts dtype, given upper bound on number of frames per bin
    for tensor_dtype in ['uint8', 'uint16', 'uint32', 'uint64']:
        if n_frames_per_bin < onp.iinfo(tensor_dtype).max: break
    print(f"\nUsing dtype {tensor_dtype} for counts tensor (max value {onp.iinfo(tensor_dtype).max})")

    rtol_frames_per_bin = args.rtol_frames_per_bin
    atol_bins_per_session = args.atol_bins_per_session

    # ----------------------------------------------------------------------

    # Get index locations for each valid fish in source file
    print("\nGetting subject ilocs...", end="")
    name_start_nrows_list = get_valid_ilocs_by_fish(src_path)
    print(f"Found {len(name_start_nrows_list)} subjects.")

    # For each fish, convert data to counts tensor and save
    for name, start_row, n_rows in tqdm(name_start_nrows_list):
        # Get sub-dataframe corresponding to current fish
        subject_df = get_subdf(src_path, start_row, n_rows, n_syllables)

        # Convert usage frequency to counts tensor
        counts_tensor, (session_ids, bin_ids, syllable_ids) \
                                    = make_counts_tensor(subject_df,
                                                        n_minutes_per_bin,
                                                        frame_rate,
                                                        rtol_frames_per_bin,
                                                        atol_bins_per_session,
                                                        tensor_dtype)

        # Normalize counts tensor (in-place) so each bin of data sums exactly to `n_frames_per_bin`
        normalize_counts_tensor(counts_tensor, n_frames_per_bin, seed, tensor_dtype)
        assert onp.all(counts_tensor.sum(axis=-1) == n_frames_per_bin), f'Expected counts to be normalized to exactly `target_counts`={n_frames_per_bin}.'
    
        # Save results to a .npz file
        fname = f'fish{name}.npz'
        onp.savez_compressed(out_dir/fname,
                             tensor=counts_tensor,
                             mode_0=session_ids,
                             mode_1=bin_ids,
                             mode_2=syllable_ids)
        tqdm.write(f'\tSaved results to: {out_dir/fname}')

    return 

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Convert syllable usage frequency to tensor of counts.')
    parser.add_argument(
        '--src_path', type=str,
        help='Path to .csv file containing syllable usage frequency.')
    parser.add_argument(
        '--out_dir', type=str, default=None,
        help='Folder to save resulting binned data. If None, save in `src_dir`.')

    parser.add_argument(
        '--n_minutes_per_bin', type=float, default=10.,
        help='Bin size, in minutes. Default: 10 minutes')
    parser.add_argument(
        '--rtol_frames_per_bin', type=float, default=0.05,
        help='Relative tolerance for # of missing/extra frames per bin, as a fraction of bin size.')
    parser.add_argument(
        '--atol_bins_per_session', type=int, default=4,
        help='Absolute tolerance for # of missing bins allowed per session.')

    parser.add_argument(
        '--seed', type=int,
        help='Seed for PRNG; used to standardize all bins to have exactly the same number of counts.')

    parser.add_argument(
        '--frame_rate', type=float, default=20.,
        help='Frame rate of raw syllable label data. Default: 20  Hz [frames / sec].')
    parser.add_argument(
        '--n_syllables', type=int, default=100,
        help='Number of syllables/categories to encode. This should match number of syllables from HMM fit.')

    args = parser.parse_args()
    main(args)
