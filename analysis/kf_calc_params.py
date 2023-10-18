"""Calculate maximum minibatch size or minimum GPU memory.

Examples
--------
To calculate the optimal minibatch size:
    python kf_calc_params.py -b '(12730, 144)' -e '(100,)' -k '(40,10,30)' --gpu_mem_gb 80 --gpu_mem_frac 0.9

To calculate the minimum memory:
    python kf_calc_params.py -b '(12730, 144)' -e '(100,)' -k '(40,10,30)' --minibatch_size 18000
"""

import click

from math import prod
import numpy as np
from scipy.signal import argrelmin

def str_to_tuple_int(stup):
    """Convert a tuple of ints formatted as a string into a tuple.
    
    Remove paranetheses on either end -> split at command -> convert to int.
    """
    
    # Remove paranetheses on either end
    # Split at comma
    # Convert to integers
    return tuple([int(x) for x  in stup[1:-1].split(',') if len(x) > 1])

def calc_minibatch_size(batch_prod: int,
                        event_prod: int,
                        rank_prod: int,
                        gpu_mem: float,
                        dtype_size: int=4) -> int:
    """Calculate maximum minibatch size that minimizes last incomplete minibatch."""

    # Calculate nominal minibatch size, given fixed parameters
    m_nominal = int(gpu_mem / rank_prod / event_prod / dtype_size)

    # Find minibatch size near nominal size that minimizes remainder
    ms = np.arange(m_nominal - 512, m_nominal + 1)
    mods = batch_prod % ms

    # Find relative minima
    argmins = argrelmin(mods, order=10)[0]
    minmods = mods[argmins]

    # Choose the largest minibatch size whose mod is below the threshold
    mask = minmods < 120
    assert sum(mask) > 0, 'Expected at least one value to be below threshold, but got none.\n' \
                          + f'(m, batch_prod%m): {list(zip(ms[argmins], minmods))}'
    
    arg_opt = argmins[mask][-1]
    return ms[arg_opt], mods[arg_opt]

def calc_gpu_mem(event_prod: int,
                 rank_prod: int,
                 minibatch_size: int,
                 dtype_size: int=4) -> float:
    """Calculate minimum GPU memory needed to handle specified model."""

    return rank_prod * event_prod * minibatch_size * dtype_size

@click.command()
@click.option('-b', '--batch_shape', type=str, required=True, help='Batch shape of data')
@click.option('-e', '--event_shape', type=str, required=True, help='Event shape of data')
@click.option('-k', '--model_rank', type=str, required=True, help='Model rank shape')
@click.option('--minibatch_size', type=int, default=None)
@click.option('--gpu_mem_gb', type=float, default=None)
@click.option('--gpu_mem_frac', type=float, default=0.75,
              help='Fraction of total GPU mem pre-allocated to JAX. Default 0.75.')
def _main(batch_shape, event_shape, model_rank, minibatch_size, gpu_mem_gb, gpu_mem_frac):
    batch_shape = str_to_tuple_int(batch_shape)
    event_shape = str_to_tuple_int(event_shape)
    model_rank = str_to_tuple_int(model_rank)

    batch_prod = prod(batch_shape)
    event_prod = prod(event_shape)
    rank_prod = prod(model_rank)

    # Check that exactly one of the two variables is specified
    assert (minibatch_size is not None) or (gpu_mem_gb is not None), \
        "Must specify either `minibatch_size` or `gpu`mem_gb`"
    
    assert (minibatch_size is None) or (gpu_mem_gb is None), \
        "Expected only one of `minibatch_size` or `gpu`mem_gb`, but got both."

    # Proceed
    if minibatch_size is not None:
        gpu_mem = calc_gpu_mem(event_prod, rank_prod, minibatch_size)
        print(f"Minimum {gpu_mem / (1024**3):.1f} GB GPU memory required.")
        return gpu_mem
    
    if gpu_mem_gb is not None:
        gpu_mem = gpu_mem_gb * (1024**3) * gpu_mem_frac
        minibatch_size, incomplete_size = calc_minibatch_size(batch_prod, event_prod, rank_prod, gpu_mem)
        print(f"Optimal minibatch size: {minibatch_size}; last batch size: {incomplete_size}.")
        return minibatch_size
    
if __name__ == "__main__":
    _main()