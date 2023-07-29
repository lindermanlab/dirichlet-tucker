from __future__ import annotations

import click
import os
from pathlib import Path
from typing import Union

import numpy as onp
import jax.numpy as jnp
import jax.random as jr

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from tqdm.auto import trange
import wandb
import matplotlib.pyplot as plt

from dtd.model import DirichletTuckerDecomp

def _get_subshape(X, axes: tuple=(0,)):
    """Return shape of X along specified axes."""
    return tuple([X.shape[i] for i in axes])

def load_data(fpath_list: list[str]):
    """Load data and metadata, and concatenate it along axis=0.
    
    Parameters
        fpath_list: list of file paths to load

    Returns
        data: uint tensor of syllable data per timebin
        ages: list of recording ages, by subject
        names: list of subject names
    """

    counts, ages, names = [], [], []
    for fpath in fpath_list:
        fpath = Path(fpath)

        names.append(fpath.stem[3:])
        with jnp.load(fpath) as f:
            counts.append(f['counts'])
            ages.append(f['session_ids'])
        
    return {
        'counts': jnp.concatenate(counts, axis=0),
        'batch_axes': (0,1),
        'ages': ages, 
        'names': names
    }

def make_random_mask(key, shape, train_frac=0.8):
    """Make binary mask to split data into train (1) and test (0) sets."""
    return jr.bernoulli(key, train_frac, shape)

def fit_model(key, X, mask, k1, k2, k3, alpha=1.1, n_iters=5000):
    d1, d2, d3 = X.shape
    total_counts = X[0,0].sum() # TODO Make a flexible sum, given batch or normalized axes

    # Construct a model
    model = DirichletTuckerDecomp(total_counts, k1, k2, k3, alpha)

    # Randomly initialize parameters
    print("Initializing model...", end="")
    init_params = model.sample_params(key, d1, d2, d3)
    print("Done.")

    # Fit model to data with EM
    print("Fitting model...", end="")
    params, lps = model.fit(X, mask, init_params, n_iters)
    print("Done.")

    # Compute test log likelihood
    print("Computing test log likelihood...", end="")
    test_ll = model.heldout_log_likelihood(X, mask, params)

    # Compute test ll under baseline model (empirical average of syllable usage)
    baseline_probs = jnp.mean(X[mask], axis=0)  # TODO does X[mask] actually still preserve X shape?
    baseline_probs /= baseline_probs.sum(axis=-1, keepdims=True)
    baseline_test_ll = \
        tfd.Multinomial(total_counts, probs=baseline_probs).log_prob(X[~mask]).sum()

    # Compute test ll under under saturated model (true empiral syllable usage)
    saturated_probs = X / X.sum(axis=-1, keepdims=True)
    saturated_test_ll = \
        tfd.Multinomial(total_counts, probs=saturated_probs).log_prob(X[~mask]).sum()

    # Compute test log likelihood percent deviation of fitted model from
    # saturated model (upper bound), relative to baseline (lower bound).
    pct_dev = (test_ll - baseline_test_ll) / (saturated_test_ll - baseline_test_ll)
    print("Done.")

    return model, params, lps, pct_dev

@click.command()
@click.option('--datadir', type=click.Path(exists=True, file_okay=False),
                help='Path to directory of input files.')
@click.option('--k1', type=int, help='number of factors along dimension 1.')
@click.option('--k2', type=int, help='number of factors along dimension 2.')
@click.option('--k3', type=int, help='number of factors along dimension 3.')
@click.option('--seed', default=0, show_default=True, help='random seed for initialization.')
@click.option('--alpha', default=1.1, show_default=True, help='concentration of Dirichlet prior.')
@click.option('--train-frac', default=0.8, show_default=True, help='concentration of Dirichlet prior.')
@click.option('--n-iters', default=5000, show_default=True, help='number of em iterations to run.')
@click.option('--project', default=None, show_default=True, help='WandB project name.')
def run_one(project, datadir, k1, k2, k3, seed, alpha, train_frac=0.8, n_iters=5000):
    """Fit data to one set of model parameters."""
    # k1 = int(k1)
    k2 = int(k2)
    k3 = int(k3)
    seed = int(seed)
    alpha = float(alpha)
    train_frac = float(train_frac)
    n_iters = int(n_iters)

    if isinstance(project, str):
        wandb.init(
            project=project,
            config={
                'k1': k1,
                'k2': k2,
                'k3': k3,
                'alpha': alpha,
                'seed': seed,
            }
        )
    
    # Load data from input directory
    # TODO Currently limit files loaded to debug OOM issue
    fpath_list = sorted([f for f in Path(datadir).iterdir() if f.is_file()])
    data = load_data(fpath_list[:1])

    key = jr.PRNGKey(seed)
    key_mask, key_fit = jr.split(key)

    batch_dims = _get_subshape(data['counts'], data['batch_axes'])

    # Mask out validation data
    mask = make_random_mask(key_mask, batch_dims, train_frac)

    # Fit the model using the random seed provided
    print('data shape', data['counts'].shape)
    counts = jnp.asarray(data['counts'], dtype=jnp.float32)
    del data

    model, params, lps, pct_dev = \
        fit_model(key_fit, counts, mask, k1, k2, k3, alpha, n_iters=n_iters)
    # acc, confusion_matrix = evaluate_prediction(params["Psi"], y)

    if isinstance(project, str):
        for lp in lps:
            wandb.log({'lp': lp}, commit=False)
        wandb.run.summary["pct_dev"] = pct_dev
        wandb.finish()
    return

if __name__ == "__main__":
    run_one()