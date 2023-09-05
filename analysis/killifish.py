from __future__ import annotations

import click
import os
from pathlib import Path
from typing import Union

import numpy as onp
import jax.numpy as jnp
import jax.random as jr
import optax

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from tqdm.auto import trange
import wandb
import matplotlib.pyplot as plt

from dtd.model3d import DirichletTuckerDecomp

PROJECT_NAME = 'kf-dtd'
DEFAULT_LR_SCHEDULE_FN = (
    lambda n_minibatches, n_epochs:
        optax.cosine_decay_schedule(
            init_value=1.,
            alpha=0.,
            decay_steps=n_minibatches*n_epochs,
            exponent=0.8,
        )
)

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
        'X': jnp.concatenate(counts, axis=0),
        'batch_axes': (0,1),
        'ages': ages, 
        'names': names
    }

def make_random_mask(key, shape, train_frac=0.8):
    """Make binary mask to split data into train (1) and test (0) sets."""
    return jr.bernoulli(key, train_frac, shape)

def stochastic_fit_model(key, X, mask, total_counts, k1, k2, k3, alpha=1.1,
                         lr_schedule_fn=None, minibatch_size=20, n_epochs=100):
    """Fit 3D DTD model to data using stochastic fit algoirthm.

    (New) parameters
        lr_schedule_fn: Callable[[int, int], optax.Schedule]
            Given `n_minibatches` and `n_epochs`, returns a function mapping step
            counts to learning rate value.
        minibatch_size: int
            Number of samples per minibatch.
        n_epochs: int
            Number of full passes through the dataset to perform
    """

    key_init, key_fit = jr.split(key)

    # Set default learning rate schedule function if none provided
    lr_schedule_fn = lr_schedule_fn if lr_schedule_fn is not None else DEFAULT_LR_SCHEDULE_FN

    # Construct a model
    model = DirichletTuckerDecomp(total_counts, k1, k2, k3, alpha)

    # Randomly initialize parameters
    print("Initializing model...", end="")
    d1, d2, d3 = X.shape
    init_params = model.sample_params(key_init, d1, d2, d3)
    print("Done.")

    # Fit model to data with EM
    print("Fitting model...", end="")
    params, lps = model.stochastic_fit(X, mask, init_params, n_epochs,
                                       lr_schedule_fn, minibatch_size, key_fit)
    print("Done.")

    return params, lps

def evaluate_fit(params, X, mask, total_counts, k1, k2, k3, alpha):
    """Compute heldout log likelihood and percent deviation from saturated model."""

    # Reonstruct a model 
    model = DirichletTuckerDecomp(total_counts, k1, k2, k3, alpha)
    
    # Compute test log likelihood
    print("Evaluating fit...", end="")
    test_ll = model.heldout_log_likelihood(X, mask, params)

    # Compute test ll under baseline model (empirical average of syllable usage)
    baseline_probs = jnp.mean(X[mask], axis=0)  # TODO does X[mask] actually still preserve X shape? Shouldn't this sum over batch_axes?
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

    return test_ll, pct_dev

@click.command()
@click.argument('datadir', type=click.Path(exists=True, file_okay=False))
@click.argument('k1', type=int,)
@click.argument('k2', type=int,)
@click.argument('k3', type=int,)
@click.option('--alpha', type=float, default=1.1,
              help='Concentration of Dirichlet prior.')
@click.option('--seed',type=int, default=0,
              help='Random seed for initialization.')
@click.option('--train', 'train_frac', type=float, default=0.8,
              help='Fraction of .')
@click.option('--minibatch', 'minibatch_size', type=int, default=20,
              help='Number of samples per minibatch')
@click.option('--epoch', 'n_epochs', type=int, default=5000,
              help='Number of epochs of stochastic EM to run.')
@click.option('--wandb', 'use_wandb', is_flag=True,
              help='Log run with WandB')
def run_one(datadir, k1, k2, k3, seed, alpha, train_frac=0.8,
            minibatch_size=20, n_epochs=5000, use_wandb=False):
    """Fit data to one set of model parameters."""
    
    if use_wandb:
        wandb.init(
            project=PROJECT_NAME,
            config={
                'k1': k1,
                'k2': k2,
                'k3': k3,
                'alpha': alpha,
                'seed': seed,
                'minibatch_size': minibatch_size,
                'n_epochs': n_epochs,
            }
        )
        wandb.define_metric('avg_lp', summary='min')
    
    # ==========================================================================

    key = jr.PRNGKey(seed)
    key_mask, key_fit = jr.split(key)

    # Load data from input directory. Search in all subdirectories
    print("Loading data...",end="")
    fpath_list = sorted([f for f in Path(datadir).rglob("*") if f.is_file()])
    data = load_data(fpath_list)

    # Cast integer counts to float32 dtype
    X = jnp.asarray(data['X'], dtype=jnp.float32)
    print("Done.")
    print(f"\tData array: shape ({X.shape}), {X.nbytes/(1024**3):.1f}GB")

    # Create random mask to hold-out data for validation
    batch_shape = _get_subshape(data['X'], data['batch_axes'])
    mask = make_random_mask(key_mask, batch_shape, train_frac)

    # Get total counts. Since any random batch indices should give the same number
    # counts, i.e. if X has batch_axes of (0,1), then X[i,j].sum() = C for all i,j,
    # and since we assume that the batch axes are the leading dimensions, we can
    # calculate the total counts  of data by abusing data['batch_axes'] as indices
    total_counts = float(X[data['batch_axes']].sum())
    assert total_counts.is_integer(), \
        f'Expected `total_counts` to be integer values, but got {total_counts}'
    total_counts = int(total_counts)
    del data

    # Fit the model
    params, lps = stochastic_fit_model(
        key_fit, X, mask, total_counts, k1, k2, k3, alpha=alpha, n_epochs=n_epochs,
    )

    # Evaluate the model on heldout samples
    test_ll, pct_dev = evaluate_fit(params, X, mask, total_counts, k1, k2, k3, alpha)

    # ==========================================================================
    # Log metrics
    if use_wandb:
        for lp in lps.ravel():
            wandb.log({'avg_lp': lp / mask.sum()}, commit=False)
        wandb.run.summary["pct_dev"] = pct_dev
        wandb.run.summary["avg_test_ll"] = test_ll / (~mask).sum()
        wandb.finish()

    return

if __name__ == "__main__":
    run_one()
