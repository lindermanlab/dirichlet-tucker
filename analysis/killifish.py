from __future__ import annotations
from typing import Optional

import wandb
import click
from pathlib import Path
from datetime import datetime
import time

import numpy as onp
import jax.numpy as jnp
import jax.random as jr
import optax

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

import matplotlib.pyplot as plt

from dtd.model3d import DirichletTuckerDecomp
from kf_viz import draw_syllable_factors, draw_circadian_bases

PROJECT_NAME = 'kf-dtd-231022'
DEFAULT_LR_SCHEDULE_FN = (
    lambda n_minibatches, n_epochs:
        optax.cosine_decay_schedule(
            init_value=1.,
            alpha=0.,
            decay_steps=n_minibatches*n_epochs,
            exponent=0.8,
        )
)

def get_unique_path(fpath, fmt='02d'):
    i = 1
    while fpath.is_file():
        stem = '-'.join(fpath.stem.split('-')[:-1])
        fpath = fpath.parent / f'{stem}-{i:{fmt}}{fpath.suffix}'
        i += 1
    return fpath

# ============================================================================ #
#                               FIT & EVALUATION                               #
# ============================================================================ #

def make_random_mask(key, shape, train_frac=0.8):
    """Make binary mask to split data into train (1) and test (0) sets."""
    return jr.bernoulli(key, train_frac, shape)

def load_data(data_dir: Path,
              max_samples: int=-1,
              train_frac: float=0.8,
              key: Optional[jr.PRNGKey]=None):
    """Load data tensor 
    
    Load data and metadata, and concatenate it along axis=0.
    
    Parameters
        data_dir: Directory containining .npz files of count tensors
        max_samples: Maximum number of total days/sessions of data to load.
            Loaded data is truncated; used for limiting dataset size for debugging.
        train_frac: Fraction of data (batch dims) to use for training.
            Valid PRNGKey must be passed in; else, it is ignored
        key: PRGKey for generating train/test mask. If None, no data is held-out.

    Returns
        total_counts: int. Normalized count value of data tensor
        X: UInt[Array, "n_samples n_bins n_syllables"]
        mask: Int[Array, "n_samples n_bins"]
        ages: Int[Array, "n_samples"]
        names: Object[Array, "n_samples"] 
    """
    # These are data attributes that we would like to pass into the model.
    # Currently, this is not fully supported, so we hardcode them here.
    batch_axes, event_axes = (0,1), (2,)

    # Load data tensor and its metadata from all files in data_dir, concatenate
    #   X: UInt array, (n_samples, n_bins, n_syllables).
    #   ages: list, (n_samples,). Age/days since hatch.
    #   names: list, (n_samples,). Subject names, in long form.
    print("Loading data...",end="")
    fpath_list = sorted([f for f in Path(data_dir).rglob("*") if f.is_file()])

    X, ages, names = [], [], []
    for fpath in fpath_list:
        fpath = Path(fpath)

        with jnp.load(fpath) as f:
            X.append(f['counts'])
            ages.append(f['session_ids'])
            names.append([fpath.stem[3:],]*len(ages[-1]))
    
    X = jnp.concatenate(X, axis=0)[:max_samples]
    ages = onp.concatenate(ages)[:max_samples]
    names = onp.concatenate(names)[:max_samples]
    print("Done.")

    # Create a random mask along batch dimensions (non-normalized dimensions)
    batch_shape = tuple([X.shape[i] for i in batch_axes])
    mask = (make_random_mask(key, batch_shape, train_frac)
            if key is not None else jnp.ones(batch_shape, dtype=int))
    
    # Get total counts. Event axes are, by dimension, the axes along which
    # the data tensor X sums to a constant number.
    total_counts = X.sum(axis=event_axes)
    assert jnp.all(total_counts==total_counts.reshape(-1)[0]), \
        f'Expected data tensor to have a normalized number of counts along the event dimensions {event_axes}.'
    total_counts = int(total_counts.reshape(-1)[0])

    return total_counts, X, mask, ages, names

def fit_model(method, key, X, mask, total_counts, k1, k2, k3, alpha=1.1,
              lr_schedule_fn=None, minibatch_size=1024, n_epochs=100, drop_last=False,
              wnb=None):
    """Fit 3D DTD model to data using stochastic fit algoirthm.

    (New) parameters
        lr_schedule_fn: Callable[[int, int], optax.Schedule]
            Given `n_minibatches` and `n_epochs`, returns a function mapping step
            counts to learning rate value.
        minibatch_size: int
            Number of samples per minibatch.
        n_epochs: int
            Number of full passes through the dataset to perform
        wnb: wandb.Run or None
            WandB Run instance for logging metrics per epoch
    """

    key_init, key_fit = jr.split(key)

    # Construct a model
    model = DirichletTuckerDecomp(total_counts, k1, k2, k3, alpha)

    # Randomly initialize parameters
    print("Initializing model...", end="")
    d1, d2, d3 = X.shape
    init_params = model.sample_params(key_init, d1, d2, d3)
    print("Done.")

    # Fit model to data with EM
    print(f"Fitting model with {method} EM...", end="")
    if method == 'stochastic':
        # Set default learning rate schedule function if none provided
        lr_schedule_fn = lr_schedule_fn if lr_schedule_fn is not None else DEFAULT_LR_SCHEDULE_FN

        params, lps = model.stochastic_fit(X, mask, init_params, n_epochs,
                                           lr_schedule_fn, minibatch_size, key_fit,
                                           drop_last=drop_last, wnb=wnb)
    else:
        params, lps = model.fit(X, mask, init_params, n_epochs, wnb=wnb)

    print("Done.")

    return model, params, lps

def evaluate_fit(model, X, mask, params, ):
    """Compute heldout log likelihood and percent deviation from saturated model."""

    # Compute test log likelihood
    print("Evaluating fit...", end="")
    test_ll = model.heldout_log_likelihood(X, mask, params)

    # Compute test ll under baseline model (average syllable usage across all held-in samples)
    # - baseline_probs: (D3,)
    baseline_probs = jnp.mean(X[mask], axis=0)
    baseline_probs /= baseline_probs.sum(axis=-1, keepdims=True)
    baseline_test_ll = \
        tfd.Multinomial(model.C, probs=baseline_probs).log_prob(X[~mask]).sum()

    # Compute test ll under under saturated model (true empirical syllable usage)
    # - saturated_probs = X / total_counts: (D1, D2, D3)
    saturated_probs = X / X.sum(axis=-1, keepdims=True)
    saturated_test_ll = \
        jnp.where(~mask, tfd.Multinomial(model.C, probs=saturated_probs).log_prob(X), 0.0).sum()

    # Compute test log likelihood percent deviation of fitted model from
    # saturated model (upper bound), relative to baseline (lower bound).
    pct_dev = (test_ll - baseline_test_ll) / (saturated_test_ll - baseline_test_ll)
    pct_dev *= 100
    print("Done.")

    return pct_dev, test_ll, baseline_test_ll, saturated_test_ll

@click.command()
@click.argument('datadir', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--k1', type=int,)
@click.option('--k2', type=int,)
@click.option('--k3', type=int,)
@click.option('--alpha', type=float, default=1.1,
              help='Concentration of Dirichlet prior.')
@click.option('--seed',type=int, default=None,
              help='Random seed for initialization.')
@click.option('--train_frac', type=float, default=0.8,
              help='Fraction of .')
@click.option('--method', type=str, default='full',
              help="'full' for full-batch em, 'stochastic' for stochastic em" )
@click.option('--epoch', 'n_epochs', type=int, default=5000,
              help='# iterations of full-batch EM to run/# epochs of stochastic EM to run.')
@click.option('--minibatch_size', type=int, default=1024,
              help='# samples per minibatch, if METHOD=stochastic.')
@click.option('--max_samples', type=int, default=-1,
              help='Maximum number of samples to load, performed by truncating first mode of dataset. Default: [-1], load all.')
@click.option('--drop_last', is_flag=True,
              help='Do not process incomplete minibatch when using stochasitc EM. Useful for OOM and speed issues.')
@click.option('--wandb', 'use_wandb', is_flag=True,
              help='Log run with WandB')
@click.option('--outdir', type=click.Path(file_okay=False, resolve_path=True, path_type=Path),
              default='./', help='Local directory to save results and wandb logs to.')
def run_one(datadir, k1, k2, k3, seed, alpha, train_frac=0.8,
            method='full', minibatch_size=1024, n_epochs=5000, max_samples=-1,
            drop_last=False, use_wandb=False, outdir=None):
    """Fit data to one set of model parameters."""
    
    print(f"Loading data from...{str(datadir)}")
    print(f"Saving results to...{str(outdir)}")

    # If no seed provided, generate a random one based on the timestamp
    if seed is None:
        seed = int(datetime.now().timestamp())

    if use_wandb:
        wnb = wandb.init(
            project=PROJECT_NAME,
            config={
                'k1': k1,
                'k2': k2,
                'k3': k3,
                'alpha': alpha,
                'seed': seed,
                'method': method,
                'minibatch_size': minibatch_size,
                'max_samples': max_samples
            }
        )
        wandb.define_metric('avg_lp', summary='min')
        
        # Create a subdirectory at outdir based on run_id hash
        run_id = wandb.run.id
        outdir = outdir/run_id
        outdir.mkdir(parents=True)
    else:
        wnb = None
        run_id = None
    
    # ==========================================================================

    run_start_time = time.time()

    key = jr.PRNGKey(seed)
    key_mask, key_fit = jr.split(key)

    total_counts, X, mask, _, _ \
        = load_data(datadir, max_samples=max_samples, train_frac=train_frac, key=key_mask)
    
    # Convert UInt data tensor to float32 dtype
    X = jnp.asarray(X, dtype=jnp.float32)
    print(f"\tData array: shape {X.shape}, {X.nbytes/(1024**3):.1f}GB")

    # Fit the model
    model, params, lps = fit_model(
            method, key_fit, X, mask, total_counts, k1, k2, k3, alpha=alpha,
            minibatch_size=minibatch_size, n_epochs=n_epochs, drop_last=drop_last,
            wnb=wnb)

    # Evaluate the model on heldout samples
    pct_dev, test_ll, baseline_test_ll, saturated_test_ll \
                                        = evaluate_fit(model, X, mask, params)
    
    # Sort the model by principal factors of variation
    params, lofo_test_lls = model.sort_params(X, mask, params)

    run_elapsed_time = time.time() - run_start_time

    # ==========================================================================
    
    # Save results locally
    fpath_params = outdir/'params.npz'
    n_train = mask.sum()
    n_test = (~mask).sum()
    onp.savez_compressed(fpath_params,
                         G=params[0],
                         F1=params[1],
                         F2=params[2],
                         F3=params[3],
                         seed=seed,
                         avg_train_lps=lps/n_train,
                         avg_test_ll=test_ll/n_test,
                         avg_baseline_test_ll=baseline_test_ll/n_test,
                         avg_saturated_test_ll=saturated_test_ll/n_test,
                         avg_lofo_test_ll_1=lofo_test_lls[0]/n_test,
                         avg_lofo_test_ll_2=lofo_test_lls[1]/n_test,
                         avg_lofo_test_ll_3=lofo_test_lls[2]/n_test,
                         run_id=run_id,
                         )
    
    # Visualize parameters and save them
    fig_topics = plt.figure(figsize=(16, 4.5), dpi=96)
    draw_syllable_factors(params, ax=plt.gca())
    fpath_topics = get_unique_path(outdir/'behavioral-topics.png')
    plt.savefig(fpath_topics, bbox_inches='tight')
    plt.close()

    fig_bases = draw_circadian_bases(params)
    fpath_bases = get_unique_path(outdir/'circadian-bases.png')
    plt.savefig(fpath_bases, bbox_inches='tight')
    plt.close()

    # Log summry metrics. lps are automatically logged by model.stochastic_fit
    if use_wandb:
        wnb.summary["pct_dev"] = pct_dev
        wnb.summary["avg_test_ll"] = test_ll / (~mask).sum()
        wnb.summary['total_time [min]'] = run_elapsed_time/60

        wandb.save(str(fpath_topics), policy='now')
        wandb.save(str(fpath_bases), policy='now') 
        wandb.save(str(fpath_params), policy='now')
        wandb.finish()

    return

if __name__ == "__main__":
    run_one()
