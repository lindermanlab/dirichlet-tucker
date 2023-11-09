from __future__ import annotations

import click
from pathlib import Path

from collections import OrderedDict
import itertools
import numpy as onp
import jax.numpy as jnp
import jax.random as jr
import optax
from datetime import datetime

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

import time
from tqdm.auto import trange
import wandb

import matplotlib.pyplot as plt
import matplotlib.colors as mplc
from scipy.cluster.hierarchy import linkage, leaves_list    # For hierarchically clustering topics
import pandas as pd                                         # For making time-of-day tick labels

from dtd.model3d import DirichletTuckerDecomp

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

# Syllable permutations, based on KL-divergence of syllable parameters
# - SYLLABLE_PERM_DICT: Consists of (cluster_name: indices) items and is used
#                       for annotating figures (see `set_syllable_cluster_ticks`)
# - SYLLABLE_PERM: Used for permutating syllable parameters
SYLLABLE_PERM_DICT = OrderedDict([
    ('inactive', [91, 97, 30, 10, 1, 43, 69] + [33,] + [99,]),                   # 99: Belly
    ('pause and drift', [32, 51, 82, 0, 56, 14, 19, 84, 92, 7, 66, 3, 17, 57, 85, 46, 27, 65]),
    ('straight swim', [9, 31, 28, 80, 68, 73, 6, 54, 58, 70, 39, 40, 25, 42, 89, 45, 77, 24, 60, 74, 94]),
    ('edge+sidebody', [63, 72, 15, 98, 23, 96, 36, 83, 55, 90, 76, 61, 88] + [50,]),
    ('j-turn+reverse', [44, 75, 59, 95, 48, 78, 18, 16, 35, 47],),
    ('aggression+glass surf', [71, 11, 64, 81, 87, 20, 21, 37, 8, 41, 52, 26, 93, 38, 12, 22, 5]),
    ('nose down', [86, 49, 67, 79, 2, 13, 62, 29, 34, 4, 53]),
])
SYLLABLE_PERM = list(itertools.chain.from_iterable(SYLLABLE_PERM_DICT.values()))

# ============================================================================ #
#                                 VISUALIZATION                                #
# ============================================================================ #

def get_unique_path(fpath, fmt='02d'):
    i = 1
    while fpath.is_file():
        stem = '-'.join(fpath.stem.split('-')[:-1])
        fpath = fpath.parent / f'{stem}-{i:{fmt}}{fpath.suffix}'
        i += 1
    return fpath

def set_syllable_cluster_ticks(ax=None, axis='x', font_kws={'fontsize': 'small'}):
    """Label specified axis with syllable _cluster_ names."""

    if ax is None:
        ax = plt.gca()

    cluster_names = list(SYLLABLE_PERM_DICT.keys())
    cluster_sizes = [len(v) for v in SYLLABLE_PERM_DICT.values()]
    
    maj_ticks = onp.cumsum([0,] + cluster_sizes) - 0.5
    min_ticks = onp.diff(maj_ticks) / 2 + maj_ticks[:-1]

    if axis == 'x':
        # Draw major ticks
        ax.set_xticks(maj_ticks)
        ax.tick_params(axis='x', which='major', length=5, labelbottom=False,)
        ax.grid(visible=True, which='major', axis='x', alpha=0.2, lw=0.5)

        # Annotate between the ticks
        ax.set_xticks(min_ticks, cluster_names, minor=True, **font_kws)
        ax.tick_params(axis='x', which='minor', bottom=False, labelbottom=True)
    else:
        # Draw major ticks
        ax.set_yticks(maj_ticks)
        ax.tick_params(axis='y', which='major', length=5, labelleft=False,)
        ax.grid(visible=True, axis='y', which='major', alpha=0.2, lw=0.5)

        # Annotate between the ticks
        ax.set_yticks(min_ticks, cluster_names, minor=True, **font_kws)
        ax.tick_params(axis='y', which='minor', left=False, labelleft=True)
    
    return ax

def draw_syllable_factors(params):
    # Permute syllables to match our KL-clustering for better interpretability
    syllable_factors = params[3][:,SYLLABLE_PERM]
    K, D = syllable_factors.shape

    # Use hiearchical clustering on syllable factor ("behavioral topic") axis
    method = 'centroid'
    metric = 'euclidean'
    topic_perm = leaves_list(linkage(syllable_factors, method, metric)).astype(int)

    syllable_factors = syllable_factors[topic_perm,:]

    # ------------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 4.5), dpi=96)
    ax = plt.gca()
    im = ax.imshow(syllable_factors, interpolation='none', aspect='auto',
                   cmap='magma', norm=mplc.LogNorm(0.5/D, 1.0))
    set_syllable_cluster_ticks(ax)
    plt.colorbar(im, ax=ax, extend='min')

    # Visually demarcate each behavioral topic
    ax.set_yticks(onp.arange(K)-0.5, [])
    ax.grid(visible=True, which='major', axis='y', alpha=0.8, lw=0.5)
    ax.set_ylabel('syllable factors / "behavioral topics"')

    return fig

def make_tod_series(freq):
    """Make time-of-day datetime.time points spaced at the given frequency.

    If `resample_rule='30min'`, then constructs returns array consisting of
        [(00:00:00), (00:30:00), ..., (23:00:00), (23:30:00)]

    Parameters
        freq: Pandas timedelta offset or offset string alias. See:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    """

    last_label = (
        (pd.to_datetime("00:00:00") - pd.Timedelta(freq)).to_pydatetime()
                                                         .time()
                                                         .strftime(format='%H:%M:%S')
    )

    return pd.date_range('00:00:00', last_label, freq=freq).time

def draw_circadian_bases(params):
    circadian_bases = params[2]
    D, K = circadian_bases.shape

    # Permute the circadian bases so that they are sorted by earliest peak
    t_peak = onp.argmax(circadian_bases, axis=0)
    basis_perm = onp.argsort(t_peak, kind='stable')
    circadian_bases = circadian_bases[:, basis_perm]

    # Share a common a y-axis
    ymax = circadian_bases.max()

    # ------------------------------------------------------------------------
    fig, axs = plt.subplots(nrows=K, ncols=1, squeeze=True,
                            gridspec_kw={'hspace':0.1}, figsize=(8,9), dpi=96)
    
    for k, ax in enumerate(axs):
        # Plot basis, and adjust x-axis days with human-interpretable times
        ax.plot(circadian_bases[:,k])
    
        # Grey out background if factor L2 norm is below a threshold
        mag = onp.linalg.norm(circadian_bases[:,k])
        if mag <= 0.3:
            ax.set_facecolor('0.8')
            ax.annotate(f'|factor|={mag:.2f}', (0.01,0.9), xycoords='axes fraction',
                        va='top', fontsize='small')
        
        # Label x-axis with time-of-day from 0H - 24H, every 2H
        t_dts = make_tod_series('2H')        
        t_locs = onp.concatenate([onp.linspace(0, D, num=len(t_dts), endpoint=False), [D]])
        t_labels = list(map(lambda dt: dt.strftime('%H'), t_dts)) + ['24']
        
        ax.set_xticks(t_locs)
        ax.set_xticklabels(t_labels)

        # Label y-axis with "circadian bases"
        if k == K // 2:
            ax.set_ylabel('time-of-day factors / "circadian bases"')

        # Set axis limits; reduce blank space margins
        ax.set_ylim(bottom=0, top=ymax)
        ax.margins(x=0.01, y=0.5)

        # Draw time-of-day ticks; only annotate bottom-most subplot
        ax.tick_params(labelleft=False, labelbottom=False)
        if k == K-1:
            ax.tick_params(labelbottom=True)
            ax.set_xlabel('time of day [24hr]')
        
        # Despine
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    return fig

# ============================================================================ #
#                               FIT & EVALUATION                               #
# ============================================================================ #

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

    return params, lps

def evaluate_fit(params, X, mask, total_counts, k1, k2, k3, alpha):
    """Compute heldout log likelihood and percent deviation from saturated model."""

    # Reonstruct a model 
    model = DirichletTuckerDecomp(total_counts, k1, k2, k3, alpha)
    
    # Compute test log likelihood
    print("Evaluating fit...", end="")
    test_ll = model.heldout_log_likelihood(X, mask, params)

    # Compute test ll under baseline model (average syllable usage across all held-in samples)
    # - baseline_probs: (D3,)
    baseline_probs = jnp.mean(X[mask], axis=0)
    baseline_probs /= baseline_probs.sum(axis=-1, keepdims=True)
    baseline_test_ll = \
        tfd.Multinomial(total_counts, probs=baseline_probs).log_prob(X[~mask]).sum()

    # Compute test ll under under saturated model (true empirical syllable usage)
    # - saturated_probs = X / total_counts: (D1, D2, D3)
    saturated_probs = X / X.sum(axis=-1, keepdims=True)
    saturated_test_ll = \
        jnp.where(~mask, tfd.Multinomial(total_counts, probs=saturated_probs).log_prob(X), 0.0).sum()

    # Compute test log likelihood percent deviation of fitted model from
    # saturated model (upper bound), relative to baseline (lower bound).
    pct_dev = (test_ll - baseline_test_ll) / (saturated_test_ll - baseline_test_ll)
    pct_dev *= 100
    print("Done.")

    return test_ll, pct_dev

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
    else:
        wnb = None
    
    # ==========================================================================

    run_start_time = time.time()

    key = jr.PRNGKey(seed)
    key_mask, key_fit = jr.split(key)

    # Load data from input directory. Search in all subdirectories
    print("Loading data...",end="")
    fpath_list = sorted([f for f in Path(datadir).rglob("*") if f.is_file()])
    data = load_data(fpath_list)
    
    # Cast integer counts to float32 dtype
    X = jnp.asarray(data['X'], dtype=jnp.float32)[:max_samples]
    print("Done.")
    print(f"\tData array: shape {X.shape}, {X.nbytes/(1024**3):.1f}GB")

    # Create random mask to hold-out data for validation
    batch_shape = _get_subshape(X, data['batch_axes'])
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
    params, lps = fit_model(
            method, key_fit, X, mask, total_counts, k1, k2, k3, alpha=alpha,
            minibatch_size=minibatch_size, n_epochs=n_epochs, drop_last=drop_last, wnb=wnb)

    # Evaluate the model on heldout samples
    test_ll, pct_dev = evaluate_fit(params, X, mask, total_counts, k1, k2, k3, alpha)

    run_elapsed_time = time.time() - run_start_time

    # Visualize parameters
    fig_topics = draw_syllable_factors(params)
    fig_bases = draw_circadian_bases(params)

    # ==========================================================================
    
    # Save results locally
    fpath_params = outdir/'params.npz'
    onp.savez_compressed(fpath_params, G=params[0], F1=params[1], F2=params[2], F3=params[3])
    
    plt.figure(fig_topics)
    fpath_topics = get_unique_path(outdir/'behavioral-topics.png')
    plt.savefig(fpath_topics, bbox_inches='tight')
    plt.close()

    plt.figure(fig_bases)
    fpath_bases = get_unique_path(outdir/'circadian-bases.png')
    plt.savefig(fpath_bases, bbox_inches='tight')
    plt.close()

    # Log summry metrics. lps are logged by model.stochastic_fit
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
