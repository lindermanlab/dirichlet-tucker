"""Cross-validate model parameters on held-out reconstruction error.

Some parameters of this file need to be manually edited.
"""
from typing import Optional
from jax import Array

import argparse
from pathlib import Path
import itertools
from math import prod
import time
import wandb
from tqdm import tqdm

import jax.numpy as jnp
import jax.random as jr
import numpy as onp
from sklearn.model_selection import (
    BaseCrossValidator,
    StratifiedKFold
)

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from dtd.data import load_wiltschko22_data
from dtd.model3d import DirichletTuckerDecomp
from dtd.utils import (
    create_block_speckled_mask,
    get_jax_rng_state,
    get_numpy_rng_state,
)

from dtd.viz import (
    draw_drug_class_boxes,
    plot_buffered_mask,
    set_syllable_cluster_ticks,
    set_time_within_session_ticks,
)

PathLike = Path | str

# Global variables for visualization, set when loading data in __main__
session_drug_class = None
frames_per_bin = None
frames_per_sec = None
syllable_cluster_names = None
tick_period = 10
tick_units = 'min'

def make_summary_figure(
    params, avg_train_lps, holdout_mask, buffer_mask, figsize=(8.5,11), dpi=72
):

    interaction_tensor, session_factors, temporal_factors, syllable_factors = params
    dynamic_topics = jnp.einsum(
        "ijk,tj,kv->itv", interaction_tensor, temporal_factors, syllable_factors
    )

    k1, k2, k3 = interaction_tensor.shape
    d1, d2, d3 = len(session_factors), len(temporal_factors), syllable_factors.shape[-1]

    fig = plt.figure(figsize=(8.5, 11), dpi=72)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    
    subfigs = fig.subfigures(nrows=4, height_ratios=[1,1,1,3], hspace=0.4)

    # ===========================================
    #       Mask             |  Train log prob
    # ===========================================
    mask_subfig, lp_subfig = subfigs[0].subfigures(
        ncols=2, width_ratios=[4,1], wspace=0.35
    )

    # ---------------------------------------------
    _, axs = plot_buffered_mask(
        holdout_mask, buffer_mask,
        max_sessions_per_col=100, cax_width_ratio = 0.1,
        fig=mask_subfig,
    )

    # annotate x-axes
    for ax in axs[:-1]:
        set_time_within_session_ticks(
            d2, frames_per_bin, frames_per_sec, tick_period, tick_units,
            axis='x', ax=ax,
        )

    plt.annotate(
        f"time within session [{tick_units}]",
        (0.5, -0.15), xycoords=mask_subfig.transSubfigure,
        ha='center', va='top',
    )

    axs[0].set_ylabel('session index')

    mask_subfig.suptitle("train/test mask", fontsize='x-large', y=1.2, va='bottom')

    # ---------------------------------------------
    ax = lp_subfig.subplots()
    ax.plot(avg_train_lps)
    ax.set_xlabel('iterations')
    ax.set_ylabel('log prob')
    ax.set_title('avg training joint log prob')
    sns.despine(ax=ax)

    # ==========================================
    #   Syllable factors  |   Temporal factors
    # ==========================================
    topic_subfig, temp_subfig = subfigs[1].subfigures(ncols=2, wspace=0.2)

    # ---------------------------------------
    # Syllable factors aka behavioral topics
    # ---------------------------------------
    ax = topic_subfig.add_subplot(1,1,1)

    vmax = syllable_factors.max()
    im = ax.imshow(
        syllable_factors, cmap='magma', clim=(0,vmax), interpolation='none', aspect='auto',
    )
    plt.colorbar(im, extend='max', label='syllable probability')

    # Label x-axis
    set_syllable_cluster_ticks(syllable_cluster_names, ax=ax)
    ax.set_xlabel("syllable ids")

    # Label y-axis
    ax.set_yticklabels([])
    ax.set_ylabel("topics")

    # Draw grid lines to indicate what direction sums to 1
    # ax.set_yticks(onp.arange(k3) + 0.5, minor=True)
    # ax.tick_params(axis='y', which='minor', left=False)  # hide minor ticks
    # ax.grid(axis='y', which='minor', color='white')  # draw seperating lines

    ax.set_title('topic-syllable distributions')

    # ---------------------------------------
    # Temporal factors
    # ---------------------------------------
    ax = temp_subfig.subplots()

    im = ax.imshow(
        temporal_factors.T,
        cmap='magma', clim=(0,1), interpolation='none', aspect='auto',
    )
    plt.colorbar(im, label='factor probability')

    # Label x-axis
    set_time_within_session_ticks(
        d2, frames_per_bin, frames_per_sec, tick_period, tick_units=tick_units, ax=ax
    )
    ax.set_xlabel(f"time within session [{tick_units}]")

    # Label y-axis
    ax.set_yticklabels([])
    ax.set_ylabel("temporal factors")

    # Draw grid lines to indicate which axis sums to 1
    ax.set_xticks(onp.arange(d2) + 0.5, minor=True)
    ax.tick_params(axis='x', which='minor', left=False)  # hide minor ticks
    ax.grid(axis='x', which='both', color='white')  # draw seperating lines

    ax.set_title('temporal factor contribution per time bin')

    # ======================================
    #      Interaction tensor slices
    # ======================================
    interaction_subfig = subfigs[2]
    interaction_subfig.subplots_adjust(left=0, right=1)
    axs = interaction_subfig.subplots(
        ncols=k1+1,  # Extra axis is for colorbar
        width_ratios=(1,)*k1 + (0.1,),
        gridspec_kw=dict(wspace=0.1),
    )
    axs, cax = axs[:-1], axs[-1]

    for i_session_factor, ax in enumerate(axs):
        im = ax.imshow(
            interaction_tensor[i_session_factor].T,
            cmap='magma', clim=(0,1), interpolation='none', aspect='auto',
        )

        # Label x-axis
        ax.set_xticks(onp.arange(k2))
        ax.set_xticklabels([])
        ax.tick_params(axis='x', which='minor', bottom=False)
        # ax.set_xlabel("temporal factors")

        # Label y-axis
        ax.set_yticks(onp.arange(k3))
        ax.set_yticklabels([])

        # Draw grid lines to indicate which axis sums to 1
        # ax.set_xticks(onp.arange(k2) + 0.5, minor=True)
        # ax.tick_params(axis='x', which='minor', left=False)  # hide minor ticks
        # ax.grid(axis='x', which='minor', color='white')  # draw seperating lines

        ax.set_title(f"i={i_session_factor}", fontsize='small', y=0.96)

    plt.annotate("temporal factors",
        (0.5, -0.15), xycoords=interaction_subfig.transSubfigure,
        ha='center',
    )
    axs[0].set_ylabel('topics')

    # Add colorbar to last axis
    plt.colorbar(im, cax=cax, label='factor probability')

    interaction_subfig.suptitle("interaction tensor slices", y=1.15, fontsize='x-large', va='bottom')

    # ==========================================================
    # Session factor weights |       Dynamic topic factors
    # ==========================================================

    weights_subfig, factors_subfig = subfigs[-1].subfigures(
        ncols=2, wspace=0.25, width_ratios=[1, k1]
    )

    # ---------------------
    # Plot session factors
    # ---------------------
    ax = weights_subfig.subplots()
    ax.imshow(
        session_factors,
        cmap='magma', clim=(0,1), interpolation='none', aspect='auto',
    )

    # Label axes
    ax.set_xticks(onp.arange(k1))
    # ax.set_xticklabels([])
    ax.set_xlabel('session factors')

    ax.set_yticks([])
    draw_drug_class_boxes(session_drug_class, ax=ax)
    ax.set_ylabel('sessions (drug x dose)', labelpad=75)

    weights_subfig.suptitle('session factor weights', y=1.1, fontsize='x-large')

    # ---------------------
    # Plot dynamic_topics
    # ---------------------
    axs = factors_subfig.subplots(
        ncols=k1+1,  # add 1 for cax
        width_ratios=(1,)*k1 + (0.1,),
        gridspec_kw=dict(wspace=0.1),
    )
    axs, cax = axs[:-1], axs[-1]

    vmax = dynamic_topics.max()
    for i_session_factor, ax in enumerate(axs):
        im = ax.imshow(
            dynamic_topics[i_session_factor].T,
            cmap='magma', clim=(0,vmax),
            interpolation='none', aspect='auto',
        )

        # Set within-session time ticks
        set_time_within_session_ticks(
            d2, frames_per_bin, frames_per_sec,
            tick_period=tick_period, tick_units=tick_units, ax=ax
        )

        # Remove syllable axis ticks
        ax.set_yticks([])
        set_syllable_cluster_ticks(
            syllable_cluster_names,
            label=(True if i_session_factor==0 else False),
            ax=ax, axis='y',
        )

        ax.set_title(f"i={i_session_factor}")

    plt.colorbar(im, cax=cax, extend='max')

    plt.annotate(
        f"time within session [{tick_units}]",
        (0.5, -0.07), xycoords=factors_subfig.transSubfigure, ha='center'
    )
    axs[0].set_ylabel('syllable ids')

    factors_subfig.suptitle('dynamic topic factors', y=1.1, fontsize='x-large')

    return fig


def train_and_evaluate_one(
    key: Array,
    k1: int,
    k2: int,
    k3: int,
    X: Array,
    *,
    holdout_mask: Array,
    buffer_mask: Array,
    n_epochs: int=500,
    wnb=None,
):
    """Train and evaluate a single model.
    
    Logs results online to WandB and locally.

    Parameters
    ----------
    key (jax.Array): PRNG key for initializing model parameters.
    k1, k2, k3 (int): Model rank.
    X (jax.Array): shape (d1, d2, d3), float dtype.
    holdout_mask (jax.Array): shape (d1, d2), bool dtype.
        Data entries to withhold from training and use from evaluation.
    buffer_mask (jax.Array): shape (d1, d2), bool dtype.
        Data entries to withhold from training and evaluation.
    n_epochs (int). Number of epochs to run EM algorithm.
    wnb (wandb.Run). WandB Run instance to which to log results.
        Default: None, no logging performed.
    
    Returns
    -------
    params (tuple of Arrays): Fitted parameters, consisting of interaction tensor and
        factors for each mode.
    avg_train_lps (Array): shape (n_epochs, ), float dtype
        Joint log probability on held-in entries, averaged over number of entries and
        event size.
    avg_test_ll (float): Log likelihood of held-out entries, averaged over number of
        entries and event size.
    run_elapsed_time (float): Elapsed time for fit.
    
    """
    holdout_mask = jnp.asarray(holdout_mask, dtype=bool)
    buffer_mask = jnp.asarray(buffer_mask, dtype=bool)

    run_start_time = time.time()

    # Instantiate model and initialize parameters
    d1, d2, d3 = X.shape
    
    event_ndims = X.ndim - holdout_mask.ndim
    event_axes = tuple([i for i in range(X.ndim)[-event_ndims:]])
    total_counts = int((X.sum(event_axes).ravel())[0])  # Assumes that all bins sum to exact same count
    
    model = DirichletTuckerDecomp(total_counts, k1, k2, k3, alpha=1.1)
    init_params = model.sample_params(key, d1, d2, d3, conc=0.5)

    # Fit model to held-in data
    train_mask = ~(holdout_mask | buffer_mask)
    params, train_lps = model.fit(X, train_mask, init_params, n_epochs, wnb=wnb)
    avg_train_lps = train_lps / (train_mask.sum() * d3)

    # Evaluate fitted parameters on held-out data
    test_mask = ~(holdout_mask)  # model.heldout_log_likelihood inverts this
    test_ll = model.heldout_log_likelihood(X, test_mask, params)
    avg_test_ll = test_ll / ((~test_mask).sum() * d3)

    run_elapsed_time = time.time() - run_start_time

    return params, avg_train_lps, avg_test_ll, run_elapsed_time

def train_and_eval_and_log_nfolds(
    X: Array,
    event_ndims: int,
    *,
    k1: int, 
    k2: int,
    k3: int,
    mask_frac: float=0.2,
    mask_block_shape: tuple,
    mask_buffer_size: tuple,
    n_folds: int=10,
    n_epochs: int=500,
    wandb_project: str,
    output_dir: PathLike,
    seed: Optional[int]=None,
    wandb_debug: bool=False,
):
    """Train and evaluate a given model over multiple folds

    Loads data, constructs masks, then calls `train_and_eval_one` for each fold.

    Note: Speckled block masking preclude "proper" k-folds cross-validation.

    Parameters
    ----------
    X (jax.Array): shape (d1, d2, d3). Count data.
    event_ndims (int): Number of event dimensions, expected to be exactly 1.
    k1, k2, k3 (int): Model rank.
    mask_frac (float): Fraction of batched data to hold-out for evaluation.
        Does _not_ include fraction of data held-out for buffer.
    mask_block_shape (tuple): Length of mask for each batch mode.
    mask_buffer_size (tuple): len (batch_ndims,). Number of entries to buffer on _either_
        side of mask block, for each batch mode. Default: 0.2.
    n_folds (int): Number of folds to run. Since speckled block masking makes it difficult
        to cleanly split data between folds, it is recommended to set number of folds to
        a value _greater than_ 1/mask_frac. Default: 10.
    n_epochs (int): Number of epochs to run (full-batch) EM algorithm. Default: 500
    wandb_project (str): WandB project name in which to save run.
    output_dir (PathLike): Local base directory to which to save outputs (params and figs).
        Outpus are organized in `<output_dir>/<wandb_project>/<wandb_run_id>`,
        where `<wandb_run_id>` is a unique hash generated by WandB.
    seed (int): Used to seed jax and numpy RNGs. If None, set seed based on current time.
    wandb_debug (bool): If True, set wandb run display name to DEBUG
    """

    assert event_ndims==1, "DirichletTuckerDecomp3D currently only handles 1-dimensional events."
    
    output_dir = Path(output_dir)

    if seed is None:
        seed = time.time_ns()
    onp_rng = onp.random.default_rng(seed=seed)  # Used by `create_block_speckled_mask`
    jax_rng = jr.key(seed)

    # ===================================================================================
    # Load data
    # ===================================================================================
    d1, d2, d3 = n_sessions, n_timebins, n_syllables = X.shape
    batch_shape = X.shape[:-event_ndims]

    # ===================================================================================
    # Run folds
    # ===================================================================================
    pbar = tqdm(range(n_folds), desc="Crossval")
    for i_fold in pbar:
        # Numpy RNG will automatically update.
        this_jax_key = jr.fold_in(jax_rng, i_fold)

        # ---------------------------------------------------------------------
        # Setup WandB and local logging
        # ---------------------------------------------------------------------
        project_dir = output_dir/wandb_project/"wandb"
        wnb = wandb.init(
            project=wandb_project,
            name="DEBUG" if wandb_debug else None,
            config=dict(
                k1=k1, k2=k2, k3=k3,
                mask_frac=mask_frac,
                mask_block_shape=mask_block_shape,
                mask_buffer_size=mask_buffer_size,
                n_folds=n_folds,
                n_epochs=n_epochs,
                onp_rng_state=get_numpy_rng_state(onp_rng),
                jax_rng_state=get_jax_rng_state(this_jax_key),
            ),
            resume="allow",
            dir=str(project_dir),  # absolute path to directory with experimental logs
        )
        wandb.define_metric('avg_lp', summary='min')

        # Create a subdirectory at output_dir based on run_id hash
        run_id = wnb.id
        run_dir = output_dir/wandb_project/run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        import pdb; pdb.set_trace()
        
        # ---------------------------------------------------------------------
        # Construct masks
        # ---------------------------------------------------------------------
        holdout_mask, buffer_mask = create_block_speckled_mask(
            onp_rng,
            batch_shape,
            mask_block_shape,
            mask_buffer_size,
            frac_mask=mask_frac,
            frac_include_buffer=False,  # Do not include buffered elements in `frac_mask`
        )  # both arrays have shape (d1, d2)

        params, avg_train_lps, avg_test_ll, run_elapsed_time = train_and_evaluate_one(
            this_jax_key, k1, k2, k3, X,
            holdout_mask=holdout_mask,
            buffer_mask=buffer_mask,
            n_epochs=n_epochs,
            wnb=wnb,
        )

        pbar.set_postfix(train_lp=avg_train_lps[-1], test_ll=avg_test_ll)

        # Visualize
        fig = make_summary_figure(
            params, avg_train_lps, holdout_mask, buffer_mask, figsize=(8.5,11), dpi=120
        )
        fig_path = run_dir/'summary.png'
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()

        # ---------------------------------------------------------------------
        # Save results
        # ---------------------------------------------------------------------
        # Log to WandB
        wnb.summary['run_time_minutes'] = run_elapsed_time/60
        wnb.summary["avg_test_ll"] = avg_test_ll
        wnb.save(str(fig_path), base_path=str(run_dir), policy="now")  # save figure as file
        wnb.finish()

        # Log locally
        # Note: To save / recover a JAX PRNG key: 
        #   - key_data_bits = jr.key_data(key)  # uint32 dtype array
        #   - key = jr.wrap_key_data(key_data_bits)  # prng key array
        # Note: To save / recover NumPY PRNG
        #   - generator_state = onp_rng.bit_generator.state  # dict
        #   - onp_rng = onp.random.default_generator(); onp_rng.bit_generator.state = generator_state
        onp.savez_compressed(run_dir/'params.npz',
                             seed=seed,
                             run_id=run_id,
                             G=params[0], F1=params[1], F2=params[2], F3=params[3],
                             avg_train_lps=avg_train_lps,
                             avg_test_ll=avg_test_ll
                             )

    return

# To be used in conjunction with a WandB Sweep file (.yaml)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type=str,
        help='Path to .npz file containining binned syllable data.'
    )
    parser.add_argument('--k1', type=int)
    parser.add_argument('--k2', type=int)
    parser.add_argument('--k3', type=int)
    parser.add_argument('--mask_frac', type=float, default=0.2)
    parser.add_argument('--mask_block_shape', type=int, nargs=2, default=(1,3))
    parser.add_argument('--mask_buffer_size', type=int, nargs=2, default=(0,1))
    parser.add_argument('--n_folds', type=int, default=10)
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--wandb_project', type=str,)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--wandb_debug', action='store_true',
        help="Set wandb run display name to DEBUG"
    )

    args: dict = vars(parser.parse_args())

    # ===================================================================================
    # Load data
    # ===================================================================================
    # data_path = "/home/groups/swl1/eyz/data/moseq-drugs/syllable_binned_1min.npz"
    # output_base_dir = "/scratch/users/eyz/"
    data_path = args.pop('data_path')
    X, batch_axes, event_axes, metadata = load_wiltschko22_data(data_path)
    event_ndims = len(event_axes)
    print(f"X: shape={X.shape}, dtype={X.dtype}, event_ndims={event_ndims}")

    # Set global metadata variables for visualization
    session_drug_class = [
        name if name[:4] != 'anti' else 'anti-\n' + name[4:]
        for name in list(metadata['session']['drug_class'])
    ] # Modify long-form session drug class to print over two lines, for readability
    frames_per_bin = metadata['frames_per_bin']
    frames_per_sec = metadata['frames_per_sec']
    syllable_cluster_names = metadata['syllable']['cluster_names']

    # ===================================================================================
    # Run cross-validation!
    # ===================================================================================
    wandb.login()
    train_and_eval_and_log_nfolds(X, event_ndims, **args)