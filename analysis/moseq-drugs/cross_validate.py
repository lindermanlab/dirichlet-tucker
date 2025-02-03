"""Sweep over ranks to cross-validate model parameters using WandB

Some parameters of this file need to be manually edited.
"""
from pathlib import Path
import itertools
from math import prod
import time
import wandb

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
from dtd.utils import create_block_speckled_mask

from dtd.viz import (
    draw_drug_class_boxes,
    plot_buffered_mask,
    set_syllable_cluster_ticks,
    set_time_within_session_ticks,
)

def make_summary_figure(
    params, avg_train_lps, holdout_mask, buffer_mask,
    session_drug_class,
    frames_per_bin, frames_per_sec,
    syllable_cluster_names, 
    tick_period=10, tick_units='min',
    figsize=(8.5,11), dpi=72
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

class DummyKFold(BaseCrossValidator):
    """Dummy generator, yields all indices as train indices."""

    def __init__(self, n_splits, *, shuffle=False, random_state=None):
        n_splits = int(n_splits)
        self.n_splits = n_splits
        self.shuffle = False
        self.random_state = None

    def get_n_splits(self, split, X, y=None, groups=None):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        indices = onp.arange(len(X))
        for _ in range(self.n_splits):
            yield indices, []


def run_one(
    key, k1, k2, k3,
    total_counts, X, holdout_mask, buffer_mask,
    session_drug_class, 
    frames_per_bin, frames_per_sec,
    syllable_cluster_names,
    n_epochs=500,
    wandb_project_name=None, out_dir=None
):

    assert wandb_project_name is not None

    wnb = wandb.init(
        project=wandb_project_name,
        config={'k1': k1, 'k2': k2, 'k3': k3,}
    )
    wandb.define_metric('avg_lp', summary='min')

    if out_dir is not None:
        # Create a subdirectory at outdir based on run_id hash
        run_id = wandb.run.id
        
        out_dir = out_dir/run_id            
        out_dir.mkdir(parents=True)
    
    # ==========================================================================

    run_start_time = time.time()

    d1, d2, d3 = X.shape
    init_key, stem_key, = jr.split(key)  # stem_key used for stochastic em
    
    # Instantiate model and initialize parameters
    model = DirichletTuckerDecomp(total_counts, k1, k2, k3, alpha=1.1)
    init_params = model.sample_params(init_key, d1, d2, d3, conc=0.5)

    # Fit model to held-in data
    train_mask = jnp.asarray(~(holdout_mask + buffer_mask))
    params, train_lps = model.fit(X, train_mask, init_params, n_epochs, wnb=wnb)
    avg_train_lps = train_lps / (train_mask.sum() * d3)

    # Evaluate fitted parameters on held-out data
    test_mask = jnp.asarray(~(holdout_mask))  # model.heldout_log_likelihood inverts this
    test_ll = model.heldout_log_likelihood(X, test_mask, params)
    avg_test_ll = test_ll / ((~test_mask).sum() * d3)

    run_elapsed_time = time.time() - run_start_time

    # ==========================================================================
    
    # Save results locally
    onp.savez_compressed(out_dir/'params.npz',
                         key=jr.key_data(key), # recover key via jr.wrap_key_data(key)
                         run_id=run_id,
                         G=params[0], F1=params[1], F2=params[2], F3=params[3],
                         avg_train_lps=avg_train_lps,
                         avg_test_ll=avg_test_ll
                         )
    
    # # Visualize parameters and save them
    # frac_dev_1, frac_dev_2, frac_dev_3 = [
    #     (fit_results['avg_baseline_test_ll'] - lofo_test_ll) / fit_results['avg_baseline_test_ll']
    #     for lofo_test_ll in [fit_results['avg_lofo_test_ll_1'], fit_results['avg_lofo_test_ll_2'], fit_results['avg_lofo_test_ll_3']]
    # ]

    fig = make_summary_figure(
        params, avg_train_lps, holdout_mask, buffer_mask,
        session_drug_class,
        frames_per_bin, frames_per_sec,
        syllable_cluster_names,
        tick_period=10, tick_units='min',
        figsize=(8.5,11), dpi=72
    )
    fig_path = out_dir/'summary.png'
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

    # Log to WandB
    wnb.summary['run_time_min'] = run_elapsed_time/60
    wnb.summary["avg_test_ll"] = avg_test_ll
    wandb.save(str(fig_path), base_path=str(out_dir), policy="now")  # save figure as file
    wandb.finish()

    return


def run_sweep(
    filepath, project_name, output_base_dir,
    k1_sweep, k2_sweep, k3_sweep,
    vldtn_frac=0.10, vldtn_frac_include_buffer=True,
    vldtn_block_shape=(1,3), vldtn_buffer_size=(0,1),
    test_frac=0.10,
    max_kfolds=5,
    n_epochs=500,
    seed=0
):
    output_dir = Path(output_base_dir) / project_name

    # ===================================================================================
    # Load data
    # ===================================================================================
    X, batch_axes, event_axes, metadata = load_wiltschko22_data(filepath)
    print(f"X: shape {X.shape}, {X.dtype} dtype")

    d1, d2, d3 = n_sessions, n_timebins, n_syllables = X.shape
    batch_shape = X.shape[:len(batch_axes)]
    
    X = jnp.asarray(X, dtype=float)
    total_counts = int(X.sum(event_axes).ravel()[0])
    assert jnp.all(X.sum(event_axes) == total_counts)

    # Modify long-form session drug class to print over two lines, for readability
    session_drug_class_twoline = [
        name if name[:4] != 'anti' else 'anti-\n' + name[4:]
        for name in list(metadata['session']['drug_class'])
    ]

    # ===================================================================================
    # Run cross validation sweep
    # ===================================================================================
    jax_rng = jr.key(seed)
    onp_rng = onp.random.default_rng(seed=seed)

    frac_mask = vldtn_frac / (1-test_frac)  # adjust validation mask fraction

    x_kfold = onp.arange(n_sessions)
    y_kfold = metadata['session']['drug_class']

    if test_frac > 0:
        n_splits = int(1/test_frac)
        kfold = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    else:
        kfold = DummyKFold(max_kfolds)

    for i_fold, (train_idxs, test_idxs) in enumerate(kfold.split(x_kfold, y_kfold)):
        # ----------------------------------------
        # Contruct mask
        # ----------------------------------------
        holdout_mask = onp.zeros(batch_shape, dtype=bool)
        buffer_mask = onp.zeros(batch_shape, dtype=bool)

        _holdout_mask, _buffer_mask = create_block_speckled_mask(
            onp_rng,
            (len(train_idxs), *batch_shape[1:]),
            vldtn_block_shape, vldtn_buffer_size,
            frac_mask=frac_mask, frac_include_buffer=vldtn_frac_include_buffer,
        )  # shape (n_train, n_timebins)

        # Fill in mask
        holdout_mask[train_idxs] = _holdout_mask
        holdout_mask[test_idxs] = True

        # Fill in buffer
        buffer_mask[train_idxs] = _buffer_mask

        # NOTE Original code expects mask to indicate which entries to hold-in.
        train_mask = jnp.asarray(~(holdout_mask + buffer_mask))
        test_mask = jnp.asarray(~(holdout_mask))  # model.heldout_log_likelihood inverts this

        for i_grid, (k1, k2, k3) in enumerate(
            itertools.product(k1_sweep, k2_sweep, k3_sweep)
        ):
            this_key = jr.fold_in(jr.fold_in(jax_rng, i_fold), i_grid)

            run_one(
                this_key, k1, k2, k3,
                total_counts, X, holdout_mask, buffer_mask,
                session_drug_class_twoline,
                metadata['frames_per_bin'], metadata['frames_per_sec'],
                metadata['syllable']['cluster_names'],
                n_epochs=n_epochs,
                wandb_project_name=project_name, out_dir=output_dir,
            )

        if i_fold == max_kfolds-1:
            break


if __name__ == "__main__":
    wandb.login()
    
    filepath = "/home/groups/swl1/eyz/data/moseq-drugs/syllable_binned_1min.npz"
    output_base_dir = Path(f"/scratch/users/eyz/")
    
    # # ------------------------------------------------------------------
    # Re-use run_sweep to run with multi-init fit
    project_name = "moseq-dtd-fit-20250130"
    k1_sweep = [20]
    k2_sweep = [4]
    k3_sweep = [30]
    max_kfolds = 10  # number of initializations
    #  seed = 20250130  # first 5 runs, originally forgot to update max kfolds
    seed = 20250129

    run_sweep(
        filepath, project_name, output_base_dir, k1_sweep, k2_sweep, k3_sweep,
        vldtn_frac=0.0, test_frac=0.0,
        seed=seed
    )

    # # ------------------------------------------------------------------
    # # SWEEP WITH vldtn_frac = 0.20, test_frac=0.0
    # project_name = "moseq-dtd-sweep-20250130-speckle-only"

    # k1_sweep = [5,10,20,30,40]
    # k2_sweep = [2,4,8]
    # k3_sweep = [5,10,20,30,40,50]

    # run_sweep(
    #     filepath, project_name, output_base_dir, k1_sweep, k2_sweep, k3_sweep,
    #     vldtn_frac=0.20, vldtn_frac_include_buffer=False, test_frac=0.0
    # )

    # ------------------------------------------------------------------
    # SWEEP WITH vldtn_frac = 0.10, test_frac =.10
    # project_name = "moseq-dtd-sweep-20250128-2"

    # k1_sweep = [5,10,15,20,30,50]
    # k2_sweep = [2,5,10]
    # k3_sweep = [5,10,15,20,30,50]

    # run_sweep(
    #     filepath, project_name, output_base_dir, k1_sweep, k2_sweep, k3_sweep,
    # )