from __future__ import annotations
from typing import Optional
from omegaconf import DictConfig
import wandb
import hydra
from pathlib import Path
from datetime import datetime
import time

import numpy as onp
import jax.numpy as jnp
import jax.random as jr
import optax
from sklearn.model_selection import StratifiedGroupKFold

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from dtd.data import concatenate_sessions, generate_cross_validation_masks
from dtd.model3d import DirichletTuckerDecomp
from viz import draw_syllable_factors, draw_circadian_bases

DEFAULT_LR_SCHEDULE_FN = (
    lambda n_minibatches, n_epochs:
        optax.cosine_decay_schedule(
            init_value=1.,
            alpha=0.,
            decay_steps=n_minibatches*n_epochs,
            exponent=0.8,
        )
)

# ============================================================================ #
#                               FIT & EVALUATION                               #
# ============================================================================ #

def evaluate_fit(model, X, mask, params, verbose=False):
    """Compute heldout log likelihood and percent deviation from saturated model."""

    # Compute test log likelihood
    if verbose: print("Evaluating fit...", end="")
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

    # Compute test log likelihood fraction deviation explained of fitted model from
    # saturated model (upper bound), relative to baseline (lower bound).
    frac_dev_explained = (test_ll - baseline_test_ll) / (saturated_test_ll - baseline_test_ll)
    if verbose: print("Done.")

    return frac_dev_explained, test_ll, baseline_test_ll, saturated_test_ll


def fit_model(key, method, X, masks_dict, total_counts, k1, k2, k3, alpha=1.1,
              lr_schedule_fn=None, minibatch_size=1024, n_epochs=100,
              wnb=None, verbose=False):
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

    # -----------------------------------------------------------------------------------
    # Randomly initialize parameters
    # -----------------------------------------------------------------------------------
    if verbose: print("Initializing model...", end="")
    d1, d2, d3 = X.shape
    init_params = model.sample_params(key_init, d1, d2, d3)
    if verbose: print("Done.")

    # -----------------------------------------------------------------------------------
    # Fit model to data with EM
    # -----------------------------------------------------------------------------------
    mask = masks_dict['val'] & masks_dict['buffer'] & masks_dict['test']

    if verbose: print(f"Fitting model with {method} EM...", end="")
    if method == 'stochastic':
        # Set default learning rate schedule function if none provided
        lr_schedule_fn = lr_schedule_fn if lr_schedule_fn is not None else DEFAULT_LR_SCHEDULE_FN

        params, lps = model.stochastic_fit(X, mask, init_params, n_epochs,
                                           lr_schedule_fn, minibatch_size, key_fit,
                                           drop_last=False, wnb=wnb)
    else:
        params, lps = model.fit(X, mask, init_params, n_epochs, wnb=wnb)

    if verbose:  print("Done.")

    # -----------------------------------------------------------------------------------
    # Evaluate the model on heldout samples
    # -----------------------------------------------------------------------------------
    if verbose: print("Evaluating model fit...", end="")
    
    # TODO Evaluate validation and test sets seperately
    frac_dev_explained, test_ll, baseline_test_ll, saturated_test_ll \
                                        = evaluate_fit(model, X, mask, params)
    
    # Sort the model by principal factors of variation
    params, lofo_test_lls = model.sort_params(X, mask, params)

    if verbose:  print("Done.")

    # -----------------------------------------------------------------------------------
    # Return results
    # -----------------------------------------------------------------------------------
    n_train = mask.sum()
    n_test = (~mask).sum()
    return dict(G=params[0],
                F1=params[1],
                F2=params[2],
                F3=params[3],
                mask=mask,
                avg_train_lps=lps/n_train,
                avg_test_ll=test_ll/n_test,
                avg_baseline_test_ll=baseline_test_ll/n_test,
                avg_saturated_test_ll=saturated_test_ll/n_test,
                avg_lofo_test_ll_1=lofo_test_lls[0]/n_test,
                avg_lofo_test_ll_2=lofo_test_lls[1]/n_test,
                avg_lofo_test_ll_3=lofo_test_lls[2]/n_test,
               )

def make_visual_summary(F1, F2, F3, frac_dev_1, frac_dev_2, frac_dev_3):

    fig = plt.figure(figsize=(8.5,11), dpi=120)
    gs_main = mpl.gridspec.GridSpec(nrows=3, ncols=1, height_ratios=[0.5,2,1], hspace=0.5)
    
    # -----------------------------------------------------------------------------------
    # Fraction deviation explained
    # -----------------------------------------------------------------------------------
    gs_lofo = gs_main[0].subgridspec(nrows=1, ncols=3)
    for i, (name, frac_dev) in enumerate([('aging factors', frac_dev_1),
                                          ('circadian bases', frac_dev_2),
                                          ('behavioral topics', frac_dev_3)]):
        ax = fig.add_subplot(gs_lofo[0,i])

        # Draw frac deviation and reference (y=0)
        ax.plot(frac_dev, marker='.', color='k')
        ax.axhline(0, alpha=0.4, color='k', lw=1)

        # Formatting
        if i == 0: ax.set_ylabel('frac dev from baseline')
            
        ax.set_title(name, fontsize='medium')
        ax.tick_params(labelsize='small')
        
        sns.despine(ax=ax)

    # -----------------------------------------------------------------------------------
    # Behavioral syllables
    # -----------------------------------------------------------------------------------
    ax = fig.add_subplot(gs_main[1])

    K, D = F3.shape
    draw_syllable_factors(F3, autosort=False, ax=ax, im_kw={'cmap': 'Greys', 'norm': mpl.colors.LogNorm(0.5*1/D, 1.0)})
    ax.text(0.5, 1.1, 'behavioral syllables', transform=ax.transAxes, ha='center')
    
    # Label each factor with fraction deviation explained
    ax.set_yticks(onp.arange(K), minor=True)
    ax.set_yticklabels([f'{frac:+.2f}' for frac in frac_dev_3], minor=True)
    ax.tick_params(which='minor', axis='y', left=False, right=True, labelleft=False, labelright=True, labelsize='x-small')
    
    # -----------------------------------------------------------------------------------
    # Circadian_bases
    # -----------------------------------------------------------------------------------
    D, K = F2.shape
    gs2 = gs_main[2].subgridspec(nrows=K, ncols=1, hspace=0.1)
    axs = [fig.add_subplot(gs2[k]) for k in range(K)]
    draw_circadian_bases(F2, tod_freq='4H', autosort=False, axs=axs);
    
    # Label each factor with fraction deviation explained
    for ax, frac in zip(axs, frac_dev_2):
        ax.text(1.02, 0.5, f'{frac:+.2f}', transform=ax.transAxes, ha='left',
                fontsize='x-small', va='center',
                bbox=dict(facecolor='white', alpha=0.1, pad=2))

    return fig
    

def run_one(X, masks_dict, total_counts, k1, k2, k3, alpha,
            seed, method='full', minibatch_size=1024, n_epochs=5000,
            wnb_project=None, out_dir=None):
    """Fit data to one set of model parameters."""

    # If no seed provided, generate a random one based on the timestamp
    if seed is None:
        seed = int(datetime.now().timestamp())

    if wnb_project is not None:
        wnb = wandb.init(
            project=wnb_project,
            config={
                'k1': k1,
                'k2': k2,
                'k3': k3,
                'alpha': alpha,
                'seed': seed,
                'method': method,
                'minibatch_size': minibatch_size,
            }
        )
        wandb.define_metric('avg_lp', summary='min')
        
        run_id = wandb.run.id
    else:
        wnb = None
        run_id = datetime.now().strftime("%Y%b%d-%H:%M:%S")
        run_id += f"-{k1=}-{k2=}-{k3=}"
    
    # Create a subdirectory at outdir based on run_id hash
    out_dir = out_dir/run_id
    out_dir.mkdir(parents=True)
    print(f"Saving results to...{str(out_dir)}")
    
    # ==========================================================================

    run_start_time = time.time()

    key = jr.key(seed)
    fit_results = fit_model(
            key, method, X, mask, total_counts, k1, k2, k3, alpha=alpha,
            minibatch_size=minibatch_size, n_epochs=n_epochs,
            wnb=wnb)

    run_elapsed_time = time.time() - run_start_time

    # ==========================================================================
    
    # Save results locally
    fpath_params = out_dir/'params.npz'
    onp.savez_compressed(fpath_params,
                         seed=seed,
                         run_id=run_id,
                         **fit_results
                         )
    
    # Visualize parameters and save them
    frac_dev_1, frac_dev_2, frac_dev_3 = [
        (fit_results['avg_baseline_test_ll'] - lofo_test_ll) / fit_results['avg_baseline_test_ll']
        for lofo_test_ll in [fit_results['avg_lofo_test_ll_1'], fit_results['avg_lofo_test_ll_2'], fit_results['avg_lofo_test_ll_3']]
    ]

    fig = make_visual_summary(fit_results['F1'], fit_results['F2'], fit_results['F3'],
                              frac_dev_1, frac_dev_2, frac_dev_3)
    fpath_viz = out_dir/'summary.png'
    plt.savefig(fpath_viz, bbox_inches='tight')
    plt.close()

    # Log summry metrics. lps are automatically logged by model.stochastic_fit
    if wnb_project:
        frac_dev_explained = (fit_results['avg_test_ll'] - fit_results['avg_baseline_test_ll'])
        frac_dev_explained /= (fit_results['avg_saturated_test_ll'] - fit_results['avg_baseline_test_ll'])
        
        wnb.summary["frac_dev_explained"] = frac_dev_explained
        wnb.summary["avg_test_ll"] = fit_results['avg_test"ll']
        wnb.summary['run_time_min'] = run_elapsed_time/60

        wandb.save(str(fpath_topics), policy='now')
        wandb.save(str(fpath_bases), policy='now') 
        wandb.save(str(fpath_params), policy='now')
        wandb.finish()

    return

@hydra.main(config_path="../../configs", config_name="default", version_base=None)
def run_cross_validation(config: DictConfig):

    # ---------------------------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------------------------
    # data_dict = load_dataset(config.dataset.type, ...)
    data_dict = concatenate_sessions(
        config.dataset.data_dir,
        transform_method=config.dataset.transform_method,
        transform_kwargs=config.dataset.transform_kwargs,
    )
    batch_ndim, event_ndim = config.dataset.batch_ndim, config.dataset.event_ndim
    event_shape = data_dict['tensor'].shape[-event_ndim:]
    batch_shape = data_dict['tensor'].shape[:batch_ndim]
    assert batch_shape + event_shape == data_dict['tensor'].shape, \
        f"Expected batch and event shape to produce {data_dict['tensor'].shape}, but got {batch_shape=} and {event_shape=}."
    
    total_counts = data_dict['tensor'].sum(axis=tuple(list(range(batch_ndim, batch_ndim+event_ndim))))

    # ---------------------------------------------------------------------------------
    # Cross-validate
    # ---------------------------------------------------------------------------------
    masks_dict = generate_cross_validation_masks(
        batch_shape,
        groups=data_dict['mode_0_ids'],
        val_frac=config.cross_validation.val_frac,
        test_frac=config.cross_validation.test_frac,
        mask_method=config.cross_validation.mask_method,
        mask_kwargs=config.cross_validation.mask_kwargs,
        n_folds=config.cross_validation.get("n_folds", None),
        seed=config.cross_validation.seed,
    )
        
    for i_fold, mask in enumerate(cv):
        run_one(data_dict['tensor'],
                masks_dict,
                total_counts,
                k1=config.model.k1,
                k2=config.model.k2,
                k3=config.model.k3,
                alpha=config.model.alpha,
                seed=config.fit.seed,
                method=config.fit.method,
                n_epochs=config.fit.n_epochs,
                wnb_project=config.logging.wandb_project,
                out_dir=config.logging.output_dir
        )

        if i_fold + 1 == n_splits:
            break

if __name__ == "__main__":
    run_cross_validation()