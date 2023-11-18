"""Principal factors analysis -- leave-one-factor-out (lofo) approach.

Identify the principle factors of analysis
Evaluate held-out log likelihood to sort factors by importance.

This script uses functions from the main `killifish.py` script to ensure
consistency of how data is loaded and masked.

This script is intermediary script -- it is used to update the `params.npz` file
from prior to Nov. 17, 2023 to match what would be saved after this date,
e.g. avg_test_lps, avg_baseline_ll + etc, avg_lofo_test_ll_1 + etc, seed, run_id
"""

from __future__ import annotations
import os
from pathlib import Path
import wandb
import time

import numpy as onp
import jax.numpy as jnp
import jax.random as jr

from dtd.model3d import DirichletTuckerDecomp
from dtd.utils import get_wnb_project_df, download_wnb_params

from killifish import load_data, evaluate_fit, make_random_mask

import matplotlib.pyplot as plt
from kf_viz import draw_syllable_factors, draw_circadian_bases

# --------------------------------------------------------------

wnb_entity = 'eyz'
wnb_project = 'kf-dtd-231022'
OUT_DIR =  Path('/scratch/groups/swl1/killifish/p3_20230726-20230915') / wnb_project
DATA_DIR = Path('/scratch/groups/swl1/killifish/p3_20230726-20230915/q2-aligned_10min')

# --------------------------------------------------------------
# We can save time by loading data once!! However, make to ignore mask
total_counts, X, _, _, _ = load_data(DATA_DIR, key=None)

def main(run_id, seed, k1, k2, k3):
    # Generate mask, as it was generated in the original script
    key = jr.PRNGKey(seed)
    key_mask, _ = jr.split(key)
    mask = make_random_mask(key_mask, X.shape[:-1], train_frac=0.8)
    
    # Load fitted params
    params = download_wnb_params(wnb_entity, wnb_project, run_id, OUT_DIR)
    assert params[0].shape == (k1,k2,k3), \
        f"Expected loaded core tensor to have shape ({k1}, {k2}, {k3}), but got {params[0].shape}."

    # Instantiate model
    alpha = 1.1
    model = DirichletTuckerDecomp(total_counts, k1, k2, k3, alpha)

    # (Re-)evaluate the fit
    _, test_ll, baseline_test_ll, saturated_test_ll \
                                        = evaluate_fit(model, X, mask, params)
    
    # Sort parameters by their variance explained
    print("Sorting parameters...", end='')
    params, lofo_test_lls = model.sort_params(X, mask, params)
    print("Done.")
    
    # Save results locally
    run_root_dir = OUT_DIR / run_id
    
    fpath_params = run_root_dir / 'params.npz'
    n_test = (~mask).sum()
    onp.savez_compressed(fpath_params,
                         G=params[0],
                         F1=params[1],
                         F2=params[2],
                         F3=params[3],
                         seed=seed,
                         avg_train_lps=None,
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
    fpath_topics = run_root_dir/'behavioral-topics.png'
    plt.savefig(fpath_topics, bbox_inches='tight')
    plt.close()

    fig_bases = draw_circadian_bases(params)
    fpath_bases = run_root_dir/'circadian-bases.png'
    plt.savefig(fpath_bases, bbox_inches='tight')
    plt.close()

    # Sync these files with WnB run
    run = wandb.Api().run(f"{wnb_entity}/{wnb_project}/{run_id}")
    run.upload_file(str(fpath_params), root=str(run_root_dir))
    run.upload_file(str(fpath_topics), root=str(run_root_dir))
    run.upload_file(str(fpath_bases), root=str(run_root_dir))
    wandb.finish()
    return

# Retrieve "id", "name", "seed"
config_keys, summary_keys = ['seed', 'k1', 'k2', 'k3'], []
df = get_wnb_project_df(wnb_entity, wnb_project, config_keys, summary_keys)

for i_df, run_info in df.iterrows():
    start_time = time.time()
    print(run_info)
    main(run_info['id'], run_info['seed'], run_info['k1'], run_info['k2'], run_info['k3'])
    print("run time:", time.time() - start_time)