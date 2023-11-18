"""Principal factors analysis -- leave-one-factor-in (lofi) approach.

Identify the principle factors of analysis
Evaluate held-out log likelihood to sort factors by importance.

This script uses functions from the main `killifish.py` script to ensure
consistency of how data is loaded and masked.
"""

from __future__ import annotations
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
import tensorflow_probability.substrates.jax.distributions as tfd

from dtd.model3d import DirichletTuckerDecomp
from dtd.utils import get_wnb_project_df, download_wnb_params

from killifish import load_data
from kf_viz import draw_syllable_factors, draw_circadian_bases


# --------------------------------------------------------------

# Hardcode run information in for now
run_id = 'ig6dh2fo'
run_seed = 1698740237

# eternel-sweep 28
# run_id = '46grddu7'
# run_seed = 1698499854

# --------------------------------------------------------------
# --------------------------------------------------------------

wnb_entity = 'eyz'
wnb_project = 'kf-dtd-231022'
params_dir =  Path('/scratch/groups/swl1/killifish/p3_20230726-20230915/kf-dtd-231022')
data_dir = Path('/scratch/groups/swl1/killifish/p3_20230726-20230915/q2-aligned_10min')
          
# --------------------------------------------------------------

def main(datadir, run_id, seed):
    key = jr.PRNGKey(seed)
    key_mask, _ = jr.split(key)

    # Load data
    X, total_counts, mask = load_data(Path(datadir), key_mask)
    
    # Load fitted params
    params = download_wnb_params(wnb_entity, wnb_project, run_id, params_dir)
    k1, k2, k3 = params[0].shape
    alpha = 1.1

    # Instantiate model
    model = DirichletTuckerDecomp(total_counts, k1, k2, k3, alpha)

    # Leave one factor in!
    sorted_params, lls = model.sort_params(X, mask, params)
    
    print(lls[1-1])
    print(lls[2-1])
    print(lls[3-1])
    
    # jnp.savez(params_dir/run_id/'lofi.npz',
    #           F1=lls_1,
    #           F2=lls_2,
    #           F3=lls_3)
    return
  
main(data_dir, run_id, run_seed)
