"""Principal factors analysis -- leave-one-factor-in (lofi) approach.

Identify the principle factors of analysis
Evaluate held-out log likelihood to sort factors by importance.

TODO the lofi analysis should be folded in the `model[X]d.py` source code.
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

def _get_subshape(X, axes: tuple=(0,)):
    """Return shape of X along specified axes."""
    return tuple([X.shape[i] for i in axes])

def _load_data(fpath_list: list[str]):
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

def load_data(datadir: Path, key: jr.PRNGKey, train_frac: float=0.8):
    """Return data array and total counts."""
    
    # This variable is left over from debugging the killifish.py code.
    # However, keeping and hardcoding the default value for patter recognition
    max_samples = -1
    
    # Load data from input directory. Search in all subdirectories
    print("Loading data...",end="")
    fpath_list = sorted([f for f in Path(datadir).rglob("*") if f.is_file()])
    data = _load_data(fpath_list)
    
    # Cast integer counts to float32 dtype
    X = jnp.asarray(data['X'], dtype=jnp.float32)[:max_samples]
    print("Done.")
    print(f"\tData array: shape {X.shape}, {X.nbytes/(1024**3):.1f}GB")

    # Create random mask to hold-out data for validation
    batch_shape = _get_subshape(X, data['batch_axes'])
    mask = make_random_mask(key, batch_shape, train_frac)

    # Get total counts. Since any random batch indices should give the same number
    # counts, i.e. if X has batch_axes of (0,1), then X[i,j].sum() = C for all i,j,
    # and since we assume that the batch axes are the leading dimensions, we can
    # calculate the total counts  of data by abusing data['batch_axes'] as indices
    total_counts = float(X[data['batch_axes']].sum())
    assert total_counts.is_integer(), \
        f'Expected `total_counts` to be integer values, but got {total_counts}'
    total_counts = int(total_counts)
    
    return X, total_counts, mask
    
def make_random_mask(key, shape, train_frac=0.8):
    """Make binary mask to split data into train (1) and test (0) sets."""
    return jr.bernoulli(key, train_frac, shape)

def leave_one_out_heldout_log_likelihood(X, mask, model, params):
    """scan step involves taking all but the first index.
    for the carry, the parameter and core tensor are then rolled so that the next iteration, there will be a new index that is not taken
    """
    
    G, F1, F2, F3 = params
    K1, K2, K3 = G.shape

    def f1_step(carry, k):
        g_axis, f_axis = 0, 1

        rolled_G, rolled_F = carry
        K = rolled_G.shape[g_axis]
        
        # G[1:,:,:], shape (K1-1, K2, K3)
        G_ = lax.dynamic_slice_in_dim(rolled_G, 1, K-1, axis=g_axis)

        # F[:,1:], shape(D1, K1-1)
        F_ = lax.dynamic_slice_in_dim(rolled_F, 1, K-1, axis=f_axis)

        # Compute held-out log likelihood
        params_ = (G_, F_, F2, F3)
        ll = model.heldout_log_likelihood(X, mask, params_)

        # Roll the carried core tensor and factor for the next iteration
        rolled_G = jnp.roll(rolled_G, -1, axis=g_axis)
        rolled_F = jnp.roll(rolled_F, -1, axis=f_axis)
        return (rolled_G, rolled_F), ll
    
    def f2_step(carry, k):
        g_axis, f_axis = 1, 1

        rolled_G, rolled_F = carry
        K = rolled_G.shape[g_axis]
        
        # G[1:,:,:], shape (K1-1, K2, K3)
        G_ = lax.dynamic_slice_in_dim(rolled_G, 1, K-1, axis=g_axis)

        # F[:,1:], shape(D1, K1-1)
        F_ = lax.dynamic_slice_in_dim(rolled_F, 1, K-1, axis=f_axis)

        # Compute held-out log likelihood
        params_ = (G_, F1, F_, F3)
        ll = model.heldout_log_likelihood(X, mask, params_)

        # Roll the carried core tensor and factor for the next iteration
        rolled_G = jnp.roll(rolled_G, -1, axis=g_axis)
        rolled_F = jnp.roll(rolled_F, -1, axis=f_axis)
        return (rolled_G, rolled_F), ll
    
    def f3_step(carry, k):
        g_axis, f_axis = 2, 0

        rolled_G, rolled_F = carry
        K = rolled_G.shape[g_axis]
        
        # G[:,1:,:], shape (K1, K2, K3-1)
        G_ = lax.dynamic_slice_in_dim(rolled_G, 1, K-1, axis=g_axis)

        # F[1:], shape(K3-1, D3)
        F_ = lax.dynamic_slice_in_dim(rolled_F, 1, K-1, axis=f_axis)

        # Compute held-out log likelihood
        params_ = (G_, F1, F2, F_)
        ll = model.heldout_log_likelihood(X, mask, params_)

        # Roll the carried core tensor and factor for the next iteration
        rolled_G = jnp.roll(rolled_G, -1, axis=g_axis)
        rolled_F = jnp.roll(rolled_F, -1, axis=f_axis)
        return (rolled_G, rolled_F), ll
    
    _, lls_1 = lax.scan(f1_step, (G, F1), jnp.arange(K1))
    _, lls_2 = lax.scan(f2_step, (G, F2), jnp.arange(K2))
    _, lls_3 = lax.scan(f3_step, (G, F3), jnp.arange(K3))
    
    return lls_1, lls_2, lls_3

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
    lls_1, lls_2, lls_3 = leave_one_out_heldout_log_likelihood(X, mask, model, params)
    
    print(lls_1)
    print(lls_2)
    print(lls_3)
    
    print()
    jnp.savez(params_dir/run_id/'lofi.npz',
              F1=lls_1,
              F2=lls_2,
              F3=lls_3)
    return
  
main(data_dir, run_id, run_seed)
