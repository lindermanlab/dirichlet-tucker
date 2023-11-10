"""Leave-one-factor-out analysis.

Evaluate held-out log likelihood to sort factors by importance.
"""

from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import pandas as pd

from dtd.model3d import DirichletTuckerDecomp
from dtd.utils import get_wnb_project_df, download_wnb_params

from kf_viz import draw_syllable_factors, draw_circadian_bases


# --------------------------------------------------------------

# Hardcode run information in for now
run_id = 'ig6dh2fo'
run_seed = 1698740237

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

def f1_loo(X, mask, model, params):
    G = params[0]
    F = params[1]
    K = F.shape[-1]
    
    lls = []
    for k in range(K):
        k_mask = (jnp.arange(K) == k)

        F_ = F[:,~k_mask]
        G_ = G[~k_mask,:,:]

        ll = model.heldout_log_likelihood(X, mask, (G_, F_, params[2], params[3]))
        lls.append(ll)
    
    return jnp.array(lls)

def f2_loo(X, mask, model, params):
    G = params[0]
    F = params[2]
    K = F.shape[-1]
    
    lls = []
    for k in range(K):
        k_mask = (jnp.arange(K) == k)

        F_ = F[:,~k_mask]
        G_ = G[:,~k_mask,:]

        ll = model.heldout_log_likelihood(X, mask, (G_, params[1], F_, params[3]))
        lls.append(ll)
    
    return jnp.array(lls)

def f3_loo(X, mask, model, params):
    G = params[0]
    F = params[3]
    K = F.shape[0]   # This is an event dimension! So things are switched.
    
    lls = []
    for k in range(K):
        k_mask = (jnp.arange(K) == k)

        F_ = F[~k_mask,:]
        G_ = G[:,:,~k_mask]

        ll = model.heldout_log_likelihood(X, mask, (G_, params[1], params[2], F_))
        lls.append(ll)
    
    return jnp.array(lls)

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

    # Leave one factor out!
    lls_1 = f1_loo(X, mask, model, params)
    lls_2 = f2_loo(X, mask, model, params)
    lls_3 = f3_loo(X, mask, model, params)
    
    print(lls_1)
    print(lls_2)
    print(lls_3)
    
    jnp.savez(params_dir/run_id/'lofo.npz',
              F1=lls_1,
              F2=lls_2,
              F3=lls_3)
    return
  
main(data_dir, run_id, run_seed)