import pytest
import jax.numpy as jnp
import jax.random as jr
from jax import lax

import dtd.model3d
import dtd.model4d
from dtd.utils import ShuffleIndicesIterator

def make_random_mask(key, shape, train_frac=0.8):
    """Make binary mask to split data into a train (1) and validation (0) sets."""
    return jr.bernoulli(key, train_frac, shape)

def make_shuffled_indices(key, shape, minibatch_size):
    """Shuffle indices and make into minibatches."""
    indices = jnp.indices(shape).reshape(len(shape), -1).T
    indices = jr.permutation(key, indices, axis=0)
    
    n_batches = len(indices) // minibatch_size
    batched_indices, remaining_indices \
                            = jnp.split(indices, (n_batches*minibatch_size,))
    batched_indices = batched_indices.reshape(n_batches, minibatch_size, -1)

    return batched_indices, remaining_indices

def test_fit_3d(shape=(20, 10, 25), 
                rank=(5, 4, 3), 
                batch_ndims=2,
                total_counts=1000,
                n_iters=5,
                key=0
                ):
    """Evaluate full-batch EM fit."""
    key = jr.PRNGKey(key)
    key_true, key_data, key_mask, key_init = jr.split(key, 4)

    # Instantiate the model
    model = dtd.model3d.DirichletTuckerDecomp(total_counts, *rank)

    # Generate observations from an underlying DTD model
    true_params = model.sample_params(key_true, *shape)
    X = model.sample_data(key_data, true_params)
    mask_shape = shape[:batch_ndims] 
    mask = make_random_mask(key_mask, mask_shape)

    # Initialize a different set of parameters and fit
    init_params = model.sample_params(key_init, *shape)

    fitted_params, lps = model.fit(X, mask, init_params, n_iters)

    # Check that log probabilities are monotonically increasing
    assert jnp.all(jnp.diff(lps) > 0), \
        f'Expected lps to be monotonically increasing.'
    

def test_fit_4d(shape=(10, 9, 8, 15), 
                rank=(5, 4, 3, 2), 
                batch_ndims=2,
                total_counts=1000,
                n_iters=5,
                key=0
                ):
    """Evaluate full-batch EM fit."""
    key = jr.PRNGKey(key)
    key_true, key_data, key_mask, key_init = jr.split(key, 4)

    # Instantiate the model
    model = dtd.model4d.DirichletTuckerDecomp(total_counts, *rank)
    
    # Generate observations from an underlying DTD model
    true_params = model.sample_params(key_true, *shape)
    X = model.sample_data(key_data, true_params)
    mask_shape = shape[:batch_ndims] 
    mask = make_random_mask(key_mask, mask_shape)

    # Initialize a different set of parameters and fit
    init_params = model.sample_params(key_init, *shape)

    # Fit the model
    fitted_params, lps = model.fit(X, mask, init_params, n_iters)

    # Check that log probabilities are monotonically increasing
    assert jnp.all(jnp.diff(lps) > 0), \
        f'Expected lps to be monotonically increasing.'
    
def test_get_minibatch_3d(shape=(20, 10, 25), 
                          rank=(5, 4, 3), 
                          batch_ndims=2,
                          total_counts=1000,
                          minibatch_size=30,
                          key=0
                          ):
    """Evaluate that lax functions are indexing arrays correctly."""

    key = jr.PRNGKey(key)
    key_params, key_data, key_mask, key_stats, key_shuffle = jr.split(key, 5)

    # Instantiate the model and generate parameters
    model = dtd.model3d.DirichletTuckerDecomp(total_counts, *rank)
    
    params = model.sample_params(key_params, *shape)
    
    # Generate data and mask
    X = model.sample_data(key_data, params)
    
    mask_shape = shape[:batch_ndims] 
    mask = make_random_mask(key_mask, mask_shape)
    
    # Generate random stats (ensure non-zero for evaluation)
    _stats = model._zero_rolling_stats(X, minibatch_size)
    stats = []
    for key_ss, ss in zip(jr.split(key_stats, len(_stats)), _stats):
        stats.append(jr.uniform(key_ss, ss.shape, minval=0, maxval=100))
    stats = tuple(stats)

    # Get a minibatch of indices. Ignore incomplete minibatch.
    # shape (n_batches, minibatch_size, batch_ndims) = (6,30,2)
    batched_indices, _ = make_shuffled_indices(key_shuffle,
                                               shape[:batch_ndims],
                                               minibatch_size,)    
    these_idxs = batched_indices[0]

    # Evaluate
    this_X, this_mask, these_params, these_stats \
                    = model._get_minibatch(these_idxs, X, mask, params, stats)
    
    refr_X = X[these_idxs[:,0], these_idxs[:,1]]
    assert jnp.allclose(this_X, refr_X, atol=1e-8)

    refr_mask = mask[these_idxs[:,0], these_idxs[:,1]]
    assert jnp.allclose(this_mask, refr_mask, atol=1e-8)

    refr_Psi = params[1][these_idxs[:,0], :]
    refr_Phi = params[2][these_idxs[:,1], :]
    assert jnp.allclose(these_params[1], refr_Psi, atol=1e-8)
    assert jnp.allclose(these_params[2], refr_Phi, atol=1e-8)

    refr_aPsi = stats[1][these_idxs[:,0], :]
    refr_aPhi = stats[2][these_idxs[:,1], :]
    assert jnp.allclose(these_stats[1], refr_aPsi, atol=1e-8)
    assert jnp.allclose(these_stats[2], refr_aPhi, atol=1e-8)

    # Make sure we can lax.scan through it
    def fn(carry, these_idxs):
        this_X, this_mask, these_params, these_stats \
                    = model._get_minibatch(these_idxs, X, mask, params, stats)
        
        return None, jnp.array([this_X.mean(),
                                this_mask.mean(),
                                these_params[1].mean(),
                                these_stats[2].mean()])
    
    _, outs = lax.scan(fn, None, batched_indices)
    assert len(outs) == len(batched_indices)
