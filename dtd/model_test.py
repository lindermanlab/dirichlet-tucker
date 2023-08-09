import pytest
import jax.numpy as jnp
import jax.random as jr

import dtd.model3d
import dtd.model4d

def make_random_mask(key, shape, train_frac=0.8):
    """Make binary mask to split data into a train (1) and validation (0) sets."""
    return jr.bernoulli(key, train_frac, shape)


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
    
