import pytest
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import lax
import optax

from dtd.poisson_tucker_3d import (
    PoissonTucker,
    L2PenalizedMultinomialTucker,
    SimplexMultinomialTucker,
)

from dtd.fit import fit_opt

def test_fit_opt(
    optimizer=optax.adam(1e-1),
    model_klass=PoissonTucker,
    model_full_shape=(10,9,8),
    model_core_shape=(5,4,3),
    n_iters=50,
):
    key = jr.PRNGKey(42)
    key_model, key_sample, key_mask, key_init = jr.split(key, 4)

    # Generate some random data from a model
    true_model = model_klass.random_init(
        key_model, model_full_shape, model_core_shape
    )
    data = true_model.sample(key_sample,)
    data_mask = true_model.random_mask(key_mask, frac=0.8)

    # --------------------------------------------------------------------------
    minibatch_size = 0
    log_prior_weight = 1

    # Define objective function
    def objective_fn(model) -> float:
        """Objective function, differentiate with respect to first argument."""
        lp = jnp.where(data_mask, model.log_likelihood(data, minibatch_size), 0.0)
        lp += log_prior_weight * model.log_prior()
        lp /= data_mask.sum()
        return -lp.sum()
    
    # --------------------------------------------------------------------------
    # Initialize model and optimizer
    model = model_klass.random_init(key_init, model_full_shape, model_core_shape)
    opt_state = optimizer.init(model)

    # Fit with default iter_callback
    model, outputs = fit_opt(model, data, data_mask,
                             objective_fn,
                             optimizer, opt_state,
                             n_iters)

    # Assert that loss is decreasing
    assert jnp.all(outputs.train_loss[1:] - outputs.train_loss[:-1] < 0)

@pytest.mark.parametrize("model_klass", [
    L2PenalizedMultinomialTucker,
    SimplexMultinomialTucker,
])
def test_fit_opt_2(model_klass):  
    optimizer=optax.adam(1e-1)
    model_full_shape=(10,9,8)
    model_core_shape=(5,4,3)
    scale=321
    n_iters=50

    key = jr.PRNGKey(42)
    key_model, key_sample, key_mask, key_init = jr.split(key, 4)

    # Generate some random data from a model
    true_model = model_klass.random_init(
        key_model, model_full_shape, model_core_shape, scale
    )
    data = true_model.sample(key_sample,)
    data_mask = true_model.random_mask(key_mask, frac=0.8)

    # --------------------------------------------------------------------------
    minibatch_size = 0
    log_prior_weight = 1

    # Define objective function
    def objective_fn(model) -> float:
        """Objective function, differentiate with respect to first argument."""
        lp = jnp.where(data_mask, model.log_likelihood(data, minibatch_size), 0.0)
        lp += log_prior_weight * model.log_prior()
        lp /= data_mask.sum()
        return -lp.sum()
    
    # --------------------------------------------------------------------------
    # Initialize model and optimizer
    model = model_klass.random_init(key_init, model_full_shape, model_core_shape, scale)
    opt_state = optimizer.init(model)

    # Fit with default iter_callback
    model, outputs = fit_opt(model, data, data_mask,
                             objective_fn,
                             optimizer, opt_state,
                             n_iters)

    # Assert that loss is decreasing
    assert jnp.all(outputs.train_loss[1:] - outputs.train_loss[:-1] < 0)