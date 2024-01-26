import pytest
import jax.numpy as jnp
import jax.random as jr
from jax import lax
import optax

from dtd.poisson_tucker_3d import PoissonTucker

from dtd.sgd import setup_optimization, fit_opt

def test_fit_opt(
    model_klass=PoissonTucker,
    model_full_shape=(10,9,8),
    model_core_shape=(5,4,3),
    n_iters=50,
):
    key = jr.PRNGKey(42)
    key_init, key_sample, key_mask, key_opt = jr.split(key, 4)

    # Generate some random data from a model
    true_model = model_klass.random_init(
        key_init, model_full_shape, model_core_shape
    )
    data = true_model.sample(key_sample,)
    data_mask = true_model.random_mask(key_mask, frac=0.8)

    # Set up optimizer
    optimizer = optax.adam(1e-1)
    init_fn, step_fn = setup_optimization(
        optimizer,
        model_klass,
        model_full_shape,
        model_core_shape,
    )

    model, outputs = fit_opt(key_opt, init_fn, step_fn, data, data_mask, n_iters)

    # Assert that loss is decreasing
    assert jnp.all(outputs[1:] - outputs[:-1] < 0)