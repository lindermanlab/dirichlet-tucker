import pytest
import jax.numpy as jnp
import jax.random as jr
from jax import lax
import optax

from dtd.poisson_tucker_3d import PoissonTucker

def test_poisson_tucker(full_shape=(10,8,5),
                        core_shape=(4,3,2,)):
    
    key = jr.PRNGKey(0)

    key, key_ = jr.split(key)
    model = PoissonTucker.random_init(key_, full_shape, core_shape)

    assert model.full_shape == full_shape

    assert model.core_shape == core_shape

    key, key_ = jr.split(key)
    sample_shape = (2,)
    samples = model.sample(key_, sample_shape=sample_shape)
    assert samples.shape == (*sample_shape, *full_shape)

    lp = model.log_prob(samples)
    assert jnp.isfinite(lp)