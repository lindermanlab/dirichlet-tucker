import pytest
import jax.numpy as jnp
import jax.random as jr
from jax import lax
import optax

from dtd.poisson_tucker_3d import (
    PoissonTucker,
    ProjectedPoissonTucker,
    L2PenalizedPoissonTucker,
    SimplexPoissonTucker
)

from dtd.dirichlet_tucker_3d import DirichletTucker

def test_poisson_tucker(full_shape=(10,8,5),
                        core_shape=(4,3,2,)):
    
    key = jr.PRNGKey(0)

    key_init, key_sample = jr.split(key)
    model = PoissonTucker.random_init(key_init, full_shape, core_shape)

    assert model.full_shape == full_shape

    assert model.core_shape == core_shape

    sample_shape = (2,)
    samples = model.sample(key_sample, sample_shape=sample_shape)
    assert samples.shape == (*sample_shape, *full_shape)

    lp = model.log_prob(samples)
    assert jnp.all(jnp.isfinite(lp))

@pytest.mark.parametrize("model_klass", [
    ProjectedPoissonTucker,
    L2PenalizedPoissonTucker,
    SimplexPoissonTucker,
    DirichletTucker
])
def test_tucker_models_with_scale_param(model_klass):
    
    key = jr.PRNGKey(0)
    key_init, key_sample = jr.split(key)

    full_shape = (10,8,5)
    core_shape = (4,3,2,)
    scale = 123
    model = model_klass.random_init(key_init, full_shape, core_shape, scale)

    assert model.full_shape == full_shape

    assert model.core_shape == core_shape

    sample_shape = (2,)
    samples = model.sample(key_sample, sample_shape=sample_shape)
    assert samples.shape == (*sample_shape, *full_shape)

    lp = model.log_prob(samples)
    assert jnp.all(jnp.isfinite(lp))