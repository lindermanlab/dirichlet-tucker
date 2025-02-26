import pytest

from math import prod
import jax.numpy as jnp
import jax.random as jr
import numpy as onp

from dtd.utils import (
    create_block_speckled_mask,
    get_jax_rng_state,
    get_numpy_rng_state,
    set_jax_rng_state,
    set_numpy_rng_state,
    ShuffleIndicesIterator,
)

def test_iterator(batch_shape=(8,4), 
                  minibatch_size=6,
                  seed=0):
    """Evaluate full-batch EM fit."""

    key = jr.PRNGKey(seed)
    
    # Instantiate the iterator
    indices_iterator = ShuffleIndicesIterator(key, batch_shape, minibatch_size)

    indices_iterator = iter(indices_iterator)
    
    # Compare two sets of produced indices
    batched_indices_1, remaining_indices_1 = next(indices_iterator)
    batched_indices_2, remaining_indices_2 = next(indices_iterator)
    
    assert remaining_indices_1.ndim == 2, \
        f"Expected `remaining_indices` to have shape ({minibatch_size}, {len(batch_shape)}), but got {remaining_indices_1.shape}."
    assert jnp.any(jnp.all(batched_indices_1==batched_indices_2, axis=-1)) == False, \
        f"Expected `batched_indices` from different iterations to be different"
    assert jnp.any(jnp.all(remaining_indices_1==remaining_indices_2, axis=-1)) == False, \
        f"Expected `batched_indices` from different iterations to be different"


@pytest.mark.parametrize("n_blocks, frac_mask", [(10, None), (None, 0.1)])
@pytest.mark.parametrize("exact", [True, False])
def test_create_block_speckled_mask(
    n_blocks, frac_mask, exact,
):
    """Test that that create_block_speckled_mask exactly recovers vanilla speckled masking."""

    batch_shape = (50,20)
    block_shape = 1
    buffer_sizes = 0

    rng = onp.random.default_rng(seed=0)
    n_blocks_ = n_blocks if n_blocks is not None else int(frac_mask * prod(batch_shape))
    if exact:
        refr = onp.concatenate([
            onp.ones(n_blocks_, dtype=bool),
            onp.zeros(prod(batch_shape)-n_blocks_, dtype=bool)
        ])
        refr = rng.permutation(refr)
        refr = refr.reshape(*batch_shape)
    else:
        p = n_blocks_ / prod(batch_shape)
        refr = rng.binomial(1, p, size=batch_shape)

    # Reset RNG and check
    rng = onp.random.default_rng(seed=0)
    mask, buffer = create_block_speckled_mask(
        rng, batch_shape, block_shape, buffer_sizes,
        n_blocks=n_blocks, frac_mask=frac_mask, exact=exact
    )

    assert buffer.sum() == 0
    assert onp.all(refr==mask)


def test_jax_rng_state(batch_shape=(8,4), 
                  minibatch_size=6,
                  seed=0):
    """Evaluate {get,set}_jax_rng_state."""

    key = jr.key(0)
    rng_state = get_jax_rng_state(key)

    # Ensure the keys are the same
    recovered_key = set_jax_rng_state(rng_state)
    assert jnp.all(key==recovered_key)

    # Ensure that they produce the same outputs
    arr = jr.uniform(key, 5)
    recovered_arr = jr.uniform(recovered_key, 5)
    assert all(arr==recovered_arr)


def test_numpy_rng_state(batch_shape=(8,4), 
                  minibatch_size=6,
                  seed=0):
    """Evaluate {get,set}_jax_rng_state."""

    rng = onp.random.default_rng(seed=0)
    rng_state = get_numpy_rng_state(rng)
    recovered_rng = set_numpy_rng_state(rng_state)

    # Ensure that they produce the same outputs
    arr = rng.uniform(size=5)
    recovered_arr = recovered_rng.uniform(size=5)
    assert all(arr==recovered_arr)
