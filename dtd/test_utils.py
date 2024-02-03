import pytest
import jax.numpy as jnp
import jax.random as jr

from dtd.utils import ShuffleIndicesIterator

def test_iterator(batch_shape=(8,4), 
                  minibatch_size=6,
                  seed=0
                  ):
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