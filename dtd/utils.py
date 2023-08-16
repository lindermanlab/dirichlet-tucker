import jax.numpy as jnp
import jax.random as jr

class ShuffleIndicesIterator():
    """Custom Iterator that produces minibatches of shuffled indices.
    
    Parameters
        key: PRNGKey
            Used to shuffle the indices at each iteration.
        batch_shape: tuple
            Shape of target array's batch dimensions
        minibatch_size: int
            Number of indices per minibatch
    
    Outputs
        batched_indices:   shape (n_batches, minibatch_size, batch_ndims)
        remaining_indices: shape (n_remaining, batch_ndims)
    """

    def __init__(self, key, batch_shape, minibatch_size):
        self.key = key

        # Sequentially enumerated indices, shape (n_samples, batch_ndims)
        # where n_samples = prod(*batch_shape) and batch_ndims = len(batch_shape)
        self._indices = jnp.indices(batch_shape).reshape(len(batch_shape), -1).T

        self.minibatch_size = minibatch_size
        self.n_complete = len(self._indices) // self.minibatch_size
        self.has_incomplete = (len(self._indices) % self.minibatch_size) > 0

    def __iter__(self):
        return self
    
    def __next__(self):
        # Shuffle indices
        _, self.key = jr.split(self.key)
        indices = jr.permutation(self.key, self._indices, axis=0)

        # Split indices into complete minibatch sets and an incomplete set
        batched_indices, remaining_indices \
            = jnp.split(indices, (self.n_complete*self.minibatch_size,))
        batched_indices \
            = batched_indices.reshape(self.n_complete, self.minibatch_size, -1)

        return batched_indices, remaining_indices