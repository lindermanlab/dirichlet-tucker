import jax.numpy as jnp
import jax.random as jr

def calculate_minibatch_size(d1,d2,d3,k1,k2,k3,mem_gb,mem_frac=0.75):
    """Calculate minibatch size that maximizes available memory.
    
    Assumes 3D model with batch_dims (0,1).

    If setting mem_frac > 0.75 (the default fraction of GPU that JAX allocates
    on start up), make sure to adjust the environmental variable appropriately:
        XLA_PYTHON_CLIENT_MEM_FRACTION=<mem_frac>

    Note that the resulting minibatch size may not minimize the size of the last
    incomplete batch. To then identify a better minibatch size, you could sweep
    through minibatch sizes up to this returned value. For example:
        import plotly.express as px
        mods = [(m, (d1,d2) % m) for m in range(m_start,m)]
        mod_m, mod_val = list(zip(*mods))
        px.line(x=mod_m, y=mod_val)
    """
    m = (mem_gb*mem_frac) * (1024**3) / 4
    #m -= (d1*d2*d3)
    m /= (k1*k2*k3*d3)
    return int(m)

def calculate_memory(d1,d2,d3,k1,k2,k3,minibatch_size):
    """Calculate memory needed for a given minibatch size.
    
    Assumes 3D model with batch_dims (0,1).

    Note that this does NOT account for the memory needed to calculate the 
    incomplete batch (if drop_last=False), which would use _additional_ memory.
    """
    mem_gb = k1*k2*k3*d3*minibatch_size
    # m += d1*d2*d3
    mem_gb *= 4 / (1024**3)
    return mem_gb

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
        self.incomplete_size = len(self._indices) % self.minibatch_size
        self.has_incomplete = self.incomplete_size > 0

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