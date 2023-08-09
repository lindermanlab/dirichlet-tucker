import jax.numpy as jnp
import jax.random as jr

from dtd.model import DirichletTuckerDecomp

def make_random_mask(key, shape, train_frac=0.8):
    """Make binary mask to split data into a train (1) and validation (0) sets."""
    return jr.bernoulli(key, train_frac, shape)

# Generate a random tensor of counts
shape = (50, 25, 20)
rank = (10, 8, 5)
batch_ndims = 2
total_counts = 1000
model = DirichletTuckerDecomp(total_counts, *rank)

key = jr.PRNGKey(0)
key_true, key_data, key_mask, key_init = jr.split(key, 4)

# Generate observations from an underlying DTD model
true_params = model.sample_params(key_true, *shape)
X = model.sample_data(key_data, true_params)
mask = make_random_mask(key_mask, X.shape[:batch_ndims])

def test_fit():
    """Evaluate full-batch EM fit."""
    
    n_iters = 100
    
    # Initialize a different set of parameters and fit
    init_params = model.sample_params(key_init, *shape)

    fitted_params, lps = model.fit(X, mask, init_params, n_iters)

    # Check that log probabilities are monotonically increasing
    assert jnp.all(jnp.diff(lps) > 0), \
        f'Expected lps to be monotonically increasing.'