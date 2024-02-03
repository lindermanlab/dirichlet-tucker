"""Three-mode Dirichlet Tucker decomposition.

Implemented as an equinox module to support performing gradient descent.

See all: model3d -- implemented in more pure functional style, for handling EM
"""

from typing import Optional, Sequence, Tuple, Union
from jax._src.prng import PRNGKeyArray
from jaxtyping import Array, Bool, Float, Integer

import warnings
import itertools

from tqdm.auto import trange

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from tensorflow_probability.substrates import jax as tfp

from dtd.poisson_tucker_3d import BaseTucker
from dtd.utils import softplus_forward, softplus_inverse

tfd = tfp.distributions
warnings.filterwarnings("ignore")


class DirichletTucker(BaseTucker):
    """Three-mode Dirichlet Tucker decomposition class.

    All shape annotations assume `batch_dims=(d1, d2)`, `event_dims=(d3,)`.

    Attributes
    ----------
    G_param: float array, shape (k1, k2, k3)
    F1_param: float array, shape (d1, k1)
    F2_param: float array, shape (d2, k2)
    F3_param: float array, shape (d3, k3)
        Unconstrained parameterization of the core tensor and factor matrices
        of a 3-mode non-negative Tucker decomposition.
    
    Static attributes
    -----------------
    event_ndim: int
        Dimension of a single draw from the distribution. A property of the
        distribution; Poisson distribution event_ndim = 0.
    tensor_mode: int
        Tensor mode, hard-coded to 3 modes.

    """

    G_param: Float[Array, "k1 k2 k3"]
    F1_param: Float[Array, "d1 k1"]
    F2_param: Float[Array, "d2 k2"]
    F3_param: Float[Array, "d3 k3"]

    event_ndim: int = 1
    tensor_mode: int = 3
    normalized_axes: tuple = (2, 1, 1, 0)
    scale: int = eqx.static_field()  # Replace with `eqx.field(static=True)` for equinox version >= 0.10.5
    alpha: alpha = eqx.static_field()  # Replace with `eqx.field(static=True)` for equinox version >= 0.10.5

    def __init__(self,
                 G: Float[Array, "k1 k2 k3"],
                 F1: Float[Array, "d1 k1"],
                 F2: Float[Array, "d2 k2"],
                 F3: Float[Array, "d3 k3"],
                 scale: int,
                 alpha: float = 1.1,
                 *args, **kwargs
    ):
        super().__init__(G, F1, F2, F3)
        
        self.scale = scale
        self.alpha = alpha
        
    @classmethod
    def _transform(cls, param: Float[Array, "..."]) -> Float[Array, "..."]:
        """Transform unconstrained parameter to non-negative value."""
        return softplus_forward(param)

    @classmethod
    def _inverse_transform(cls, val: Float[Array, "..."]) -> Float[Array, "..."]:
        """Transform non-negative value to unconstrained parameter."""
        return softplus_inverse(val)

    @classmethod
    def sample_factors(cls,
                       key: PRNGKeyArray,
                       full_shape: Sequence[int],
                       core_shape: Sequence[int],
                       alpha: float=1.1,
                       *args, **kwargs
    ):
        """Sample parameters from Dirichlet prior distirbutions with concentration alpha."""
        
        if len(full_shape) != cls.tensor_mode:
            raise ValueError(f"Expecting {cls.tensor_mode}-mode full tensor, but got {full_shape}.")

        if len(core_shape) != cls.tensor_mode:
            raise ValueError(f"Expecting {cls.tensor_mode}-mode core tensor, but got {core_shape}.")

        d1, d2, d3 = full_shape
        k1, k2, k3 = core_shape

        key_0, key_1, key_2, key_3 = jr.split(key, 4)
        G = jr.dirichlet(key_0, alpha=alpha*jnp.ones(k3), shape=(k1, k2))
        F1 = jr.dirichlet(key_1, alpha=alpha*jnp.ones(k1), shape=(d1,))
        F2 = jr.dirichlet(key_2, alpha=alpha*jnp.ones(k2), shape=(d2,))
        F3 = jr.dirichlet(key_3, alpha=alpha*jnp.ones(d3), shape=(k3,)).T

        return G, F1, F2, F3
                    
    @classmethod
    def random_init(cls,
                    key: PRNGKeyArray,
                    full_shape: Sequence[int],
                    core_shape: Sequence[int],
                    *args, **kwargs,
    ):
        """Instantiate class with randomly initialized factors."""
        
        factors = cls.sample_factors(key, full_shape, core_shape, *args, **kwargs)

        return cls(*factors, *args, **kwargs)
    
    def sample(self,
               key: PRNGKeyArray,
               sample_shape: tuple=(),
    ) -> Integer[Array, "..."]:
        """Sample a data tensor from the parameterized decompositions.
        
        Parameters
        ----------
        key: PRNGKeyArray
        sample_shape: tuple, result shape

        Returns
        -------
        samples: shape (*sample_shape, *full)

        """

        event_probs = self.reconstruct()
        return tfd.Multinomial(self.scale, probs=event_probs).sample(sample_shape, key)
    
    def _fullbatch_log_likelihood(self,
                                  data: Integer[Array, "*full"],
    ) -> Float[Array, "*batch"]:
        """Compute full-batch Poisson log-likelihood under the current parameters."""
        
        event_probs = self.reconstruct()
        return tfd.Multinomial(self.scale, probs=event_probs).log_prob(data)
    
    def log_prior(self,) -> float:
        """Compute log prior of parameter values."""
        
        G, F1, F2, F3 = self.factors

        d1, d2, d3 = self.full_shape
        k1, k2, k3 = self.core_shape

        # (k1, k2)
        lp = tfd.Dirichlet(self.alpha * jnp.ones(k3)).log_prob(G).sum()  # (k1, k2)
        lp += tfd.Dirichlet(self.alpha * jnp.ones(k1)).log_prob(F1).sum() # (d1,)
        lp += tfd.Dirichlet(self.alpha * jnp.ones(k2)).log_prob(F2).sum() # (d2,)
        lp += tfd.Dirichlet(self.alpha * jnp.ones(d3)).log_prob(F3.T).sum() # (k3,)
                
        return lp