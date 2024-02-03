"""Three-mode Tucker decomposition base class."""

from typing import Optional, Sequence, Tuple, Union
from jax._src.prng import PRNGKeyArray
from jaxtyping import Array, Bool, Float, Integer

import warnings
import itertools

import equinox as eqx
import jax
import jax.numpy as jnp

warnings.filterwarnings("ignore")

class BaseTucker(eqx.Module):
    """Three-mode Tucker decomposition base class.

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

    event_ndim: int = 0
    tensor_mode: int = 3

    def __init__(self,
                 G: Float[Array, "k1 k2 k3"],
                 F1: Float[Array, "d1 k1"],
                 F2: Float[Array, "d2 k2"],
                 F3: Float[Array, "d3 k3"],
                 *args, **kwargs
    ):
        G_param, F1_param, F2_param, F3_param \
            = jax.tree_util.tree_map(self._inverse_transform, (G, F1, F2, F3))
        
        self.G_param = G_param
        self.F1_param = F1_param
        self.F2_param = F2_param
        self.F3_param = F3_param
        
    @classmethod
    def _transform(cls, param: Float[Array, "..."]) -> Float[Array, "..."]:
        """Transform unconstrained parameter to non-negative value."""
        raise NotImplementedError

    @classmethod
    def _inverse_transform(cls, val: Float[Array, "..."]) -> Float[Array, "..."]:
        """Transform non-negative value to unconstrained parameter."""
        raise NotImplementedError

    @property
    def full_shape(self,) -> Tuple[int, int, int]:
        """Shape of reconstructed tensor."""
        return tuple([len(f) for f in [self.F1_param, self.F2_param, self.F3_param]])
    
    @property
    def core_shape(self,) -> Tuple[int, int, int]:
        """Shape of core tensor."""
        return self.G_param.shape

    @property
    def batch_ndims(self,) -> int:
        """Number of batch (independent) dimensions in reconstructed tensor."""
        return len(self.full_shape) - self.event_ndim
    
    @property
    def params(self,) -> tuple:
        return self.G_param, self.F1_param, self.F2_param, self.F3_param

    @property
    def factors(self,) -> tuple:
        return jax.tree_util.tree_map(self._transform, self.params)
    
    def reconstruct(self,) -> Float[Array, "*full"]:
        """Reconstruct mean rate from parameterized decomposition."""

        G, F1, F2, F3 = self.factors
        
        tnsr = jnp.einsum('zc, abc-> abz', F3, G)
        tnsr = jnp.einsum('yb, abz-> ayz', F2, tnsr)
        tnsr = jnp.einsum('xa, ayz-> xyz', F1, tnsr)

        return tnsr

    @classmethod
    def sample_factors(cls,
                       key: PRNGKeyArray,
                       full_shape: Sequence[int],
                       core_shape: Sequence[int],
                       *args, **kwargs,
    ):
        """Randomly sample factors given full and core tensor shapes."""
        raise NotImplementedError
                    
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
        
        raise NotImplementedError
    
    def _fullbatch_log_likelihood(self,
                                  data: Integer[Array, "*full"],
    ) -> Float[Array, "*batch"]:
        """Compute full-batch Poisson log-likelihood under the current parameters."""
        
        raise NotImplementedError
    
    def log_likelihood(self,
                       data: Integer[Array, "*full"],
                       minibatch_size: int=0,
    ) -> Float[Array, "*batch"]:
        """Compute log likelihood of data under the model."""
        
        if minibatch_size > 0:
            raise NotImplementedError
        else:
            return self._fullbatch_log_likelihood(data)
        
    def log_prior(self,) -> float:
        """Compute log prior of parameter values."""

        raise NotImplementedError
    
    def log_prob(self,
                data: Integer[Array, "*full"],
                minibatch_size: int=0,
                log_prior_scale: float=1.) -> float:
        
        lp = self.log_likelihood(data, minibatch_size).sum()
        lp += log_prior_scale * self.log_prior()
        return lp
