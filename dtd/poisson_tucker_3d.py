"""Three-mode Poisson Tucker model variants"""

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

from dtd.utils import softmax_forward, softmax_inverse, softplus_forward, softplus_inverse

tfd = tfp.distributions
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

    @classmethod
    def _transform(cls, param: Float[Array, "..."]) -> Float[Array, "..."]:
        """Transform unconstrained parameter to non-negative value."""
        raise NotImplementedError

    @classmethod
    def _inverse_transform(cls, val: Float[Array, "..."]) -> Float[Array, "..."]:
        """Transform non-negative value to unconstrained parameter."""
        raise NotImplementedError

    @classmethod
    def sample_factors(cls,
                       key: PRNGKeyArray,
                       full_shape: Sequence[int],
                       core_shape: Sequence[int],
                       *args, **kwargs,
    ):
        raise NotImplementedError
                    
    @classmethod
    def random_init(cls,
                    key: PRNGKeyArray,
                    full_shape: Sequence[int],
                    core_shape: Sequence[int],
                    *args, **kwargs,
    ):
        
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
        
    def log_prior(self,) -> Float[Array, "#batch"]:
        """Compute log prior of parameter values."""

        raise NotImplementedError
    
    def log_prob(self,
                data: Integer[Array, "*full"],
                minibatch_size: int=0,
                log_prior_scale: float=1.) -> Float[Array, "*batch"]:
        
        lp = self.log_likelihood(data, minibatch_size)
        lp += log_prior_scale * self.log_prior()
        return lp

class PoissonTucker(eqx.Module):
    """Three-mode Poisson Tucker class.

    Given core tensor G and factors F1, F2, F3, the Tucker reconstruction
    gives the elementwise mean rate of a Poisson distribution:
        X_mnp ~ Poisson( [[G; F1 F2 F3]]_mnp )
    where [[G; ...]] denotes the Tucker decomposition in "Kolda notation" and
    for m=(1,...,d1), n=(1,...,d2), and p=(1,...,d3).

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
                       alpha: float=0.1,
    ):

        if len(full_shape) != cls.tensor_mode:
            raise ValueError(f"Expecting {cls.tensor_mode}-mode full tensor, but got {full_shape}.")

        if len(core_shape) != cls.tensor_mode:
            raise ValueError(f"Expecting {cls.tensor_mode}-mode core tensor, but got {core_shape}.")
            
        key, key_ = jr.split(key)
        G = jr.gamma(key, alpha, shape=tuple(core_shape))
        Fs = [
            jr.gamma(key_, alpha, shape=F_shape)
            for key_, F_shape in zip(jr.split(key_, cls.tensor_mode), zip(full_shape, core_shape))
        ]

        return G, *Fs
                    
    @classmethod
    def random_init(cls,
                    key: PRNGKeyArray,
                    full_shape: Sequence[int],
                    core_shape: Sequence[int],
                    alpha: float=0.1,
    ) -> "PoissonTucker":
        """Initialize a Poisson Tucker instance with random parameter values.

        (Unconstrained) parameters are sampled from Uniform(-5, 5).
        
        Parameters
        ----------
        key: PRNGKeyArray
        full_shape: Sequence, corresponds to reconstructed tensor shape (d1, d2, d3)
        core_shape: Sequence, corresponds to core tensor shape (k1, k2, k3)
        alpha: positive float, shape parameter of gamma distribution
        """

        factors = cls.sample_factors(key, full_shape, core_shape, alpha=alpha)

        return cls(*factors)

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
        
        mean_rate = self.reconstruct()

        return jr.poisson(key, mean_rate, shape=(*sample_shape, *self.full_shape))
    
    def _fullbatch_log_likelihood(self,
                                  data: Integer[Array, "*full"],
    ) -> Float[Array, "*batch"]:
        """Compute full-batch Poisson log-likelihood under the current parameters."""
        
        mean_rate = self.reconstruct()
        return tfd.Poisson(rate=mean_rate).log_prob(data)
    
    def log_likelihood(self,
                       data: Integer[Array, "*full"],
                       minibatch_size: int=0,
    ) -> Float[Array, "*batch"]:
        """Compute log likelihood of data under the model."""
        
        if minibatch_size > 0:
            raise NotImplementedError
        else:
            return self._fullbatch_log_likelihood(data)
        
    def log_prior(self,) -> Float[Array, "#batch"]:
        """Compute log prior of parameter values."""

        return jnp.array(0)
    
    def log_prob(self,
                data: Integer[Array, "*full"],
                minibatch_size: int=0,
                log_prior_scale: float=1.) -> Float[Array, "*batch"]:
        
        lp = self.log_likelihood(data, minibatch_size)
        lp += log_prior_scale * self.log_prior()
        return lp
    

class ScaledPoissonTucker(PoissonTucker):
    """Base class for scaled Poisson Tucker model.
    
    Attributes
    ----------
    G_param: ndarray, shape (k1, k2, k3)
    F1_param: ndarray, shape (d1, k1)
    F2_param: ndarray, shape (d2, k2)
    F3_param: ndarray, shape (d3, k3)
        Unconstrained parameterization of the core tensor and factor matrices
        of a 3-mode non-negative Tucker decomposition.    

    Static attributes
    -----------------
    event_ndim: int, number of event dimensions.
        Dimension of a single draw from the distribution. A property of the
        distribution; Multivariate distribution event_ndim = 1.
    normalized_axes: tuple
        Axis of each parameter that is constrained / penalized to be one
    scale: int
        Number of trials in a Multinomial distribution. Samples from the
        distribution sum to this value along the simplex-normalized axes.

    """

    event_ndim: int = 1
    normalized_axes: tuple = (2, 1, 1, 0)
    scale: int = eqx.static_field()  # Replace with `eqx.field(static=True)` for equinox version >= 0.10.5

    def __init__(self,
                 G: Float[Array, "k1 k2 k3"],
                 F1: Float[Array, "d1 k1"],
                 F2: Float[Array, "d2 k2"],
                 F3: Float[Array, "d3 k3"],
                 scale: int,
                 /
    ):
        super().__init__(G, F1, F2, F3)
        self.scale = int(scale)

    @classmethod
    def random_init(cls,
                    key: PRNGKeyArray,
                    full_shape: Sequence[int],
                    core_shape: Sequence[int],
                    scale: int,
                    /,
                    alpha: float=0.9
    ):
        """Initialize parameters as draws a Dirichlet prior.

        These inital parameters are inherently simplex-normalized.
        
        Parameters
        ----------
        key: PRNGKeyArray
        full_shape: Sequence, corresponds to reconstructed tensor shape (d1, d2, d3)
        core_shape: Sequence, corresponds to core tensor shape (k1, k2, k3)
        scale: int
            Number of trials in a Multinomial distribution. 
        alpha: float
            Concentration of Dirichlet prior. Default: 0.9, sparse prior.
        """

        factors = super().sample_factors(key, full_shape, core_shape, alpha=alpha)
        
        return cls(*factors, scale,)
    
    def reconstruct(self,):
        tnsr = super().reconstruct()
        return self.scale * tnsr
    
    def _fullbatch_log_likelihood(self,
                                  data: Integer[Array, "*full"],
                                 ) -> Float[Array, "*batch"]:
        """Compute full-batch Poisson log-likelihood under the current parameters."""
        
        ll = super()._fullbatch_log_likelihood(data)
        return ll.sum(axis=-1)
    
    
class ProjectedPoissonTucker(ScaledPoissonTucker):
    """A vanilla Poisson Tucker model with a scale parameter.

    Scale parameter is only used in evaluating multinomial log likelihood.
    """

    def reconstruct(self,):
        # Inelegant, but this returns the (unscaled) mean rate.
        # Ideally, we would like to just directly call PoissonTucker.reconstruct()
        tnsr = super().reconstruct()
        return tnsr / self.scale


class L2PenalizedPoissonTucker(ScaledPoissonTucker):
    """Scaled Poisson Tucker Tucker model with L2 penalty on simplex normality of parameters.
    
    The L2 penalty is applied by overloading the `log_prior` function. In this
    sense, this model can also be viewed as placing a Gaussian prior over the
    the event axes with a mean sum of 1.

    Attributes
    ----------
    G_param: ndarray, shape (k1, k2, k3)
    F1_param: ndarray, shape (d1, k1)
    F2_param: ndarray, shape (d2, k2)
    F3_param: ndarray, shape (d3, k3)
        Unconstrained parameterization of the core tensor and factor matrices
        of a 3-mode non-negative Tucker decomposition.    

    Static attributes
    -----------------
    event_ndim: int, number of event dimensions.
        Dimension of a single draw from the distribution. A property of the
        distribution; Multivariate distribution event_ndim = 1.
    scale: int
        Number of trials in a Multinomial distribution. Samples from the
        distribution sum to this value along the simplex-normalized axes.

    """
    
    def log_prior(self,) -> Float[Array, "#batch"]:
        """Apply L2 regularization on parameters to be simplex constrained."""
        
        penalty = jnp.asarray(
            jax.tree_util.tree_map(
                lambda fctr, axis: (-0.5*jnp.linalg.norm(1 - fctr.sum(axis=axis))**2).mean(),
                self.factors, self.normalized_axes
            )
        )
        
        return penalty.sum()


class SimplexPoissonTucker(ScaledPoissonTucker):
    """Multinomial Tucker model with simplex-constrained factors.

    Factors transformed from uncostrained to constrained simplex space via
    softmax transform, _not_ softplus transform used in previous models.
    This transform ensures that the parameter values are positive _and_
    sum to 1, in contrast to other models using softplus transform which only
    guarantees that the parameter values are nonnegative.

    Attributes
    ----------
    G_param: ndarray, shape (k1, k2, k3)
    F1_param: ndarray, shape (d1, k1)
    F2_param: ndarray, shape (d2, k2)
    F3_param: ndarray, shape (d3, k3)
        Unconstrained parameterization of the core tensor and factor matrices
        of a 3-mode non-negative Tucker decomposition.    

    Static attributes
    -----------------
    event_ndim: int, number of event dimensions.
        Dimension of a single draw from the distribution. A property of the
        distribution; Multivariate distribution event_ndim = 1.
    scale: int
        Number of trials in a Multinomial distribution. Samples from the
        distribution sum to this value along the simplex-normalized axes.
    """

    def __init__(self,
                 G: Float[Array, "k1 k2 k3"],
                 F1: Float[Array, "d1 k1"],
                 F2: Float[Array, "d2 k2"],
                 F3: Float[Array, "d3 k3"],
                 scale,
    ):
        G_param, F1_param, F2_param, F3_param \
            = jax.tree_util.tree_map(self._inverse_transform, (G, F1, F2, F3), self.normalized_axes)
        
        self.G_param = G_param
        self.F1_param = F1_param
        self.F2_param = F2_param
        self.F3_param = F3_param

        self.scale = scale

    @classmethod
    def _transform(cls,
                   param: Float[Array, "..."],
                   axis: Optional[int]=-1,
                   ) -> Float[Array, "..."]:
        """Transform unconstrained parameter to non-negative value."""
        return softmax_forward(param, axis=axis)

    @classmethod
    def _inverse_transform(cls,
                           val: Float[Array, "..."],
                           axis: Optional[int]=-1,
                           ) -> Float[Array, "..."]:
        """Transform non-negative value to unconstrained parameter."""
        return softmax_inverse(val, axis=axis)

    @property
    def factors(self,) -> tuple:
        return jax.tree_util.tree_map(self._transform, self.params, self.normalized_axes)