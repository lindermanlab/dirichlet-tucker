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

from dtd.base_tucker_3d import SoftplusTucker, SoftmaxTucker
from dtd.utils import softmax_forward, softmax_inverse, softplus_forward, softplus_inverse

tfd = tfp.distributions
warnings.filterwarnings("ignore")


class ScaledPoissonTucker(SoftplusTucker):
    """Base class for scaled Poisson Tucker model.

    Given core tensor G and factors F1, F2, F3, the Tucker reconstruction
    gives the elementwise mean rate of a Poisson distribution:
        X_mnp ~ Poisson( [[G; F1 F2 F3]]_mnp )
    where [[G; ...]] denotes the Tucker decomposition in "Kolda notation" and
    for m=(1,...,d1), n=(1,...,d2), and p=(1,...,d3).

    All shape annotations assume `batch_dims=(d1, d2)`, `event_dims=(d3,)`.
    
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
    event_ndim: int, number of event dimensions = 1
        Dimension of a single draw from the distribution. A property of the
        distribution; Multivariate distribution event_ndim = 1.
    normalized_axes: tuple
        Axis of each parameter that is constrained / penalized to be one
    scale: int
        Number of trials in a Multinomial distribution. Samples from the
        distribution sum to this value along the simplex-normalized axes.

    """

    event_ndim: int = 1
    scale: int = eqx.static_field()  # Replace with `eqx.field(static=True)` for equinox version >= 0.10.5

    def __init__(self,
                 G: Float[Array, "k1 k2 k3"],
                 F1: Float[Array, "d1 k1"],
                 F2: Float[Array, "d2 k2"],
                 F3: Float[Array, "d3 k3"],
                 scale: int,
                 *args, **kwargs
    ):
        super().__init__(G, F1, F2, F3)
        self.scale = int(scale)

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
                    scale: int,
                    alpha: float=0.1
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
            Shape of Gamma prior. Default: 0.1
        """

        factors = super().sample_factors(key, full_shape, core_shape, alpha=alpha)
        
        return cls(*factors, scale,)

    def _fullbatch_log_likelihood(self,
                                  data: Integer[Array, "*full"],
                                 ) -> Float[Array, "*batch"]:
        """Compute full-batch Poisson log-likelihood under the current parameters."""
        
        mean_rate = self.reconstruct()
        ll = tfd.Poisson(rate=self.scale*mean_rate).log_prob(data)
        return ll.sum(axis=-1)

    
class ProjectedPoissonTucker(ScaledPoissonTucker):
    """A vanilla Poisson Tucker model with a scale parameter.

    Scale parameter is only used in evaluating multinomial log likelihood.
    """

    def _fullbatch_log_likelihood(self,
                                  data: Integer[Array, "*full"],
                                 ) -> Float[Array, "*batch"]:
        """Compute full-batch Poisson log-likelihood under the current parameters.
        
        This is equivalent to a vanilla Poisson Tucker model.
        """
        
        mean_rate = self.reconstruct()
        ll = tfd.Poisson(rate=mean_rate).log_prob(data)

        # summing to maintain consistent shape with others
        # should not really matter, because it will later be sum-reduced again
        return ll.sum(axis=-1)
        

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

    normalized_axes: tuple = (2, 1, 1, 0)
    
    def log_prior(self,) -> float:
        """Apply L2 regularization on parameters to be simplex constrained."""
        
        penalty = jnp.asarray(
            jax.tree_util.tree_map(
                lambda fctr, axis: (-0.5*(1-jnp.linalg.norm(fctr, axis=axis))**2).sum(),
                self.factors, self.normalized_axes
            )
        )
        
        return penalty.sum()


class SimplexPoissonTucker(SoftmaxTucker, ScaledPoissonTucker):
    """Multinomial Tucker model with simplex-constrained factors.

    Inherits from SoftmaxTucker, such that factors are transformed from
    unconstrained to constrained simplex space via softmax transform,
    _not_ softplus transform used in previous models.
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

    normalized_axes: tuple = (2, 1, 1, 0)

    def __init__(self,
                 G: Float[Array, "k1 k2 k3"],
                 F1: Float[Array, "d1 k1"],
                 F2: Float[Array, "d2 k2"],
                 F3: Float[Array, "d3 k3"],
                 scale: int,
                 *args, **kwargs
    ):
        super().__init__(G, F1, F2, F3)
        self.scale = int(scale)