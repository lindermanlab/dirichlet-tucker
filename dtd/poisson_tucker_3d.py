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


tfd = tfp.distributions
warnings.filterwarnings("ignore")


def softplus_forward(log_x: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
    """Elementwise transform unconstrained value to non-negative values via softplus."""
    return jax.nn.softplus(log_x)


def softplus_inverse(x: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
    """Elementwise transform non-negative values to unconstrained values via inverse-softplus.
    
    Catch overflow: when values are large (i.e. x >= 20), log(exp(x)-1) ~= x.
    """
    return jnp.where(x < 20, jnp.log(jnp.expm1(x)), x)


def softmax_forward(
        log_x: Float[Array, "*shape"],
        axis: Union[int, Sequence[int]]=-1,
) -> Float[Array, "*shape"]:
    """Transform unconstrained vector to satisfy simplex constraints along `axis`.
    
    `axis` can be an int or a tuple of ints.
    """
    return jax.nn.softmax(log_x, axis=axis)


def softmax_inverse(
        x: Float[Array, "*shape"],
        axis: Union[int, Sequence[int]]=-1,
) -> Float[Array, "*shape"]:
    """Transform non-negative vectors to unconstrained values centered along mean of `axis`.

    `axis` can be an int or a tuple of ints.
    """
    log_x = jnp.log(x)
    return log_x - log_x.mean(axis=axis, keepdims=True)


def poisson_log_prob(data: Integer[Array, "*shape"],
                     mean_rate: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
    """Compute Poisson log pdf of data under the given mean rate."""

    lp = data * jnp.log(mean_rate)
    lp -= mean_rate
    lp -= jax.scipy.special.gammaln(data + 1)

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

    """

    G_param: Float[Array, "k1 k2 k3"]
    F1_param: Float[Array, "d1 k1"]
    F2_param: Float[Array, "d2 k2"]
    F3_param: Float[Array, "d3 k3"]

    event_ndim: int = 0

    def __init__(self,
                 G: Float[Array, "k1 k2 k3"],
                 F1: Float[Array, "d1 k1"],
                 F2: Float[Array, "d2 k2"],
                 F3: Float[Array, "d3 k3"],
    ):
        G_param, F1_param, F2_param, F3_param \
            = self.params_from_vals(G, F1, F2, F3)
        
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
    def params_from_vals(cls, *vals) -> tuple:
        return jax.tree_util.tree_map(cls._inverse_transform, vals)


    @classmethod
    def random_init(cls,
                    key: PRNGKeyArray,
                    full_shape: Sequence[int],
                    core_shape: Sequence[int],
    ) -> "PoissonTucker":
        """Initialize a Poisson Tucker instance with random parameter values.

        (Unconstrained) parameters are sampled from Uniform(-5, 5).
        
        Parameters
        ----------
        key: PRNGKeyArray
        full_shape: Sequence, corresponds to reconstructed tensor shape (d1, d2, d3)
        core_shape: Sequence, corresponds to core tensor shape (k1, k2, k3)
        """

        tensor_mode = 3
        if len(full_shape) != tensor_mode:
            raise ValueError(f"Expecting {tensor_mode}-mode full tensor, but got {full_shape}.")

        if len(core_shape) != tensor_mode:
            raise ValueError(f"Expecting {tensor_mode}-mode core tensor, but got {core_shape}.")

        key, key_ = jr.split(key)
        G_param = jr.uniform(key, shape=tuple(core_shape), minval=-5, maxval=5)
        F_params = [
            jr.uniform(key_, shape=F_shape, minval=-5, maxval=5)
            for key_, F_shape in zip(jr.split(key_, tensor_mode), zip(full_shape, core_shape))
        ]

        params = jax.tree_util.tree_map(cls._transform, (G_param, *F_params))
        
        return cls(*params)


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
    

    def random_mask(self, key: PRNGKeyArray, frac: float=1.0) -> Bool[Array, "*batch"]:
        """Make random boolean mask to hold-in data.
        
        Parameters
        ----------
        key: PRNGKeyArray
        frac: Fraction of batch shape to hold-in. Default: 1.0, do not mask.

        Returns
        -------
        mask: boolean array

        """

        batch_shape = self.full_shape[:self.batch_ndims]
        return jr.bernoulli(key, frac, batch_shape)
    

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
        return poisson_log_prob(data, mean_rate)
    

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
    

class MultinomialTucker(PoissonTucker):
    """Base class for Multinomial Tucker model.
    
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

    event_ndim: int = 1
    scale: int = eqx.field(static=True)


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

        tensor_mode = 3
        if len(full_shape) != tensor_mode:
            raise ValueError(f"Expecting {tensor_mode}-mode tensor shape, but got {full_shape}.")

        if len(core_shape) != tensor_mode:
            raise ValueError(f"Expecting {tensor_mode}-mode core tensor, but got {core_shape}.")

        d1, d2, d3 = full_shape
        k1, k2, k3 = core_shape

        # Sample parameters from Dirichlet; initialied to be normalized correctly
        key_0, key_1, key_2, key_3 = jr.split(key, 4)
        G = jr.dirichlet(key_0, alpha=alpha*jnp.ones(k3), shape=(k1, k2))
        F1 = jr.dirichlet(key_1, alpha=alpha*jnp.ones(k1), shape=(d1,))
        F2 = jr.dirichlet(key_2, alpha=alpha*jnp.ones(k2), shape=(d2,))
        F3 = jr.dirichlet(key_3, alpha=alpha*jnp.ones(d3), shape=(k3,)).T
        
        return cls(G, F1, F2, F3, scale,)
    
    
    def reconstruct(self,):
        tnsr = super().reconstruct()
        return self.scale * tnsr
    

    def _fullbatch_log_likelihood(self,
                                  data: Integer[Array, "*full"],
                                 ) -> Float[Array, "*batch"]:
        """Compute full-batch Poisson log-likelihood under the current parameters."""
        
        ll = super()._fullbatch_log_likelihood(data)
        return ll.sum(axis=-1)
    
    

class ProjectedSimplexPoissonTucker():
    """Multinomial Tucker model with parameters normalized post-hoc."""

    pass


class L2PenalizedMultinomialTucker(MultinomialTucker):
    """Multinomial Tucker model with L2 penalty on simplex normality of parameters.
    
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
        
        G, F1, F2, F3 = self.factors
        
        # Event mode fibers of core should sum to 1
        penalty = (-0.5*jnp.linalg.norm(1 - G.sum(axis=2))**2).mean()

        # Rows of batch factors should sum to 1
        penalty += (-0.5*jnp.linalg.norm(1 - F1.sum(axis=1))**2).mean()
        penalty += (-0.5*jnp.linalg.norm(1 - F2.sum(axis=1))**2).mean()

        # Columns of event factors should sum to 1
        penalty += (-0.5*jnp.linalg.norm(1 - F3.sum(axis=0))**2).mean()

        return penalty


class SimplexMultinomialTucker(MultinomialTucker):
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

    # Axes of each parameter that is simplex-constrained
    _param_simplex_axes: tuple = (2, 1, 1, 0)
    
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


    @classmethod
    def params_from_vals(cls, *vals) -> tuple:
        return jax.tree_util.tree_map(cls._inverse_transform, vals, cls._param_simplex_axes)


    @property
    def factors(self,) -> tuple:
        return jax.tree_util.tree_map(self._transform, self.params, self._param_simplex_axes)