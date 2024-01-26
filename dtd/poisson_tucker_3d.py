"""Three-mode Poisson Tucker model variants"""

from typing import Optional, Sequence, Tuple, Union
from jaxtyping import Array, Bool, Float, Integer
from jax._src.prng import PRNGKeyArray

import warnings
import time
import itertools

from tqdm.auto import trange

import equinox as eqx
import jax
from jax import jit, lax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
from jax.tree_util import tree_map
from tensorflow_probability.substrates import jax as tfp


tfd = tfp.distributions
warnings.filterwarnings("ignore")

def softplus_forward(log_x):
    """Elementwise transform unconstrained value to non-negative values via softplus."""
    return jax.nn.softplus(log_x)

def softplus_inverse(x):
    """Elementwise transform non-negative values to unconstrained values via inverse-softplus.
    
    Catch overflow: when values are large (i.e. x >= 20), log(exp(x)-1) ~= x.
    """
    return jnp.where(x < 20, jnp.log(jnp.expm1(x)), x)

def poisson_log_prob(data: Integer[Array, "*full"],
                     mean_rate: Float[Array, "*full"]) -> Float[Array, "*full"]:
    """Compute Poisson log pdf of data under the given mean rate."""

    lp = data * jnp.log(mean_rate)
    lp -= mean_rate
    lp -= jsp.special.gammaln(data + 1)

    return lp

class PoissonTucker(eqx.Module):
    """Three-mode Poisson Tucker class.

    Given core tensor G and factors F1, F2, F3, the Tucker reconstruction
    gives the elementwise mean rate of a Poisson distribution:
        X_mnp ~ Poisson( [[G; F1 F2 F3]]_mnp )
    where [[G; ...]] denotes the Tucker decomposition in "Kolda notation" and
    for m=(1,...,d1), n=(1,...,d2), and p=(1,...,d3).

    All shape annotations assume `batch_dims=(d1, d2)`, `event_dims=(d3,)`.

    Parameters
    ----------
    *params: Sequence[Float[Array. "..."]]
    event_ndims: int, number of event dimensions. Default: 1
        Only event_ndims = 1 is currently supported. Parameter exposed for
        documentation purposes and the sake of being explicit.
    """

    G_param: Float[Array, "k1 k2 k3"]
    F1_param: Float[Array, "d1 k1"]
    F2_param: Float[Array, "d2 k2"]
    F3_param: Float[Array, "d3 k3"]
    event_ndims: int = eqx.static_field()
    

    def __init__(self, *params, event_ndims: int=1):
        if event_ndims != 1:
            raise ValueError(
                f"Only event_ndims= 1 is currently supported, but got {event_ndims}." \
                "Dynamic event dimensioning may be explored in the future. For now," \
                "create a new class and handle event dimensioning manually."
            )
        
        self.G_param = params[0]
        self.F1_param = params[1]
        self.F2_param = params[2]
        self.F3_param = params[3]

        self.event_ndims = event_ndims
    

    @classmethod
    def random_init(cls,
                    key: PRNGKeyArray,
                    full_shape: Sequence[int],
                    core_shape: Sequence[int],
                    /,
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
            raise ValueError(f"Expecting len(full_shape)={tensor_mode}, but got full_shape={full_shape}.")

        if len(core_shape) != tensor_mode:
            raise 

        key, key_ = jr.split(key)
        G_param = jr.uniform(key, shape=tuple(core_shape), minval=-5, maxval=-5)
        F_params = [
            jr.uniform(key_, shape=F_shape, minval=-5, maxval=-5)
            for key_, F_shape in zip(jr.split(key_, tensor_mode), zip(full_shape, core_shape))
        ]
        
        return cls(G_param, *F_params)

    @property
    def full_shape(self,) -> Tuple[int, int, int]:
        """Shape of reconstructed tensor."""
        return tuple([len(f) for f in [self.F1_param, self.F2_param, self.F3_param]])
    

    @property
    def core_shape(self,) -> Tuple[int, int, int]:
        """Shape of core tensor."""
        return self.G_param.shape
    
    
    @property
    def event_ndims(self,) -> int:
        """Number of event dimensions in reconstructed tensor. This is hard-coded in."""
        return 0


    @property
    def batch_ndims(self,) -> int:
        """Number of batch (independent) dimensions in reconstructed tensor."""
        return len(self.full_shape) - self.event_ndims
    

    @property
    def params(self,) -> tuple:
        return self.G_param, self.F1_param, self.F2_param, self.F3_param
    
    def _transform(self, param: Float[Array, "..."]) -> Float[Array, "..."]:
        """Transform unconstrained parameter to non-negative value."""
        return softplus_forward(param)
    

    def _inverse_transform(self, val: Float[Array, "..."]) -> Float[Array, "..."]:
        """Transform non-negative value to unconstrained parameter."""
        return softplus_inverse(val)
    

    def reconstruct(self,) -> Float[Array, "*full"]:
        """Reconstruct mean rate from parameterized decomposition."""

        G, F1, F2, F3 = tree_map(self._transform, self.params)
        
        tnsr = jnp.einsum('zc, abc-> abz', F3, G)
        tnsr = jnp.einsum('yb, abz-> ayz', F2, tnsr)
        tnsr = jnp.einsum('xa, ayz-> xyz', F1, tnsr)

        return tnsr

        
    def sample(self,
               key: PRNGKeyArray,
               sample_shape: tuple=()) -> Integer[Array, "..."]:
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
                log_prior_scale: float=1.) -> Float[Array, "*event"]:
        
        lp = self.log_likelihood(data, minibatch_size)
        lp += log_prior_scale * self.log_prior()
        return lp
    
class ProjectedConvexPoissonTucker():
    pass

class SoftConvexPoissonTucker():
    """Convex Poisson Tucker model with factors penalized (L2 norm) to convex.
    """
    pass

class HardConvexPoissonTucker():
    """Convex Poisson Tucker model with simplex constrained factors.

    Factors transformed from uncostrained to constrained simplex space via
    softmax transform (not softplus, and not softplus_centered)

    Inherits static `scale` parameter from NormalizedPoissonTucker model,
    and `random_init` function.
    """

    pass
    # def __init__(self,
    #              scale: Scalar,
    #              unconstr_params: Optional[Sequence[ArrayLike]]=None,
    #              params: Optional[Sequence[ArrayLike]]=None):

    #     # Convert `params` to `unconstr_params` using `softmax_inverse`        
    #     _assert_single_input(unconstr_params, params)
    #     if unconstr_params is None:
    #         assert all(tree_map(lambda f: jnp.all(f >= 0), params)), "Expected `params` to be non-negative."
    #         unconstr_params = tree_map(
    #             lambda p, axis: softmax_inverse(jnp.maximum(p, SOFTPLUS_EPS), axis=axis),
    #             params, ((-1,), (-1,), (-1,), (0,)))
    #     del params

    #     # Now, call super init without worry
    #     super(SimplexPoissonTucker, self).__init__(scale, unconstr_params=unconstr_params)
    

    # def __call__(self):
    #     """Return non-negative core tensor and factors."""
    #     params = tree_map(
    #         lambda attr_name, axis: softmax_forward(self.__getattribute__(attr_name), axis=axis),
    #         ['unconstr_core', 'unconstr_fctr_1', 'unconstr_fctr_2', 'unconstr_fctr_3'],
    #         [(-1,), (-1,), (-1,), (0,)],)
    #     return *params,

    # def log_prior(self,) -> float:
    #     """Compute log prior of factor values."""
    #     return 0.