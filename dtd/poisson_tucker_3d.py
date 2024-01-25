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
                     mean_rate: Float[Array, "*full"]) -> float:
    """Compute Poisson log pdf of data under the given mean rate."""

    ll = jnp.sum(data * jnp.log(mean_rate))
    ll -= jnp.sum(mean_rate)
    ll -= jnp.sum(jsp.special.gammaln(data + 1))

    return ll

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
                    event_ndims: int=1,
    ) -> "PoissonTucker":
        """Initialize a Poisson Tucker instance with random parameter values.

        (Unconstrained) parameters are sampled from Uniform(-5, 5).
        
        Parameters
        ----------
        key: PRNGKeyArray
        full_shape: Sequence, corresponds to reconstructed tensor shape (d1, d2, d3)
        core_shape: Sequence, corresponds to core tensor shape (k1, k2, k3)
        event_ndims: int, number of event dimensions. Default: 1
            Only event_ndims = 1 is currently supported. Parameter exposed for
            documentation purposes and the sake of being explicit.
        """

        tensor_mode = 3
        if len(full_shape) != tensor_mode:
            raise ValueError(f"Expecting len(full_shape)={tensor_mode}, but got full_shape={full_shape}.")

        if len(core_shape) != tensor_mode:
            raise ValueError(f"Expecting len(core_shape)={tensor_mode}, but got core_shape={core_shape}.")

        if event_ndims != 1:
            raise ValueError(
                f"Only event_ndims= 1 is currently supported, but got {event_ndims}." \
                "Dynamic event dimensioning may be explored in the future. For now," \
                "create a new class and handle event dimensioning manually."
            )

        key, key_ = jr.split(key)
        G_param = jr.uniform(key, shape=tuple(core_shape), minval=-5, maxval=-5)
        F_params = [
            jr.uniform(key_, shape=F_shape, minval=-5, maxval=-5)
            for key_, F_shape in zip(jr.split(key_, tensor_mode), zip(full_shape, core_shape))
        ]
        
        return cls(G_param, *F_params, event_ndims=event_ndims)

    
    @property
    def full_shape(self,) -> Tuple[int, int, int]:
        """Shape of reconstructed tensor."""
        return tuple([len(f) for f in [self.F1_param, self.F2_param, self.F3_param]])
    

    @property
    def core_shape(self,) -> Tuple[int, int, int]:
        """Shape of core tensor."""
        return self.G_param.shape
    

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
                                  data: Integer[Array, "*full"],) -> float:
        """Compute full-batch Poisson log-likelihood under the current parameters."""
        
        mean_rate = self.reconstruct()
        return poisson_log_prob(data, mean_rate)
    
    def _streaming_log_likelihood(self,
                                  batch_size: int,
                                  data: Integer[Array, "*full"],) -> float:
        """Compute full-batch Poisson log-likelihood in a streaming way.

        This is useful when one dimension is extremely large and requires a more
        memory-efficient way of computing the log-likelihod. Note that the
        mini-batching for the streaming computation is only taken over a single
        axis, axis=0.

        Note that this _is_ the exact log-likelihood (and not an approximation)
        because the first axis should be a batch axis and slices are independent.
        Therefore, the full log-likelihood can be computed in mini-batches
        and cumulatively added together.

        Parameters
        ----------
        TODO
        """

        raise NotImplementedError
    
    #     core, fctr_1, fctr_2, fctr_3, *_ = self.__call__()

    #     # Define single minibatch log-likelihood
    #     def minibatch_ll(y_m, f_1_m):
    #         rate_m = self._full(core, f_1_m, fctr_2, fctr_3)
    #         return poisson_log_prob(rate_m, y_m)

    #     # Scan over complete minibatches 
    #     n_batches = len(y) // batch_size
    #     d1_full = n_batches * batch_size

    #     def step(ll_mm1, start_idx):
    #         y_m = lax.dynamic_slice_in_dim(y, start_idx, batch_size)
    #         f_1_m = lax.dynamic_slice_in_dim(fctr_1, start_idx, batch_size)
    #         return ll_mm1 + minibatch_ll(y_m, f_1_m), None
        
    #     ll, _ = lax.scan(step, 0., jnp.arange(0, d1_full, batch_size))

    #     # If last batch is incomplete, calculate its log-likelihood now
    #     ll += lax.cond((len(y) % batch_size) > 0,
    #                    lambda y_m, f_m: minibatch_ll(y_m, f_m,),
    #                    lambda *_: 0.,
    #                    y[d1_full:], fctr_1[d1_full:])
    #     return ll
    
    def log_likelihood(self,
                       data: Integer[Array, "*full"],
                       batch_size: int=0) -> Float[Array, "*event"]:
        """Compute log likelihood of data under the model."""
        
        if batch_size > 0:
            return self._streaming_log_likelihood(batch_size, data)
        else:
            return self._fullbatch_log_likelihood(data)
        
    def log_prior(self,) -> Float[Array, "#event"]:
        """Compute log prior of parameter values."""

        return jnp.array(0)
    
    def log_prob(self,
                data: Integer[Array, "*full"],
                batch_size: int=0,
                log_prior_scale: float=1.) -> Float[Array, "*event"]:
        
        return self.log_likelihood(data, batch_size) + log_prior_scale * self.log_prior()
    
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