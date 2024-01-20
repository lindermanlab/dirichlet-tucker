"""Three-mode Poisson Tucker model variants"""

from typing import Annotated, Optional, Sequence, Union
from jaxtyping import Array, ArrayLike, Bool, Float, Int, Num, PRNGKeyArray
Scalar = Union[int, float]

import warnings
import time
import itertools

from tqdm.auto import trange

import jax.numpy as jnp
import jax.random as jr
from tensorflow_probability.substrates import jax as tfp
from jax import jit, lax
from jax.tree_util import tree_map

tfd = tfp.distributions
warnings.filterwarnings("ignore")

TuckerParams = tuple(Float[Array, "K1 K2 K3"],
                     Float[Array, "D1 K1"],
                     Float[Array, "D2 K2"],
                     Float[Array, "K3 D3"],
                     )

class PoissonTucker():
    
    def __init__(self, core, factor_1, factor_2, factor_3, alpha=1.1):
        raise NotImplementedError
    
    def sample_params(self, key: PRNGKeyArray, D1: int, D2: int, D3: int):
        """Sample parameters from the model"""
        raise NotImplementedError
    
    def sample_data(self, key: PRNGKeyArray, params: TuckerParams) -> Int[Array, "D1 D2 D3"]:
        """Sample a data tensor from the model"""
        raise NotImplementedError
    

    def fit(self,
            X: Int[Array, "D1 D2 D3"],
            mask: Bool[Array, "D1 D2"],
            init_params: TuckerParams,
            num_iters: int,
            wnb=None):
        """Fit the model."""
        raise NotImplementedError