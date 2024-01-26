"""Stochastic gradient descent."""

from typing import Any, Callable, Optional, Sequence, Tuple
from jax._src.prng import PRNGKeyArray
from jaxtyping import Array, Bool, Integer, PyTree
from optax._src.base import GradientTransformation, OptState

from tqdm.auto import tqdm

import equinox as eqx
from jax import lax, vmap, value_and_grad
import jax.numpy as jnp
from jax.tree_util import tree_map
import optax

Model = eqx.Module

def setup_optimization(optimizer: GradientTransformation,
                       model_klass: Model,
                       model_full_shape: Sequence[int],
                       model_core_shape: Sequence[int],
                       model_init_params: dict={},
                       iter_callback: Optional[Callable[[float, Model], PyTree]]=None,   
                       minibatch_size: int=0,
                       log_prior_weight: float=1.,
                       ) -> Tuple[Callable, Callable]:
    """Setup optimization initialization and step functions.
    
    Parameters
    ----------
    model_klass: Model class reference
    model_full_shape: Sequence, corresponds to reconstructed tensor shape (d1, d2, d3)
    model_core_shape: Sequence, corresponds to core tensor shape (k1, k2, k3)
    model_init_params: dict, forwarded to model_klass.random_init
    optimizer_klass:
        Optax optimizer class reference; default: optax.adam
    optimzer_schedule_klass:
        Optax optimizer schedule class reference; default: optax.exponential_decay
    """

    # If iter_callback not specified, default to returning the loss
    if iter_callback is None:
        def iter_callback(model: Model,
                          data: Integer[Array, "*full"],
                          data_mask: Bool[Array, "*batch"],
                          loss: float,):
            return jnp.asarray(loss)
    
    def objective_fn(model: Model,
                     data: Integer[Array, "*full"],
                     data_mask: Bool[Array, "*batch"],) -> float:
        """Objective function, differentiate with respect to first argument."""
        lp = jnp.where(data_mask, model.log_likelihood(data, minibatch_size), 0.0)
        lp += log_prior_weight * model.log_prior()
        lp /= data_mask.sum()
        return -lp.sum()
    
    def init_fn(key: PRNGKeyArray,
                data: Integer[Array, "*full"],
                data_mask: Bool[Array, "*batch"],
    ) -> Tuple[Model, OptState]:
        
        # Randomly initialize data
        model = model_klass.random_init(
            key, model_full_shape, model_core_shape, **model_init_params
        )

        # Initialize optimizer state
        opt_state = optimizer.init(model)

        return model, opt_state
    
    def step_fn(init_model: Model,
                init_opt_state: OptState,
                data: Integer[Array, "*full"],
                data_mask: Bool[Array, "*batch"],
                n_iters: int,) -> Tuple[Model, OptState, PyTree]:

        def step(carry, itr):
            model, opt_state = carry
            loss, grads = value_and_grad(objective_fn)(model, data, data_mask)
            
            updates, updated_opt_state = optimizer.update(grads, opt_state, model)
            updated_model = optax.apply_updates(model, updates)

            output = iter_callback(updated_model, data, data_mask, loss)
            return (updated_model, updated_opt_state), output
    
        (model, opt_state), outputs \
            = lax.scan(step, (init_model, init_opt_state), jnp.arange(n_iters))
        
        return model, opt_state, outputs
    
    return init_fn, step_fn


def fit_opt(key: PRNGKeyArray,
            opt_init_fn: Callable,
            opt_step_fn: Callable,
            data: Integer[Array, "*full"],
            data_mask: Bool[Array, "*batch"],
            n_iters: int=5000):
    """Fit model to data using optimizer.
    
    Model and objective are defined implicitly in `init_fn` and `step_fn`.
    """
    
    init_model, init_opt_state = opt_init_fn(key, data, data_mask)
    model, _, outputs \
        = opt_step_fn(init_model, init_opt_state, data, data_mask, n_iters)

    return model, outputs