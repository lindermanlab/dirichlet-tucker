"""Stochastic gradient descent."""

from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple
from jax._src.prng import PRNGKeyArray
from jaxtyping import Array, Bool, Float, Integer, PyTree
from optax._src.base import GradientTransformation, OptState

from tqdm.auto import tqdm

import chex
import equinox as eqx
from jax import lax, vmap, value_and_grad
import jax.numpy as jnp
from jax.tree_util import tree_map
import optax

Model = eqx.Module
DataArray = Integer[Array, "*full"]
MaskArray = Bool[Array, "*batch"]
IterCallbackCallable = Callable[[float, Model, DataArray, MaskArray], PyTree]

@chex.dataclass
class DefaultIterOutput:
    """Default dataclass """
    loss: Float[Array, "n"]
    train_ll: Float[Array, "n"]
    vldtn_ll: Float[Array, "n"]


def _default_iter_callback(loss: float,
                           model: Model,
                           data: Integer[Array, "*full"],
                           data_mask: Bool[Array, "*batch"],) -> PyTree:
    """Default callback function for computing values at each iteration step.
    
    Returns training loss and validation loss (log likelihood of held-out data).
    """

    ll = model.log_likelihood(data)

    train_ll = (jnp.where(data_mask, ll, 0.0) / data_mask.sum()).sum()
    vldtn_ll = (jnp.where(~data_mask, ll, 0.0) / (~data_mask).sum()).sum()

    return DefaultIterOutput(loss=loss, train_ll=train_ll, vldtn_ll=vldtn_ll)


def fit_opt(model: Model,
            data: Integer[Array, "*full"],
            data_mask: Bool[Array, "*batch"],
            objective_fn: Callable[[Model], float],
            optimizer: GradientTransformation,
            opt_state: OptState,
            n_iters: int=5000,
            iter_callback: Optional[IterCallbackCallable]=None,
            ):
    """Fit model to data using optimizer.
    
    Model and objective are defined implicitly in `init_fn` and `step_fn`.

    Parameters
    ----------
    data: integer ndarray, ([bm, ..., b1], b0, [en,...,e1], e0 )
    data_mask: boolean ndarray, ([bm, ..., b1], b0)
        Data and fit/validation mask.
    model: Model
        Parameterize model to optimize
    objective_fn: Objective function to minimize
    optimizer: optax.GradientTransformation
    opt_state:  optax.OptState
        Optimizer and optimizer state
    n_iters: int
        Number of iterations to optimize over
    iter_callback: Callable[[Model, data, data_mask, float], PyTree], optional
        Callback for computing intermediate values at each iteration step.
        If None, use `_default_iter_callback` which returns the `train_loss`
        and `vldtn_loss`. Output _must_ have an attribute called `train_loss`.
        This is not explicitly checked or enforced anywhere.
    """

    if iter_callback is None:
        iter_callback = _default_iter_callback

    def _objective_fn(model_params, model_fields):
        model = eqx.combine(model_params, model_fields)
        return objective_fn(model)
    
    def step(carry, itr):
        model, opt_state = carry

        # Split the model into updateable params and static fields
        model_params, model_static_fields = eqx.partition(model, eqx.is_array)
        
        # Compute gradients and apply updates
        loss, grads = value_and_grad(_objective_fn)(model_params, model_static_fields)
        updates, updated_opt_state = optimizer.update(grads, opt_state, model)
        updated_model_params = optax.apply_updates(model_params, updates)

        # Recombine the updated model parameters with the static fields
        updated_model = eqx.combine(updated_model_params, model_static_fields)

        output = iter_callback(loss, updated_model, data, data_mask)
        return (updated_model, updated_opt_state), output

    (updated_model, updated_opt_state), outputs \
        = lax.scan(step, (model, opt_state), jnp.arange(n_iters))

    return (updated_model, updated_opt_state), outputs