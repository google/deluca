"""Unified interface for control agents.

This module provides a unified interface for different control agents:
- GPC (Generalized Predictive Control)
- SFC (Spectral Filter Control)
- DSC (Dynamic Spectral Control)
- BatchNN-DRC (Batch Neural Network Dynamic Response Control)

Each agent is implemented as a set of JIT-compiled functions that can be partially applied
for specific configurations.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax
from jax import jit
import jax.numpy as jnp
import optax
from flax import linen as nn

@dataclass
class AgentState:
    """Base class for agent states."""
    controller_t: int
    state: jnp.ndarray
    dist_history: jnp.ndarray  # Natural response history
    params: Any       # Model parameters
    opt_state: Any   # Optimizer state

# @jit
# def init_agentstate(
#     key: jax.random.PRNGKey,
#     init_scale: float,
#     init_model: Callable[
#         [jax.random.PRNGKey, float, int, int, int, int, Optional[int], Optional[int]],
#         Tuple[nn.Module, Any]
#     ],
#     init_opt_state: Any,
#     hist_len: Callable[[int, int], int],
#     n: int,
#     p: int,
#     d: int,
#     m: int,
#     hist: int,
#     h: Optional[int] = None,
#     k: Optional[int] = None
# ) -> AgentState:
#     model, params = init_model(key, init_scale, n, p, d, m, h, k)
#     z = jnp.zeros((d, 1))
#     ynat_history = jnp.zeros((hist_len(m, hist), p, 1))
#     return model, AgentState(
#         controller_t=0,
#         z=z,
#         ynat_history=ynat_history,
#         params=params,
#         opt_state=init_opt_state
#     )

# @jit
# def get_action(model: nn.Module, params: Any, features: jnp.ndarray) -> jnp.ndarray:
#     return model.apply(params, features)

@jit
def policy_loss(
    model: nn.Module,
    params: Any,
    d: int,
    hist: int,
    dist_history: jnp.ndarray,
    sim: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    # output_map: Callable[[jnp.ndarray], jnp.ndarray],
    cost_fn: Callable[[jnp.ndarray, jnp.ndarray], float],
    slice_history: Callable[[int, jnp.ndarray], jnp.ndarray]
) -> jnp.ndarray:
    def evolve(state: jnp.ndarray, offset: int) -> Tuple[jnp.ndarray, int]:
        slice = slice_history(offset, dist_history)
        actions = model.apply(params, slice)
        next_state = sim(state, actions[0]) + dist_history[-hist + offset - 1]
        return next_state, None
    final_state, _ = jax.lax.scan(evolve, jnp.zeros(d, 1), jnp.arange(hist-1))
    # final_obs = output_map(final_delta) + ynat_history[-1]
    final_slice = slice_history(hist-1, dist_history)
    final_actions = model.apply(params, final_slice)
    def collect_cost(carry: Tuple[jnp.ndarray, float, jnp.ndarray], action: jnp.ndarray) -> Tuple[Tuple[jnp.ndarray, float], jnp.ndarray]:
        state, total_cost, dist = carry
        # output = output_map(delta)
        # Add ynat_history[-1] only to the first output (i == 0)
        # output = output + ynat
        cost = cost_fn(state, action)
        next_state = sim(state, action) + dist
        
        return (next_state, total_cost+cost, jnp.zeros_like(dist)), None
    (_, total_cost), _ = jax.lax.scan(collect_cost, (final_state, 0.0, dist_history[-1]), final_actions)
    return total_cost


@jit
def update_agentstate(
    agentstate: AgentState,
    next_state: jnp.ndarray,
    action: jnp.ndarray,
    sim: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    # output_map: Callable[[jnp.ndarray], jnp.ndarray],
    grad_fn: Callable[[Any, jnp.ndarray], jnp.ndarray],
    optimizer: optax.GradientTransformation
) -> AgentState:
    # z_next = sim(agentstate.z, action)
    # y_nat = observation - output_map(z_next)

    disturbance = next_state - sim(agentstate.state, action)
    agentstate.dist_history = agentstate.dist_history.at[0].set(disturbance)
    agentstate.dist_history = jnp.roll(agentstate.dist_history, shift=1, axis=0)
    
        
    grads = grad_fn(agentstate.params, agentstate.dist_history)
    updates, new_opt_state = optimizer.update(grads, agentstate.opt_state, agentstate.params)
    new_params = optax.apply_updates(agentstate.params, updates)
    
    return AgentState(
        controller_t=agentstate.controller_t + 1,
        state=next_state,
        dist_history=agentstate.dist_history,
        params=new_params,
        opt_state=new_opt_state
    )