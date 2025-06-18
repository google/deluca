"""Unified interface for control agents.

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
from flax import struct

@struct.dataclass
class AgentState:
    """Base class for agent states."""
    controller_t: int
    state: jnp.ndarray
    dist_history: jnp.ndarray  # Natural response history
    params: Any       # Model parameters
    opt_state: Any   # Optimizer state

@partial(jit, static_argnames=['apply_fn', 'd', 'm', 'sim', 'cost_fn'])
def policy_loss(
    apply_fn: Callable,
    params: Any,
    d: int,
    m: int,
    dist_history: jnp.ndarray,
    sim: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    cost_fn: Callable[[jnp.ndarray, jnp.ndarray], float],
) -> jnp.ndarray:
    # Get the sequence of actions from the policy
    actions = apply_fn(params, dist_history)

    def evolve(state: jnp.ndarray, offset: int) -> Tuple[jnp.ndarray, int]:
        slice = jax.lax.dynamic_slice(dist_history, (offset, 0, 0), (m, d, 1))
        actions = apply_fn(params, slice)
        next_state = sim(state, actions[0]) + dist_history[-m + offset]
        return next_state, None
    final_state, _ = jax.lax.scan(evolve, jnp.zeros((d, 1)), jnp.arange(m-1))
    final_slice = jax.lax.dynamic_slice(dist_history, (m-1, 0, 0), (m, d, 1))
    final_actions = apply_fn(params, final_slice)
    def collect_cost(carry: Tuple[jnp.ndarray, float, jnp.ndarray], action: jnp.ndarray) -> Tuple[Tuple[jnp.ndarray, float], jnp.ndarray]:
        state, total_cost, dist = carry
        cost = cost_fn(state, action)
        next_state = sim(state, action) + dist
        
        return (next_state, total_cost+cost, jnp.zeros_like(dist)), None
    (_, total_cost, dist), _ = jax.lax.scan(collect_cost, (final_state, 0.0, dist_history[-1]), final_actions)
    return total_cost


@partial(jit, static_argnames=['sim', 'grad_fn', 'optimizer'])
def update_agentstate(
    agentstate: AgentState,
    next_state: jnp.ndarray,
    action: jnp.ndarray,
    sim: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    grad_fn: Callable[[Any, jnp.ndarray], jnp.ndarray],
    optimizer: optax.GradientTransformation
) -> AgentState:

    disturbance = next_state - sim(agentstate.state, action)
    
    # Update disturbance history
    dist_history = agentstate.dist_history.at[0].set(disturbance)
    dist_history = jnp.roll(dist_history, shift=1, axis=0)
    
    grads = grad_fn(agentstate.params, dist_history)
    updates, new_opt_state = optimizer.update(grads, agentstate.opt_state, agentstate.params)
    new_params = optax.apply_updates(agentstate.params, updates)
    
    return AgentState(
        controller_t=agentstate.controller_t + 1,
        state=next_state,
        dist_history=dist_history,
        params=new_params,
        opt_state=new_opt_state
    )