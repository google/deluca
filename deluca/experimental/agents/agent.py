"""Unified interface for control agents.

Each agent is implemented as a set of JIT-compiled functions that can be partially applied
for specific configurations.
"""

from functools import partial
from typing import Any, Callable, Tuple

import jax
from jax import jit
import jax.numpy as jnp
import optax
from flax import struct
from jax import debug as jax_debug

@struct.dataclass
class AgentState:
    """Base class for agent states."""
    controller_t: int
    state: jnp.ndarray
    dist_history: jnp.ndarray  # Natural response history
    state_history: jnp.ndarray # Actual state history
    params: Any       # Model parameters
    opt_state: Any   # Optimizer state

    def show(self):
        print(f'controller_t={self.controller_t}')
        print(f'state={getattr(self, "state", None)}')
        print(f'dist_history={getattr(self, "dist_history", None)}')
        print(f'state_history={getattr(self, "state_history", None)}')
        print(f'params={getattr(self, "params", None)}')
        print(f'opt_state={getattr(self, "opt_state", None)}')

@partial(jit, static_argnames=['apply_fn', 'd', 'm', 'sim', 'cost_fn', 'get_features', 'debug'])
def policy_loss(
    apply_fn: Callable,
    params: Any,
    d: int,
    m: int,
    dist_history: jnp.ndarray,
    start_state: jnp.ndarray,
    sim: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    cost_fn: Callable[[jnp.ndarray, jnp.ndarray], float],
    get_features: Callable[[int, jnp.ndarray], jnp.ndarray],
    debug: bool = False,
) -> jnp.ndarray:
    def evolve(state: jnp.ndarray, offset: int) -> Tuple[jnp.ndarray, int]:
        slice = get_features(offset, dist_history)
        actions = apply_fn(params, jnp.concatenate([jnp.reshape(state, (1, d, 1)), slice], axis=0))
        next_state = sim(state, actions[0]) + dist_history[-m + offset]
        # if debug:
        #     jax_debug.print('offset: {}, slice: {}, dist to add: {}', offset, slice, dist_history[-m + offset])
        return next_state, None
    final_state, _ = jax.lax.scan(evolve, start_state, jnp.arange(m-1))
    final_slice = get_features(m-1, dist_history)
    final_actions = apply_fn(params, jnp.concatenate([jnp.reshape(final_state, (1, d, 1)), final_slice], axis=0))
    # if debug:
    #     jax_debug.print('final_offset: {}, final_slice: {}, final dist to add: {}', m-1, final_slice, dist_history[-1])
    #     jax_debug.print('final actions: {}', final_actions)
    def collect_cost(carry: Tuple[jnp.ndarray, float, jnp.ndarray], action: jnp.ndarray) -> Tuple[Tuple[jnp.ndarray, float], jnp.ndarray]:
        state, total_cost, dist = carry
        cost = cost_fn(state, action)
        next_state = sim(state, action) + dist
        # if debug:
        #     jax_debug.print('state: {}, action: {}, dist: {}, next_state: {}', state, action, dist, next_state)
        return (next_state, total_cost+cost, jnp.zeros_like(dist)), None
    (_, total_cost, _), _ = jax.lax.scan(collect_cost, (final_state, 0.0, dist_history[-1]), final_actions)
    return total_cost


@partial(jit, static_argnames=['sim', 'grad_fn', 'optimizer', 'debug'])
def update_agentstate(
    agentstate: AgentState,
    next_state: jnp.ndarray,
    action: jnp.ndarray,
    sim: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    grad_fn: Callable[[Any, jnp.ndarray], jnp.ndarray],
    optimizer: optax.GradientTransformation,
    debug: bool = False
) -> AgentState:
    disturbance = next_state - sim(agentstate.state, action)
    
    # Update disturbance history
    dist_history = agentstate.dist_history.at[0].set(disturbance)
    dist_history = jnp.roll(dist_history, shift=-1, axis=0)

    #Update state history
    state_history = agentstate.state_history.at[0].set(jax.lax.stop_gradient(next_state))
    state_history = jnp.roll(state_history, shift=-1, axis=0)
    
    grads = grad_fn(agentstate.params, dist_history, state_history[0])
    updates, new_opt_state = optimizer.update(grads, agentstate.opt_state, agentstate.params)
    new_params = optax.apply_updates(agentstate.params, updates)
    
    # if debug:
    #     # jax_debug.print('update_agentstate: controller_t={}, next_state={}', agentstate.controller_t, next_state)
    #     jax_debug.print('dist_history={}, state_history={}', dist_history, state_history)
        # jax_debug.print('grads type: {}', type(grads))
    return AgentState(
        controller_t=agentstate.controller_t + 1,
        state=next_state,
        dist_history=dist_history,
        state_history=state_history,
        params=new_params,
        opt_state=new_opt_state
    )