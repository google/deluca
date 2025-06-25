"""Brax physics engine simulation and output functions.

This module provides JIT-compiled functions for simulating physical systems using
the Brax physics engine. It handles state conversion between the controller's format
and Brax's internal state representation, including:
- Converting between flat state vectors and Brax's (q, qd) format
- Managing pipeline states and actions
- Handling state transitions through the physics engine
"""

from functools import partial
from typing import Callable

import brax
from brax import envs
from brax.base import State
from jax import jit
import jax.numpy as jnp

@partial(jit, static_argnames=['env', 'jit_step_fn'])
def brax_sim(
    x: jnp.ndarray, 
    u: jnp.ndarray,
    env: brax.envs.Env,
    template_state: brax.base.State,
    jit_step_fn: Callable
) -> jnp.ndarray:
    q_size = env.sys.q_size()

    x_flat = x.flatten()
    q = x_flat[:q_size]
    qd = x_flat[q_size:]

    current_pipeline_state = template_state.pipeline_state.replace(q=q, qd=qd)
    input_brax_state = template_state.replace(pipeline_state=current_pipeline_state)

    action = u.flatten()
    next_brax_state = jit_step_fn(input_brax_state, action)

    # --- 4. Pack the resulting state back into the controller's format ---
    next_q = next_brax_state.pipeline_state.q
    next_qd = next_brax_state.pipeline_state.qd
    next_x_flat = jnp.concatenate([next_q, next_qd])

    # Return as a (d, 1) column vector to match the controller's format
    return next_x_flat[:, jnp.newaxis]

@jit
def brax_output(x: jnp.ndarray) -> jnp.ndarray:
    """Output for brax is the full state."""
    return x