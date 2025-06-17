"""Environment step function for dynamical systems with disturbances.

This module provides a generic step function for simulating dynamical systems
with disturbances. It supports various types of disturbances and system dynamics
through callback functions.
"""

from typing import Callable

import jax
from jax import jit
import jax.numpy as jnp

@jit
def step(
    x: jnp.ndarray,
    u: jnp.ndarray,
    sim: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    output_map: Callable[[jnp.ndarray], jnp.ndarray],
    dist_std: float,
    t_sim_step: int,
    disturbance: Callable[[int, float, int, jax.random.PRNGKey], jax.Array],
    key: jax.random.PRNGKey
) -> tuple[jnp.ndarray, jnp.ndarray]: # Returns (next_x, y_t)
    d = x.shape[0]
    w = disturbance(d, dist_std, t_sim_step, key)
    x_next = sim(x, u) + w
    y = output_map(x_next)
    return x_next, y
