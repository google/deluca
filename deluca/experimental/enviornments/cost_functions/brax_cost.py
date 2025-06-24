"""Cost function for Brax environments."""
from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit)
def brax_cost_evaluate(y: jnp.ndarray, u: jnp.ndarray) -> float:
    """
    Evaluate the quadratic cost for a Brax environment.
    This is configured for the 'inverted_pendulum' environment.
    """
    # For inverted_pendulum, state is [cart_pos, pole_angle, cart_vel, pole_ang_vel].
    # We want to keep the pole upright (angle=0) and cart at origin (pos=0).
    Q = jnp.diag(jnp.array([1.0, 10.0, 0.1, 0.1]))
    R = jnp.diag(jnp.array([0.01]))

    cost = y.T @ Q @ y + u.T @ R @ u
    return jnp.squeeze(cost) 