"""Cost function for the pendulum environment."""

import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit)
def angle_normalize(x: float) -> float:
    """Normalize an angle to the range [-pi, pi]."""
    return ((x + jnp.pi) % (2 * jnp.pi))

@partial(jax.jit)
def pendulum_cost_evaluate(y: jnp.ndarray, u: jnp.ndarray) -> float:
    """
    Evaluate the cost for the pendulum environment.

    The cost is a quadratic function of the angle, angular velocity, and control input.
    We want to keep the pendulum upright (theta=0) with zero velocity.

    Args:
        y: State vector [theta, thdot].
        u: Control input.

    Returns:
        The scalar cost value.
    """
    theta, thdot = y.flatten()
    u_val = u.flatten()[0]
    
    # Cost based on cosine of angle is better for swing-up
    cost = (1 - jnp.cos(theta)) + 0.1 * thdot**2 + 0.001 * u_val**2
    return cost 