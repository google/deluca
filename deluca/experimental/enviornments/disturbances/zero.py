"""Zero disturbance generation for control systems.

This module provides functions to generate zero disturbances for control system simulations.
This is useful for testing control algorithms in a noise-free environment or as a baseline
for comparing with other disturbance types.
"""

from functools import partial
import jax
from jax import jit
import jax.numpy as jnp

@partial(jit, static_argnames=['d'])
def zero_disturbance(d: int, dist_std: float, t: int, key: jax.random.PRNGKey) -> jax.Array:
    """Generate a zero disturbance vector.
    
    Args:
        d: Dimension of the disturbance vector
        dist_std: Standard deviation (unused in this implementation)
        t: Time step (unused in this implementation)
        key: JAX random key (unused in this implementation)
        
    Returns:
        A d-dimensional vector of zeros
    """
    return jnp.zeros((d, 1))