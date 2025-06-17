"""Gaussian disturbance generation for control systems.

This module provides functions to generate Gaussian (normal) distributed disturbances
for control system simulations. The disturbances are sampled from a normal distribution
with zero mean and specified standard deviation.
"""

import jax
from jax import jit
import jax.numpy as jnp

@jit
def gaussian_disturbance(d: int, dist_std: float, t: int, key: jax.random.PRNGKey) -> jax.Array:
    """Generate a Gaussian disturbance vector.
    
    Args:
        d: Dimension of the disturbance vector
        dist_std: Standard deviation of the Gaussian distribution
        t: Time step (unused in this implementation)
        key: JAX random key for reproducibility
        
    Returns:
        A d-dimensional vector of Gaussian disturbances
    """
    return jax.random.normal(key, (d, 1)) * dist_std