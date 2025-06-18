"""Sinusoidal disturbance generation for control systems.

This module provides functions to generate sinusoidal disturbances for control system simulations.
The disturbances follow a pure sinusoidal pattern with specified amplitude and frequency.
"""

from functools import partial
import jax
from jax import jit
import jax.numpy as jnp

@partial(jit, static_argnames=['d'])
def sinusoidal_disturbance(d: int, noise_amplitude: float, noise_frequency: jax.Array, t: int, key: jax.random.PRNGKey) -> jax.Array:
    """Generate a sinusoidal disturbance vector.
    
    Args:
        d: Dimension of the disturbance vector
        noise_amplitude: Amplitude of the sinusoidal wave
        noise_frequency: Vector of frequencies for each dimension, must have length d
        t: Time step
        key: JAX random key (unused in this implementation)
        
    Returns:
        A d-dimensional vector of sinusoidal disturbances
        
    Raises:
        AssertionError: If noise_frequency vector length does not match d
    """
    assert d == noise_frequency.shape[0], "Noise frequency vector is of incorrect dimension"
    time_scaled = noise_frequency * t.astype(jnp.float32)
    sin_values = jnp.sin(time_scaled)
    return noise_amplitude * sin_values
