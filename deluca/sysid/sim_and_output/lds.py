"""Linear Dynamical System (LDS) simulation and output functions.

This module provides JIT-compiled functions for simulating linear dynamical systems
and computing their outputs. The system follows the standard LDS equations:
    x_{t+1} = Ax_t + Bu_t
    y_t = Cx_t
"""

from jax import jit
import jax.numpy as jnp

@jit
def lds_sim(x: jnp.ndarray, u: jnp.ndarray, A: jnp.ndarray, B:jnp.ndarray) -> jnp.ndarray:
    return A @ x + B @ u

@jit
def lds_output(x: jnp.ndarray, C: jnp.ndarray) -> jnp.ndarray:
    return C @ x