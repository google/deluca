"""GPC (Generalized Predictive Control) implementation."""

import jax
import jax.numpy as jnp
from jax import jit


@jit
def get_gpc_features(
    m: int, 
    d: int, 
    offset: int, 
    dist_history: jnp.ndarray
) -> jnp.ndarray:
    return jax.lax.dynamic_slice(dist_history, (offset, 0, 0), (m, d, 1))