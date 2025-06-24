"""GPC (Generalized Predictive Control) implementation."""

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import jit


def get_gpc_features(
    m: int, 
    d: int,
) -> Callable[[int, jnp.ndarray], jnp.ndarray]:
    """
    Returns a function that computes GPC features.
    """
    def _get_features(offset: int, dist_history: jnp.ndarray) -> jnp.ndarray:
        return jax.lax.dynamic_slice(dist_history, (offset, 0, 0), (m, d, 1))

    return _get_features