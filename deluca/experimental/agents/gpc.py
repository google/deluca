"""GPC (Generalized Predictive Control) implementation."""

from typing import Callable, Tuple, Any, Sequence, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import jit, vmap

from .model import PerturbationNetwork

@jit
def get_gpc_features(
    m: int, 
    d: int, 
    offset: int, 
    dist_history: jnp.ndarray
) -> jnp.ndarray:
    return jax.lax.dynamic_slice(dist_history, (offset, 0, 0), (m, d, 1))