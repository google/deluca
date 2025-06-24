"""GPC (Generalized Predictive Control) implementation."""

import jax
import jax.numpy as jnp
from jax import jit


def get_gpc_features(m, d):
    @jit
    def _get_gpc_features(offset, dist_history):
        return jax.lax.dynamic_slice(dist_history, (offset, 0, 0), (m, d, 1))
    return _get_gpc_features