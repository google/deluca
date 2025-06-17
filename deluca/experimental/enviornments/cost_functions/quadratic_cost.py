"""Quadratic cost function evaluation for control systems.

This module provides functions to evaluate quadratic costs of the form:
    cost = y^T Q y + u^T R u
where y is the state/output vector, u is the control input vector,
and Q and R are positive semi-definite cost matrices.
"""

import jax
from jax import jit
import jax.numpy as jnp

@jit
def quadratic_cost_evaluate(y: jnp.ndarray, u: jnp.ndarray, Q: jnp.ndarray, R: jnp.ndarray) -> float:
    """Evaluate quadratic cost for a control system.
    
    Args:
        y: State/output vector
        u: Control input vector
        Q: State/output cost matrix (positive semi-definite)
        R: Control input cost matrix (positive semi-definite)
        
    Returns:
        The quadratic cost value: y^T Q y + u^T R u
    """
    return jnp.sum(y.T @ Q @ y + u.T @ R @ u)