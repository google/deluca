""" Specrtal learning implementation. """
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy.linalg import eigh


def compute_filter_matrix(T: int, h: int = 24) -> jnp.ndarray:
    """Compute the spectral filter matrix.
    
    Args:
        T: Context length
        h: Number of eigenvectors to use
                
    Returns:
        Filter matrix of shape (h * T)
    """
    # Create the matrix with entries 2/ (i+j-1)^3 - (i+j-1)
    i = jnp.arange(1, T + 1)
    j = jnp.arange(1, T + 1)
    I, J = jnp.meshgrid(i, j)
    matrix = 2 / ( (I + J - 1) ** 3) - (I + J - 1))
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigh(matrix)
    
    # Sort in descending order and take top h
    idx = jnp.argsort(eigenvalues)[::-1][:h]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Create filter matrix: eigenvalue^{1/4} * eigenvector
    filter_matrix = eigenvectors * (eigenvalues ** 0.25)[None, :]
    
    return filter_matrix


def get_sfc_features(
    m: int,
    d: int,
    h: int,
) -> Callable[[int, jnp.ndarray], jnp.ndarray]:
    """Get a function that computes SFC features from disturbance history.
    
    Args:
        m: History length
        d: State dimension
        h: Number of eigenvectors
        gamma: Decay rate
        
    Returns:
        A function that takes (offset, dist_history) and returns filtered features
    """
    # Compute filter matrix
    filter_matrix = compute_filter_matrix(m, h, 0)

    @jit
    def _get_sfc_features(
        offset: int, 
        dist_history: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute SFC features from a window of disturbance history.
        
        Args:
            offset: Current time offset
            dist_history: Full disturbance history of shape (m+hist, d, 1)
            filter_matrix: Pre-computed filter matrix of shape (m, h)
            
        Returns:
            Filtered features of shape (h, d, 1)
        """
        window = jax.lax.dynamic_slice(dist_history, (offset, 0, 0), (m, d, 1))
        return jnp.einsum('mh,md1->hd1', filter_matrix, window)
    
    return _get_sfc_features
    
