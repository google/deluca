"""SFC (Spectral Filter Control) implementation."""

from typing import Callable, Tuple, Any, Sequence, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import jit, vmap
from jax.scipy.linalg import eigh

from .model import PerturbationNetwork

def compute_filter_matrix(m: int, h: int, gamma: float) -> jnp.ndarray:
    """Compute the spectral filter matrix.
    
    Args:
        m: History length
        h: Number of eigenvectors to use
        gamma: Decay rate between 0 and 1
        
    Returns:
        Filter matrix of shape (h, m)
    """
    # Create the matrix with entries (1-gamma)^{i+j-1} / (i+j-1)
    i = jnp.arange(1, m + 1)
    j = jnp.arange(1, m + 1)
    I, J = jnp.meshgrid(i, j)
    matrix = ((1 - gamma) ** (I + J - 1)) / (I + J - 1)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigh(matrix)
    
    # Sort in descending order and take top h
    idx = jnp.argsort(eigenvalues)[::-1][:h]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Create filter matrix: eigenvalue^{1/4} * eigenvector
    filter_matrix = eigenvectors * (eigenvalues ** 0.25)[None, :]
    
    return filter_matrix

class SFCModel(nn.Module):
    """Neural network model for SFC.
    
    This model implements the SFC policy:
    1. Apply spectral filtering to reduce dimension from m to h
    2. Then apply GPC-like control: u = sum(M_i @ w_i) for i in 1..h
    where:
    - M_i are neural networks (or linear layers)
    - w_i is the filtered disturbance at timestep i
    """
    d: int  # Action dimension
    n: int  # State dimension
    m: int  # History length
    h: int  # Number of eigenvectors
    gamma: float  # Decay rate
    hidden_dims: Optional[Sequence[int]] = None  # Optional hidden dimensions for each network

    def setup(self):
        # Compute filter matrix once during setup
        self.filter_matrix = compute_filter_matrix(self.m, self.h, self.gamma)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x is the disturbance history of shape (m, n, 1)
        
        # Apply spectral filtering to reduce dimension from m to h
        # Use einsum to handle the batch dimension directly
        # mh,mn1->hn1: (m,h) @ (m,n,1) -> (h,n,1)
        filtered_x = jnp.einsum('mh,mn1->hn1', self.filter_matrix, x)
        
        # Create a single network and vmap it across timesteps
        network = PerturbationNetwork(
            d_in=self.n,
            d_out=self.d,
            hidden_dims=self.hidden_dims
        )
        
        # Initialize parameters for all timesteps at once
        params = self.param('networks',
                          lambda key, _: vmap(lambda k: network.init(k, jnp.zeros((self.n, 1))))(
                              jax.random.split(key, self.h)
                          ),
                          (self.h,))
        
        # Vectorize the network application across timesteps
        apply_network = vmap(lambda p, x: network.apply(p, x))
        perturbations = apply_network(params, filtered_x)
        
        # Sum all perturbations
        return jnp.sum(perturbations, axis=0)
