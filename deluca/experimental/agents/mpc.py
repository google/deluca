"""MPC (Model Predictive Control) implementation."""

from typing import Sequence, Optional

from flax import linen as nn
import jax.numpy as jnp

from .model import PerturbationNetwork


class MPCModel(nn.Module):
    """Neural network model for MPC.
    
    This model implements the MPC policy by taking the current state
    and outputting a sequence of k actions.
    """
    d: int  # State dimension
    n: int  # Action dimension
    k: int  # Action horizon
    hidden_dims: Optional[Sequence[int]] = None  # Optional hidden dimensions for the network

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x is the current state of shape (d, 1)
        
        network = PerturbationNetwork(
            d_in=self.d,
            d_out=self.k * self.n,
            k=self.k,
            n=self.n,
            hidden_dims=self.hidden_dims
        )
        
        # Apply the network to the state
        actions = network(x)
        
        # Output shape: (k, n, 1)
        return actions 