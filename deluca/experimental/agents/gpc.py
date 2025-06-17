"""GPC (Generalized Predictive Control) implementation."""

from typing import Callable, Tuple, Any, Sequence, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import jit, vmap

from .model import PerturbationNetwork

class GPCModel(nn.Module):
    """Neural network model for GPC.
    
    This model implements the GPC policy:
    u = sum(M_i @ w_i) for i in 1..m
    where:
    - M_i are neural networks (or linear layers)
    - w_i is the disturbance at timestep i
    
    Note: The random key for initialization is passed to model.init() when the model
    is first created, not in this class.
    """
    d: int  # Action dimension
    n: int  # State dimension
    m: int  # History length
    hidden_dims: Optional[Sequence[int]] = None  # Optional hidden dimensions for each network

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x is the disturbance history of shape (m, n, 1)
        
        # Create a single network and vmap it across timesteps
        network = PerturbationNetwork(
            d_in=self.n,
            d_out=self.d,
            hidden_dims=self.hidden_dims
        )
        
        # Initialize parameters for all timesteps at once
        # The key is passed to model.init() when the model is first created
        params = self.param('networks',
                          lambda key, _: vmap(lambda k: network.init(k, jnp.zeros((self.n, 1))))(
                              jax.random.split(key, self.m)
                          ),
                          (self.m,))
        
        # Vectorize the network application across timesteps
        apply_network = vmap(lambda p, x: network.apply(p, x))
        perturbations = apply_network(params, x)
        
        # Sum all perturbations
        return jnp.sum(perturbations, axis=0)
