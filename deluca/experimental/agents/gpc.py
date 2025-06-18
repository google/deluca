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
    d: int  # State dimension
    n: int  # Action dimension
    m: int  # History length
    k: int  # Action horizon
    hidden_dims: Optional[Sequence[int]] = None  # Optional hidden dimensions for each network

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x is the disturbance history of shape (m, d, 1)
        
        # Create m independent networks, one for each history element
        network = PerturbationNetwork(
            d_in=self.d,
            d_out=self.k * self.n,
            k=self.k,
            n=self.n,
            hidden_dims=self.hidden_dims
        )
        
        # Initialize parameters for all m networks at once
        params = self.param('networks',
                          lambda key, _: vmap(lambda k: network.init(k, jnp.zeros((self.d, 1))))(
                              jax.random.split(key, self.m)
                          ),
                          (self.m,))
        
        # Vectorize the network application across the m history elements
        # This applies each of the m networks to its corresponding disturbance vector in x.
        # Input: params(m, ...), x(m, d, 1) -> Output: perturbations(m, k, n, 1)
        apply_network = vmap(lambda p, h: network.apply(p, h))
        perturbations = apply_network(params, x)
        
        # Sum the contributions from each history element to get the final action plan
        # Output shape: (k, n, 1)
        return jnp.sum(perturbations, axis=0)
