"""Neural network models for control agents."""

from typing import Sequence, Optional

import jax.numpy as jnp
from flax import linen as nn

class PerturbationNetwork(nn.Module):
    """Neural network for computing perturbation at each timestep.
    
    Args:
        d_in: Input dimension
        d_out: Output dimension
        hidden_dims: Optional sequence of hidden layer dimensions. If None, uses a linear layer.
    """
    d_in: int
    d_out: int
    hidden_dims: Optional[Sequence[int]] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x is of shape (d_in, 1)
        if self.hidden_dims is None:
            # Linear layer
            return nn.Dense(self.d_out)(x)
            
        # Neural network with hidden layers
        x = x.reshape(-1)  # Flatten to (d_in,)
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
        x = nn.Dense(self.d_out)(x)
        return x.reshape(self.d_out, 1)  # Reshape to (d_out, 1) 