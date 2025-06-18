"""Neural network models for control agents."""

from typing import Sequence, Optional

import jax.numpy as jnp
from flax import linen as nn

class PerturbationNetwork(nn.Module):
    """
    Neural network for computing a sequence of perturbations.
    It takes a single disturbance vector and outputs a sequence of k actions.
    """
    d_in: int
    d_out: int  # This will be k * n
    k: int
    n: int
    hidden_dims: Optional[Sequence[int]] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x is of shape (d_in, 1)
        # nn.Dense expects features to be the last dimension, so transpose
        x_t = x.T 

        if self.hidden_dims is None:
            # Linear layer
            y = nn.Dense(self.d_out)(x_t) # output (1, k * n)
        else:
            # Neural network with hidden layers
            y = x_t
            for hidden_dim in self.hidden_dims:
                y = nn.Dense(hidden_dim)(y)
                y = nn.relu(y)
            y = nn.Dense(self.d_out)(y) # output (1, k * n)
            
        # Reshape to (k, n, 1)
        return y.reshape((self.k, self.n, 1)) 