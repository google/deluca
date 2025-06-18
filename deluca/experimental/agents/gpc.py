"""GPC (Generalized Predictive Control) implementation."""
from typing import Sequence, Optional

import jax.numpy as jnp
from flax import linen as nn

class GPCModel(nn.Module):
    """A model for Generalized Predictive Control.

    This model can be either a linear model or a simple MLP, depending on
    the configuration of `hidden_dims`.
    """
    d: int  # State dimension
    n: int  # Input dimension
    m: int  # History length
    k: int  # Action horizon
    hidden_dims: Optional[Sequence[int]]

    @nn.compact
    def __call__(self, dist_history: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            dist_history: (m, d, 1) array of historical disturbances.

        Returns:
            An array of actions of shape (k, n, 1).
        """
        # Flatten the disturbance history to (m * d)
        features = dist_history.flatten()

        if self.hidden_dims is None:
            # Linear model
            # The kernel will have shape (m * d, k * n)
            kernel = self.param(
                'kernel',
                nn.initializers.lecun_normal(),
                (self.m * self.d, self.k * self.n)
            )
            y = features @ kernel
        else:
            # MLP model
            x = features
            for dim in self.hidden_dims:
                x = nn.Dense(features=dim)(x)
                x = nn.relu(x)
            y = nn.Dense(features=self.k * self.n)(x)

        # Reshape the output to (k, n, 1)
        actions = y.reshape((self.k, self.n, 1))
        return actions