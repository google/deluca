"""Neural network models for control agents."""

from typing import Sequence, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import vmap

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

class FullyConnectedModel(nn.Module):
    """
    Fully connected model that takes input of shape (m, d, 1) and outputs shape (k, n, 1).
    If hidden_dims is None, it's a linear model.
    """
    m: int  # Input sequence length
    d: int  # Input dimension
    k: int  # Output sequence length
    n: int  # Output dimension
    hidden_dims: Optional[Sequence[int]] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x is of shape (m, d, 1)
        # Flatten input to (m*d, 1)
        x_flat = x.reshape((-1, 1))
        
        if self.hidden_dims is None:
            # Linear layer
            y = nn.Dense(self.k * self.n)(x_flat)  # output (1, k*n)
        else:
            # Neural network with hidden layers
            y = x_flat
            for hidden_dim in self.hidden_dims:
                y = nn.Dense(hidden_dim)(y)
                y = nn.relu(y)
            y = nn.Dense(self.k * self.n)(y)  # output (1, k*n)
        
        # Reshape to (k, n, 1)
        return y.reshape((self.k, self.n, 1))

class SequentialModel(nn.Module):
    """
    Model that creates m independent networks, each taking input[i,:,:] to (k, n, 1).
    If hidden_dims is None, each network is linear.
    """
    m: int  # Input sequence length
    d: int  # Input dimension
    k: int  # Output sequence length
    n: int  # Output dimension
    hidden_dims: Optional[Sequence[int]] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x is of shape (m, d, 1)
        
        # Create m independent networks
        network = PerturbationNetwork(
            d_in=self.d,
            d_out=self.k * self.n,
            k=self.k,
            n=self.n,
            hidden_dims=self.hidden_dims
        )
        
        # Initialize parameters for all m networks
        params = self.param('networks',
                          lambda key, _: vmap(lambda k: network.init(k, jnp.zeros((self.d, 1))))(
                              jax.random.split(key, self.m)
                          ),
                          (self.m,))
        
        # Vectorize the network application across the m inputs
        # Input: params(m, ...), x(m, d, 1) -> Output: perturbations(m, k, n, 1)
        apply_network = vmap(lambda p, h: network.apply(p, h))
        perturbations = apply_network(params, x)
        
        # Sum the contributions from each network
        # Output shape: (k, n, 1)
        return jnp.sum(perturbations, axis=0)

class GridModel(nn.Module):
    """
    Model that creates a grid of (m, k) networks, where network (i,j) takes input[i,:,:]
    and contributes to output[j,:,:]. If hidden_dims is None, each network is linear.
    """
    m: int  # Input sequence length
    d: int  # Input dimension
    k: int  # Output sequence length
    n: int  # Output dimension
    hidden_dims: Optional[Sequence[int]] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x is of shape (m, d, 1)
        
        # Create a single network that will be used for all (i,j) pairs
        network = PerturbationNetwork(
            d_in=self.d,
            d_out=self.n,  # Each network only needs to output one action
            k=1,  # Each network only needs to output one action
            n=self.n,
            hidden_dims=self.hidden_dims
        )
        
        # Initialize parameters for all m*k networks
        params = self.param('networks',
                          lambda key, _: vmap(lambda k: network.init(k, jnp.zeros((self.d, 1))))(
                              jax.random.split(key, self.m * self.k)
                          ),
                          (self.m, self.k))
        
        # Vectorize the network application across the m inputs and k outputs
        # Input: params(m, k, ...), x(m, d, 1) -> Output: perturbations(m, k, n, 1)
        apply_network = vmap(vmap(lambda p, h: network.apply(p, h)))
        perturbations = apply_network(params, x[:, None, :, :])  # Add k dimension to x
        
        # Sum the contributions from each network to get the final action plan
        # Output shape: (k, n, 1)
        return jnp.sum(perturbations, axis=0) 