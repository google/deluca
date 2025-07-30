from typing import Tuple
from jax import Array
import jax.numpy as jnp
from jax.numpy.linalg import eigh as largest_eigh

from deluca.filters.core import Filter


def get_filters(spectral_history_length: int, num_filters: int = 24):
    num_filters = min(num_filters, spectral_history_length)
    i = jnp.arange(1, spectral_history_length + 1)
    i_plus_j = i[None] + i[:, None]
    Z = 2 / (i_plus_j**3 - i_plus_j)
    Z = jnp.float32(Z)
    evals, evecs = largest_eigh(Z)
    return evecs[::-1, -num_filters:] * (evals[-num_filters:] ** 0.25)


class SpectralFilter(Filter):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_filters: int = 24,
        spectral_history_length: int = 100,
    ):
        super().__init__(
            spectral_history_length,
            obs_dim,
            action_dim,
        )

        self.filtered_obs_dim = num_filters * obs_dim
        self.filtered_action_dim = action_dim
        self.num_filters = num_filters
        self.filters = get_filters(spectral_history_length, num_filters)

    def __call__(
        self, obs_history: Array, action_history: Array
    ) -> Tuple[Array, Array]:
        spectral_features = jnp.einsum("mh,md->hd", self.filters, obs_history)

        # Reshape from (h, d, 1) to (h*d,)
        spectral_features = spectral_features.reshape(-1)

        return spectral_features, self._noop(action_history)
