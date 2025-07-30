from abc import abstractmethod
from jax import Array
import jax.numpy as jnp


class Normalizers:
    @abstractmethod
    def normalize_obs(
        self, obs: Array
    ) -> Array:
        """Normalize the input observation Array."""

    @abstractmethod
    def normalize_action(
        self, action: Array
    ) -> Array:
        """Normalize the input action Array."""

    @abstractmethod
    def denormalize_obs(
        self, obs: Array
    ) -> Array:
        """Denormalize the input observation Array."""

    @abstractmethod
    def denormalize_action(
        self, action: Array
    ) -> Array:
        """Denormalize the input action Array."""

    @abstractmethod
    def compute_normalization(self, obses: Array, actions: Array) -> None:
        """Compute the normalization of the input."""


class WithoutNormalization(Normalizers):
    def normalize_obs(self, obs: Array) -> Array:
        return obs

    def normalize_action(self, action: Array) -> Array:
        return action

    def denormalize_obs(self, obs: Array) -> Array:
        return obs

    def denormalize_action(self, action: Array) -> Array:
        return action

    def compute_normalization(self, obses, actions):
        pass


class DefaultNormalizers(Normalizers):
    _eps: float = 1e-8

    def normalize_obs(
        self,
        obs: Array,
    ) -> Array:
        return (obs - self.obs_mean) / (self.obs_std + self._eps)

    def normalize_action(
        self, action: Array
    ) -> Array:
        return (action - self.action_mean) / (self.action_std + self._eps)

    def denormalize_obs(
        self,
        obs: Array,
    ) -> Array:
        return obs * (self.obs_std + self._eps) + self.obs_mean

    def denormalize_action(
        self, action: Array
    ) -> Array:
        return action * (self.action_std + self._eps) + self.action_mean

    def compute_normalization(self, obses: Array, actions: Array) -> None:
        self.obs_mean = jnp.mean(obses, axis=0)
        self.obs_std = jnp.sqrt(jnp.var(obses, axis=0))
        self.action_mean = jnp.mean(actions, axis=0)
        self.action_std = jnp.sqrt(jnp.var(actions, axis=0))
