import flax.struct
import jax
import jax.numpy as jnp
from jax import Array
import flax.nnx as nnx
import optax

from deluca.learners.core import (
    Learner,
    LearnerModel,
    LearnerSettings,
    DefaultSettings as DefaultLearnerSettings,
)
from deluca.memory import MemorySettings
from deluca.normalizers.core import NormalizerSet


class _LinearModel(LearnerModel):
    def __init__(
        self,
        history_length: int,
        obs_dim_in: int,
        obs_dim_out: int,
        action_dim_in: int,
        rngs,
    ):
        super().__init__()
        self.M = nnx.Param(jnp.zeros((history_length, obs_dim_out, obs_dim_in)))
        self.N = nnx.Param(jnp.zeros((history_length, obs_dim_out, action_dim_in)))
        self.b = nnx.Param(jnp.zeros(obs_dim_out))

    def __call__(self, obs_history: Array, action_history: Array, _rng: Array) -> Array:
        # y_t+1 = sum_i=0^h M_i * y_t-i + sum_i=0^h N_i * u_t-i + b

        original_shape = obs_history.shape
        if len(obs_history.shape) == 2:
            obs_history = obs_history[None, :, :]
            action_history = action_history[None, :, :]

        obs_contribution = jnp.einsum("ijk,bik->bj", self.M, obs_history)
        action_contribution = jnp.einsum("ijk,bik->bj", self.N, action_history)

        y_out = obs_contribution + action_contribution + self.b[jnp.newaxis, :]

        # If input was 2D, squeeze the output back to 2D
        if len(original_shape) == 2:
            y_out = y_out.squeeze(axis=0)

        return y_out

    def _train_call(self, obs: Array, action: Array, rng: Array) -> Array:
        return self.__call__(obs, action, rng)


@flax.struct.dataclass
class LinearLearnerSettings(LearnerSettings):
    pass


DefaultSettings = LinearLearnerSettings(**DefaultLearnerSettings.__dict__)


class LinearLearner(Learner):
    name = "Linear Learner"
    settings: LinearLearnerSettings

    def __init__(
        self,
        memory_settings: MemorySettings,
        settings: LinearLearnerSettings,
        rng: Array,
        normalizers: NormalizerSet | None = None,
    ):
        self.model = _LinearModel(
            memory_settings.history_length,
            memory_settings.obs_dim_in,
            memory_settings.obs_dim_out,
            memory_settings.action_dim_in,
            nnx.Rngs(rng),
        )

        super().__init__(memory_settings, settings, rng, normalizers)
