import flax.struct
import jax
import jax.numpy as jnp
from jax import Array
import flax.nnx as nnx
import optax

from deluca.memory import MemorySettings
from deluca.core import Env
from deluca.learners.core import (
    Learner,
    LearnerModel,
    LearnerSettings,
    DefaultSettings as DefaultLearnerSettings,
)
from deluca.memory import Memory
from deluca.normalizers.core import NormalizerSet


class _FFModel(LearnerModel):
    def __init__(
        self,
        history_length: int,
        hidden_size: int,
        obs_dim_in: int,
        obs_dim_out: int,
        action_dim: int,
        rngs: nnx.Rngs,
    ):
        self.H1 = nnx.Linear(
            (obs_dim_in + action_dim) * history_length, hidden_size, rngs=rngs
        )
        self.H2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.O = nnx.Linear(hidden_size, obs_dim_out, rngs=rngs)

    def __call__(self, obs_history: Array, action_history: Array, _rng: Array) -> Array:
        # Concatenate observation and action along the last dimension
        input = jnp.concatenate([obs_history, action_history], axis=-1)

        # Flatten history dimension
        if obs_history.ndim == 3:
            input = input.reshape(obs_history.shape[0], -1)  # Keep batch dimension
        elif input.ndim == 2:
            input = input.reshape(-1)

        x = self.H1(input)
        x = nnx.relu(x)

        x = self.H2(x)
        x = nnx.relu(x)

        return self.O(x)

    def _train_call(self, obs: Array, action: Array, rng: Array) -> Array:
        return self(obs, action, rng)


@flax.struct.dataclass
class FFLearnerSettings(LearnerSettings):
    hidden_size: int


DefaultSettings = FFLearnerSettings(**DefaultLearnerSettings.__dict__, hidden_size=64)


class FFLearner(Learner):
    name = "FeedForward Learner"
    settings: FFLearnerSettings

    def __init__(
        self,
        memory_settings: MemorySettings,
        settings: FFLearnerSettings,
        rng: Array,
        normalizers: NormalizerSet | None = None,
    ):
        self.model = _FFModel(
            memory_settings.history_length,
            settings.hidden_size,
            memory_settings.obs_dim_in,
            memory_settings.obs_dim_out,
            memory_settings.action_dim_in,
            nnx.Rngs(rng),
        )
        
        super().__init__(memory_settings, settings, rng, normalizers)
