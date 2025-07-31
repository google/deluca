import flax.struct
import jax
import jax.numpy as jnp
from jax import Array
import flax.nnx as nnx
import optax

from deluca.agents.new.core import Agent, AgentModel, AgentSettings, DefaultSettings as DefaultAgentSettings
from deluca.learners.memory import MemorySettings
from deluca.normalizers.core import Normalizers

class _LinearModel(AgentModel):
    def __init__(self, history_length: int, obs_dim_in: int, action_dim_out: int, rngs):
        super().__init__()
        # Small Gaussian initialisation helps escape the flat region around 0
        key_M, key_b = jax.random.split(rngs())
        self.M = nnx.Param(0.01 * jax.random.normal(key_M, (history_length, action_dim_out, obs_dim_in)))
        self.b = nnx.Param(jnp.zeros(action_dim_out))

    def __call__(self, obs_history: Array, _action_history: Array, _rng: Array) -> Array:
        # y_t+1 = sum_i=0^h M_i * O_t-i + b

        original_shape = obs_history.shape
        if len(obs_history.shape) == 2:
            obs_history = obs_history[None, :, :]

        y_out = jnp.einsum("ijk,bik->bj", self.M, obs_history) + self.b[jnp.newaxis, :]

        # If input was 2D, squeeze the output back to 2D
        if len(original_shape) == 2:
            y_out = y_out.squeeze(axis=0)

        return y_out
    
    def _train_call(self, obs: Array, action: Array, rng: Array) -> Array:
        return self.__call__(obs, action, rng)
    
@flax.struct.dataclass
class LinearAgentSettings(AgentSettings):
    pass

DefaultSettings = LinearAgentSettings(**DefaultAgentSettings.__dict__)

class LinearAgent(Agent):
    name = "Linear Agent"
    settings: LinearAgentSettings
    
    def __init__(self, memory_settings: MemorySettings, settings: LinearAgentSettings, rng: Array, normalizers: Normalizers | None = None):
        super().__init__(memory_settings, settings, rng, normalizers)
        
        self.model = _LinearModel(memory_settings.history_length, memory_settings.obs_dim_in, memory_settings.action_dim_out, nnx.Rngs(rng))
        self.optimizer = nnx.Optimizer(self.model, optax.sgd(learning_rate=settings.learning_rate, momentum=settings.momentum))



    