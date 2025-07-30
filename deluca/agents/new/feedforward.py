import flax.struct
import jax
import jax.numpy as jnp
from jax import Array
import flax.nnx as nnx
import optax

from deluca.agents.new.core import Agent, AgentModel, AgentSettings, DefaultSettings as DefaultAgentSettings
from deluca.learners.memory import Memory
from deluca.normalizers.core import Normalizers

class _FFModel(AgentModel):
    def __init__(self, history_length: int, hidden_size: int, obs_dim_in: int, action_dim_in: int, action_dim_out: int, rngs):
        super().__init__()
        self.H1 = nnx.Linear((obs_dim_in + action_dim_in) * history_length, hidden_size, rngs=rngs)
        # self.H2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.O = nnx.Linear(hidden_size, action_dim_out, rngs=rngs)

    def __call__(self, obs_history: Array, action_history: Array, _rng: Array) -> Array:
        # Concatenate observation and action along the last dimension
        input = jnp.concatenate([obs_history, action_history], axis=-1)

        # Flatten history dimension
        if (action_history.ndim == 3):
            input = input.reshape(action_history.shape[0], -1) # Keep batch dimension
        elif (input.ndim == 2):
            input = input.reshape(-1)

        x = self.H1(input)
        x = nnx.relu(x)

        # x = self.H2(x)
        # x = nnx.relu(x)

        return self.O(x)
    
    def _train_call(self, obs: Array, action: Array, rng: Array) -> Array:
        return self.__call__(obs, action, rng)
    
@flax.struct.dataclass
class FFAgentSettings(AgentSettings):
    hidden_size: int
    grad_clip: float | None = None

DefaultSettings = FFAgentSettings(**DefaultAgentSettings.__dict__, hidden_size=24)

class FFAgent(Agent):
    name = "FF Agent"
    settings: FFAgentSettings
    
    def __init__(self, memory: Memory, settings: FFAgentSettings, rng: Array, normalizers: Normalizers | None = None):
        super().__init__(memory, settings, rng, normalizers)
        
        self.model = _FFModel(memory.history_length, settings.hidden_size, memory.obs_dim_in, memory.action_dim_in, memory.action_dim_out, nnx.Rngs(rng))
        optimizer = optax.sgd(learning_rate=settings.learning_rate, momentum=settings.momentum)
        if settings.grad_clip is not None:
            optimizer = optax.chain(
                optax.clip_by_global_norm(settings.grad_clip),
                optimizer
            )
        self.optimizer = nnx.Optimizer(self.model, optimizer)



    