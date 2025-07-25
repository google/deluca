import flax.struct
import jax
import jax.numpy as jnp
from jax import Array
import flax.nnx as nnx
import optax

from deluca.core import Env
from deluca.learners.core import Learner, LearnerModel, LearnerSettings, DefaultSettings as DefaultLearnerSettings, Normalizers
from deluca.learners.memory import Memory

class _FFModel(LearnerModel):
    def __init__(self, history_length: int, hidden_size: int, obs_dim_in: int, obs_dim_out: int, action_dim: int, rngs: nnx.Rngs):
        self.H1 = nnx.Linear((obs_dim_in + action_dim) * history_length, hidden_size, rngs=rngs)
        self.H2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.O = nnx.Linear(hidden_size, obs_dim_out, rngs=rngs)


    def __call__(self, obs: Array, action: Array) -> Array:
        # Concatenate observation and action along the last dimension
        input = jnp.concatenate([obs, action], axis=-1)

        # Flatten history dimension
        if (obs.ndim == 3):
            input = input.reshape(obs.shape[0], -1)
        elif (input.ndim == 2):
            input = input.reshape(-1)

        x = self.H1(input)
        x = nnx.relu(x)

        x = self.H2(x)
        x = nnx.relu(x)

        return self.O(x)
    
    def _train_call(self, obs: Array, action: Array) -> Array:
        return self(obs, action)
    
@flax.struct.dataclass
class FFLearnerSettings(LearnerSettings):
    hidden_size: int = 64

DefaultSettings = FFLearnerSettings(**DefaultLearnerSettings.__dict__)

class FFLearner(Learner):
    name = "FeedForward Learner"
    settings: FFLearnerSettings
    
    def __init__(self, memory: Memory, settings: FFLearnerSettings, rng: Array, normalizers: Normalizers | None = None):
        super().__init__(memory, settings, rng, normalizers)
        
        self.model = _FFModel(memory.history_length, settings.hidden_size, memory.obs_dim_in, memory.obs_dim_out, memory.action_dim_in, nnx.Rngs(rng))
        self.loss_fn = lambda pred, next_obs: optax.squared_error(pred, next_obs)
        self.optimizer = nnx.Optimizer(self.model, optax.sgd(learning_rate=settings.learning_rate, momentum=settings.momentum))



    