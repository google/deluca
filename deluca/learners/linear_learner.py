import flax.struct
import jax
import jax.numpy as jnp
from jax import Array
import flax.nnx as nnx
import optax

from deluca.core import Env
from deluca.learners.core import Learner, LearnerModel, LearnerSettings, DefaultSettings as DefaultLearnerSettings, Normalizers
from deluca.learners.memory import Memory

class _LinearModel(LearnerModel):
    def __init__(self, history_length: int, obs_dim_in: int, obs_dim_out: int, action_dim: int, rngs):
        super().__init__()
        self.history_length = history_length
        self.obs_dim = obs_dim_out
        self.action_dim = action_dim


        self.M = nnx.Param(jnp.zeros((history_length, obs_dim_out, obs_dim_in)))
        self.N = nnx.Param(jnp.zeros((history_length, obs_dim_out, action_dim)))
        self.b = nnx.Param(jnp.zeros(obs_dim_out))

    def __call__(self, obs_history: Array, action_history: Array) -> Array:
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
    
    def _train_call(self, obs: Array, action: Array) -> Array:
        return self.__call__(obs, action)
    
@flax.struct.dataclass
class LinearLearnerSettings(LearnerSettings):
    pass

DefaultSettings = LinearLearnerSettings(**DefaultLearnerSettings.__dict__)

class LinearLearner(Learner):
    name = "Linear Learner"
    settings: LinearLearnerSettings
    
    def __init__(self, memory: Memory, settings: LinearLearnerSettings, rng: Array, normalizers: Normalizers | None = None):
        super().__init__(memory, settings, rng, normalizers)
        
        self.model = _LinearModel(memory.history_length, memory.obs_dim_in, memory.obs_dim_out, memory.action_dim_in, nnx.Rngs(rng))
        self.loss_fn = lambda pred, next_obs: optax.squared_error(pred, next_obs)
        self.optimizer = nnx.Optimizer(self.model, optax.sgd(learning_rate=settings.learning_rate, momentum=settings.momentum))



    