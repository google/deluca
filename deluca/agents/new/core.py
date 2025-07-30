import functools
from typing import Callable, Dict, Tuple, Any

import flax.struct
import jax
import chex

from deluca.core import Env
from deluca.learners.memory import Memory
from deluca.normalizers.core import Normalizers, WithoutNormalization
from deluca.utils.printing import Task
from abc import abstractmethod
import jax.numpy as jnp
from jax import Array
import flax.nnx as nnx

History = Tuple[Array, Array]

@flax.struct.dataclass
class AgentSettings:
    learning_rate: float
    momentum: float

    loss_fn: Callable[[Array, Any, Array], Array | chex.Array]
    train_kwargs: Dict[str, Any] | None

DefaultSettings = AgentSettings(
    learning_rate=1e-4,
    momentum=0.0,
    loss_fn=(lambda action, next_state, next_obs: action.T @ action + next_obs.T @ next_obs),
    train_kwargs=None
)

class AgentModel(nnx.Module):
    @abstractmethod
    def __call__(self, obs: Array, action: Array, rng: Array) -> Array:
        """Predict the next action given the history of most recent observations and actions."""
    
    @abstractmethod
    def _train_call(self, obs: Array, action: Array, rng: Array) -> Array:
        """Separate function if needed for training."""

class Agent:
    name: str

    model: AgentModel

    obs_dim_in: int
    action_dim_in: int
    obs_dim_out: int
    action_dim_out: int
    history_length: int

    normalizers: Normalizers
    optimizer: nnx.Optimizer

    def __init__(self, memory: Memory, settings: AgentSettings, rng: Array, normalizers: Normalizers | None = None):
        self.obs_dim_in = memory.obs_dim_in
        self.action_dim_in = memory.action_dim_in
        self.obs_dim_out = memory.obs_dim_out
        self.action_dim_out = memory.action_dim_out
        self.history_length = memory.history_length

        self.settings = settings

        self.normalizers = normalizers or WithoutNormalization()

    @staticmethod
    def _loss_fn_wrapper(model, env_call, settings, normalizers, env_t, env_state, rng, obs_history, action_history):
        action = model._train_call(obs_history, action_history, rng)
        
        # Postprocess the action
        denormalized_action = normalizers.denormalize_action(action)
        action_post = jnp.expand_dims(denormalized_action, -1)

        new_t, new_state, new_obs = env_call(env_t, env_state, action_post, rng)
        loss = settings.loss_fn(action_post, new_state, new_obs)
        return jnp.mean(loss), ((new_t, new_state, new_obs), action_post)

    @staticmethod
    @functools.partial(nnx.jit, static_argnums=(2, 3, 4)) # env_call, settings, normalizers
    def _update(model, optimizer, env_call, settings, normalizers, env_t, env_state, rng, obs_history, action_history):
        (loss, aux), grads = nnx.value_and_grad(Agent._loss_fn_wrapper, has_aux=True)(
            model, env_call, settings, normalizers, env_t, env_state, rng, obs_history, action_history
        )
        optimizer.update(grads)
        return loss, aux

    def step_and_update(self, env: Env, env_t: int, env_state, history: History, rng: Array) -> Tuple[Array, Tuple[int, Any, Array], Array]:
        """
        Update the agent's model using the history.

        Returns:
            loss: The loss of the agent's model.
            (new_t, new_state, new_obs): The new time step, state, and observation of the environment.
            action: The action taken by the agent.
        """
        obs_history, action_history = self._preprocess_history(history)
        loss, ((new_t, new_state, new_obs), action) = Agent._update(
            self.model, self.optimizer, env.__call__, self.settings, self.normalizers,
            env_t, env_state, rng, obs_history, action_history
        )
        return loss, (new_t, new_state, new_obs), action


    def _preprocess_history(self, history: History) -> History:
        """
        Preprocess a single history for prediction.
        For prediction, we expect a history to be a tuple of (obs_history, action_history)
        """

        assert len(history) == 2, "History must be a tuple of (obs_history, action_history)"

        obs_history, action_history = history

        assert obs_history.ndim == action_history.ndim and obs_history.ndim in [2, 3], "Histories must be 2D or 3D (batched) arrays"

        hist_index = 0 if obs_history.ndim == 2 else 1

        assert obs_history.shape[hist_index] == action_history.shape[hist_index] == self.history_length, "obs and actions must have length history_length"
        assert obs_history.shape[hist_index + 1] == self.obs_dim_in, "obs must have obs_dim_in dimensions"
        assert action_history.shape[hist_index + 1] == self.action_dim_in, "actions must have action_dim_in dimensions"
        
        obs_history = self.normalizers.normalize_obs(obs_history)
        action_history = self.normalizers.normalize_action(action_history)

        return obs_history, action_history
    
    
        
    def __call__(self, history: History, rng: Array) -> Array:
        """Given the history of most recent observations and actions, return the next action.
        Note: if histories are longer than history_length, the oldest observations and actions are ignored.
        """

        obs_history, action_history = self._preprocess_history(history)

        pred_action = self.model(obs_history, action_history, rng)

        # Add a singleton dimension to the action
        pred_action = jnp.expand_dims(pred_action, -1)  

        return pred_action