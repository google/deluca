import functools
from typing import Callable, Dict, Tuple, Any

import flax.struct
import jax
import chex
import optax

from deluca.core import Env
from deluca.memory import Memory, MemorySettings
from deluca.normalizers.core import Normalizers, WithoutNormalization
from deluca.utils.printing import Task
from abc import abstractmethod
import jax.numpy as jnp
from jax import Array
import flax.nnx as nnx

History = Tuple[Array, Array]
LossFunction = Callable[[Array, Any, Array], Array | chex.Array]


@flax.struct.dataclass
class AgentSettings:
    loss_fn: LossFunction
    train_kwargs: Dict[str, Any] | None

    optimizer: optax.GradientTransformation = flax.struct.field(
        pytree_node=False,
    )


DefaultSettings = AgentSettings(
    optimizer=optax.chain(
        optax.adam(1e-3),
        optax.clip_by_global_norm(1.0),
    ),
    loss_fn=(
        lambda action, next_state, next_obs: action.T @ action + next_obs.T @ next_obs
    ),
    train_kwargs=None,
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
    memory_settings: MemorySettings

    normalizers: Normalizers
    optimizer: nnx.Optimizer

    def __init__(
        self,
        memory_settings: MemorySettings,
        settings: AgentSettings,
        rng: Array,
        normalizers: Normalizers | None = None,
    ):
        self.obs_dim_in = memory_settings.obs_dim_in
        self.action_dim_in = memory_settings.action_dim_in
        self.obs_dim_out = memory_settings.obs_dim_out
        self.action_dim_out = memory_settings.action_dim_out
        self.history_length = memory_settings.history_length

        self.settings = settings

        self.normalizers = normalizers or WithoutNormalization()

        self.memory_settings = memory_settings

        # Cache of compiled loss/grad functions keyed by (env_id, N, T)
        self._compiled_loss_cache: dict[tuple[int, int, int], Any] = {}

        # Cache of compiled train_step functions keyed by (env_id)
        self._compiled_train_step_cache: dict[int, Any] = {}

    def _get_compiled_loss_fn(self, env: Env, N: int, T: int):
        """Return a cached `loss_and_grad` function specialised to (env, N, T)."""
        cache_key = (id(env), N, T)
        fn = self._compiled_loss_cache.get(cache_key)
        if fn is not None:
            return fn

        def _NT_loss(model: AgentModel, rng: Array):
            return jnp.mean(
                jax.vmap(
                    lambda k: _episode_loss(
                        model,
                        self.memory_settings,
                        env,
                        self.settings.loss_fn,
                        self._preprocess_history,
                        self._postprocess_action,
                        T,
                        k,
                    )
                )(rng)
            )

        fn = nnx.jit(nnx.value_and_grad(_NT_loss))
        self._compiled_loss_cache[cache_key] = fn
        return fn
    
    def _get_compiled_train_step(self, env: Env):
        """Return a cached `train_step` function specialised to (env)."""
        cache_key = (id(env))
        fn = self._compiled_train_step_cache.get(cache_key)
        if fn is not None:
            return fn
        
        def _ts(model: AgentModel, memory: Memory, rng: Array):
            return _train_step(model, env, memory, self.settings.loss_fn, self._preprocess_history, self._postprocess_action, rng)
        
        fn = nnx.jit(nnx.value_and_grad(_ts, has_aux=True))
        self._compiled_train_step_cache[cache_key] = fn
        return fn

    def train(self, env: Env, N: int, T: int, rng: Array) -> Array:
        """
        Train the agent on the environment. Runs N episodes of length T.
        Loss is the average of the loss over each step of each episode.

        Args:
            env: The environment to train on.
            N: The number of episodes to run.
            T: The length of each episode.
            rng: A JAX PRNG key

        Returns:
            The average loss over all episodes.
        """

        loss_and_grad = self._get_compiled_loss_fn(env, N, T)

        rng, split_key = jax.random.split(rng)
        episode_keys = jax.random.split(split_key, N)

        loss, grads = loss_and_grad(self.model, episode_keys)
        self.optimizer.update(grads)

        return loss

    def train_step(self, env: Env, memory: Memory, rng: Array) -> Tuple[Array, Memory]:
        """
        Trains the environment on one step of the environment at the given memory.

        Args:
            env: The environment to train on.
            memory: The memory to train on. Should be primed and
                    updated with the current env state.
            rng: A JAX PRNG key

        Returns:
            loss: loss from the update
            new_memory: memory after the step
        """

        (loss, new_memory), grads = self._get_compiled_train_step(env)(self.model, memory, rng)
        self.optimizer.update(grads)

        return loss, new_memory

    def _preprocess_history(self, history: History) -> History:
        """
        Preprocess a single history for prediction.
        For prediction, we expect a history to be a tuple of (obs_history, action_history)
        """

        assert (
            len(history) == 2
        ), "History must be a tuple of (obs_history, action_history)"

        obs_history, action_history = history

        assert obs_history.ndim == action_history.ndim and obs_history.ndim in [
            2,
            3,
        ], "Histories must be 2D or 3D (batched) arrays"

        hist_index = 0 if obs_history.ndim == 2 else 1

        assert (
            obs_history.shape[hist_index]
            == action_history.shape[hist_index]
            == self.history_length
        ), "obs and actions must have length history_length"
        assert (
            obs_history.shape[hist_index + 1] == self.obs_dim_in
        ), "obs must have obs_dim_in dimensions"
        assert (
            action_history.shape[hist_index + 1] == self.action_dim_in
        ), "actions must have action_dim_in dimensions"

        obs_history = self.normalizers.normalize_obs(obs_history)
        action_history = self.normalizers.normalize_action(action_history)

        return obs_history, action_history

    def _postprocess_action(self, action: Array) -> Array:
        """
        Postprocess an action for prediction.
        """
        action = self.normalizers.denormalize_action(action)
        action = jnp.expand_dims(action, -1)  # Add a singleton dimension to the action

        return action

    def __call__(self, history: History, rng: Array) -> Array:
        """Given the history of most recent observations and actions, return the next action.
        Note: if histories are longer than history_length, the oldest observations and actions are ignored.
        """

        obs_history, action_history = self._preprocess_history(history)

        pred_action = self.model(obs_history, action_history, rng)
        pred_action = self._postprocess_action(pred_action)

        return pred_action


def _episode_loss(
    model: AgentModel,
    mem_settings: MemorySettings,
    env: Env,
    loss_fn: LossFunction,
    hist_preprocessor: Callable[[History], History],
    action_postprocessor: Callable[[Array], Array],
    T: int,
    rng: Array,
) -> Array:
    """Rolls out one episode and returns the mean loss."""

    def _step(
        memory: Memory,
        step_key: Array,
    ):
        """Single environment transition used inside `jax.lax.scan`."""

        obs_hist, act_hist = hist_preprocessor(memory.get_history())

        action_raw = model._train_call(obs_hist, act_hist, step_key)
        action = action_postprocessor(action_raw)

        memory = memory.act(env, action, step_key)

        step_loss = loss_fn(jnp.squeeze(action, -1), memory.state, memory.obs)

        return memory, step_loss

    memory = Memory(mem_settings).reset_env(env, rng)

    _, losses = jax.lax.scan(_step, memory, jax.random.split(rng, T))
    return jnp.sum(losses)


def _train_step(
    model: "AgentModel",
    env: Env,
    memory: Memory,
    loss_fn: LossFunction,
    hist_preprocessor: Callable[[History], History],
    action_postprocessor: Callable[[Array], Array],
    rng: jax.Array,
):
    """Inner JIT-compiled step returning (loss, new_mem)."""
    obs_hist, act_hist = hist_preprocessor(memory.get_history())

    raw_action = model._train_call(obs_hist, act_hist, rng)
    action = action_postprocessor(raw_action)

    t, state, obs = env(memory.t, memory.state, action, rng)
    loss = loss_fn(jnp.squeeze(action, -1), state, jnp.squeeze(obs, -1))

    return loss, memory.resolve(action).prime(obs)
