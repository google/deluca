from functools import partial
from typing import Any, Tuple
import flax.struct
from jax import Array
import jax
import jax.numpy as jnp

from deluca.core import Env
from deluca.filters.core import Filter


@flax.struct.dataclass(frozen=True, slots=True)
class MemorySettings:
    """
    Holds the settings for a memory. Agents, Learners, and Memories should share consistent MemorySettings, as
    they will be used to decide properties like model parameter sizes and input/output dimensions.
    """

    history_length: int
    obs_dim: int
    action_dim: int
    filter: Filter | None

    @classmethod
    def from_env(
        cls, env: Env, history_length: int, filter: Filter | None = None
    ) -> "MemorySettings":
        """
        Create a MemorySettings object from an environment, history length, and a filter (optional).
        """
        return cls(
            history_length=history_length,
            obs_dim=env.observation_size,
            action_dim=env.action_size,
            filter=filter,
        )

    @property
    def obs_dim_in(self):
        """
        The dimension of the observations after filtering. Models should take in observations of this dimension.
        """
        return self.filter.filtered_obs_dim if self.filter is not None else self.obs_dim

    @property
    def action_dim_in(self):
        """
        The dimension of the actions after filtering. Models should take in actions of this dimension.
        """
        return (
            self.filter.filtered_action_dim
            if self.filter is not None
            else self.action_dim
        )

    @property
    def obs_dim_out(self):
        """
        The true dimension of the environment observation space. Models should output predictions in this space.
        """
        return self.obs_dim

    @property
    def action_dim_out(self):
        """
        The true dimension of the environment action space. Models should output decisions in this space.
        """
        return self.action_dim


@jax.tree_util.register_pytree_node_class
class Memory:
    """
    Memory is a class that stores the unfolding of a trajectory, and can be used to generate histories,
    which are arrays of a fixed number of previous observations and actions up to the most recent.

    Memory accepts a filter (eg. spectral filter) that can influence the way the history is generated.

    The memory treats action_i as the action taken *after* observing obs_i.
    """

    settings: MemorySettings

    obs_memory: Array
    action_memory: Array

    _filtered_obs_memory: Array
    _filtered_action_memory: Array

    # Current state variables
    _current_t: int
    _current_state: Any
    _current_obs: Array

    _unresolved: bool

    def __init__(
        self,
        spec: MemorySettings,
    ):
        """
        Args:
            memory_capacity: int, the number of observations and actions to store in the Memory
            obs_dim: int, the dimension of the observation
            action_dim: int, the dimension of the action
            filter: Filter | None
        """
        self.settings = spec

        self._unresolved = False

        self._real_history_length = spec.history_length

        if (
            spec.filter is not None
            and spec.filter.filter_history_length > spec.history_length
        ):
            self._real_history_length = spec.filter.filter_history_length

        self._init_buffers()

    def _init_buffers(self):
        rh = self._real_history_length
        od, ad = self.settings.obs_dim, self.settings.action_dim
        self.obs_memory = jnp.zeros((rh, od))
        self.action_memory = jnp.zeros((rh, ad))
        self._filtered_obs_memory = jnp.zeros(
            (rh, self.settings.filter.filtered_obs_dim if self.settings.filter else od)
        )
        self._filtered_action_memory = jnp.zeros(
            (
                rh,
                (
                    self.settings.filter.filtered_action_dim
                    if self.settings.filter
                    else ad
                ),
            )
        )
        self._current_t = 0
        self._current_state = None
        self._current_obs = jnp.zeros((od))

    def reset_env(self, env: Env, rng: Array) -> "Memory":
        """
        A helper function that resets the environment and updates the memory.
        """
        env_state = env.reset(rng)
        new_mem = self.clone().reset().prime(env_state[2])
        new_mem.env_state = env_state
        return new_mem

    def act(self, env: Env, action: Array, rng: Array) -> "Memory":
        """
        A helper function that takes an action in the environment and updates the memory.
        The memory must be primed with the current observation before calling this method.
        Returns a new memory primed with the next observation.
        """

        mem = self.resolve(action)

        t, state, obs = env(self.t, self.state, action, rng)

        mem = jax.lax.cond(
            t == 0,
            lambda m: m.reset(),  # reset history on episode boundaries/env resets
            lambda m: m,
            mem,
        )

        mem = mem.prime(obs)
        mem.env_state = (t, state, obs)

        return mem

    def prime(self, obs: Array) -> "Memory":
        """
        Add the next observation into memory in preparation for an agent to take an action.
        Since we only have obs_i and are waiting for action_i, we store the a placeholder action (a_i-1) as the "current" action.
        Used in conjunction with resolve() to replace the placeholder action.

        Returns:
            A new Memory instance
        """

        assert (
            not self._unresolved
        ), "Trying to prime a memory that is already unresolved from a previous call to prime()."

        updated = self.store(obs, self.action_memory[-1])
        updated._unresolved = True
        return updated

    def resolve(self, action: Array) -> "Memory":
        """
        Replaces the most recent action with the given action. Used in conjunction with prime() to replace the placeholder action.

        Returns:
            A new Memory instance
        """

        assert (
            self._unresolved
        ), "Trying to resolve a memory that has not been primed to receive an action."

        # Remove singleton dimension
        action = jnp.squeeze(action, -1)

        # Update raw action buffer
        new_action_memory = self.action_memory.at[-1].set(action)

        if self.settings.filter is None:
            new_f_obs_mem = self._filtered_obs_memory
            new_f_act_mem = self._filtered_action_memory.at[-1].set(action)
        else:
            # Recompute filter on the last `filter_history_length` window
            k = self.settings.filter.filter_history_length
            obs_window = self.obs_memory[-k:]
            act_window = new_action_memory[-k:]

            f_obs, f_act = self.settings.filter(obs_window, act_window)

            new_f_obs_mem = self._filtered_obs_memory.at[-1].set(f_obs)
            new_f_act_mem = self._filtered_action_memory.at[-1].set(f_act)

        updated = self.clone()
        updated.action_memory = new_action_memory
        updated._filtered_obs_memory = new_f_obs_mem
        updated._filtered_action_memory = new_f_act_mem
        updated._unresolved = False
        return updated

    def store(self, obs: Array, action: Array) -> "Memory":
        """
        Store the observation and action in the memory. The most recent observation is stored at the last index of the memory.

        Returns:
            A new Memory instance
        """

        # Strip trailing singleton dimension added by env/agent
        obs = jnp.squeeze(obs, -1)
        action = jnp.squeeze(action, -1)

        # Roll buffers and insert latest entry
        new_obs_memory = jnp.roll(self.obs_memory, -1, axis=0).at[-1].set(obs)
        new_action_memory = jnp.roll(self.action_memory, -1, axis=0).at[-1].set(action)

        # Apply filter if present
        if self.settings.filter is None:
            new_f_obs, new_f_act = new_obs_memory, new_action_memory
        else:
            new_f_obs_mem = jnp.roll(self._filtered_obs_memory, -1, axis=0)
            new_f_act_mem = jnp.roll(self._filtered_action_memory, -1, axis=0)

            f_obs, f_act = self.settings.filter(
                new_obs_memory[-self.settings.filter.filter_history_length :],
                new_action_memory[-self.settings.filter.filter_history_length :],
            )

            new_f_obs = new_f_obs_mem.at[-1].set(f_obs)
            new_f_act = new_f_act_mem.at[-1].set(f_act)

        # Build and return new instance carrying the updated buffers
        updated = self.clone()
        updated.obs_memory = new_obs_memory
        updated.action_memory = new_action_memory
        updated._filtered_obs_memory = new_f_obs
        updated._filtered_action_memory = new_f_act
        return updated

    def reset(self) -> "Memory":
        """Returns a new Memory with cleared buffers."""

        # Preserve current_state to preserve treedef
        # Effectively not a problem since environment state will be updated
        # before the memory is used to do anything.
        cleared = Memory(self.settings)
        cleared._current_state = self._current_state
        cleared._current_obs = self._current_obs
        cleared._current_t = self._current_t
        return cleared

    def get_history(self):
        """
        Get the history from the memory. If a filter is provided, it will be applied to the history.
        The last index of a history is the most recent observation/action. The history is padded with zeros to the left.

        Returns:
            obs_history: Array
            action_history: Array
        """

        return (
            self._filtered_obs_memory[-self.settings.history_length :],
            self._filtered_action_memory[-self.settings.history_length :],
        )

    @staticmethod
    def from_trajectories(
        memory_settings: MemorySettings,
        trajectories: Tuple[Array, Array, Array],
        remove_incomplete_histories: bool = True,
    ):
        """
        Convert trajectories (obs, action, next_obs) into histories.

        obs must be of shape (N, T, obs_dim, 1)
        action must be of shape (N, T, action_dim, 1)
        next_obs must be of shape (N, T, obs_dim, 1)

        Returns:
            obs_histories (N*T, history_length, filtered_obs_dim)
            action_histories (N*T, history_length, filtered_action_dim)
        """

        all_obs, all_actions, all_next_obs = trajectories

        # all_obs are of shape (N, T, obs_dim, 1) -> Remove singleton dimension
        # This is because environment observations and agent actions have singleton dimensions.
        all_obs = jnp.squeeze(all_obs, -1)
        all_actions = jnp.squeeze(all_actions, -1)
        all_next_obs = jnp.squeeze(all_next_obs, -1)

        fhl = (
            memory_settings.filter.filter_history_length
            if memory_settings.filter is not None
            else 0
        )
        ihl = (
            fhl
            if fhl > memory_settings.history_length
            else memory_settings.history_length
        )  # Intermediate history length, for pre-filtering

        @partial(jax.jit, static_argnums=(3, 4, 5))
        def _create_history_from_one_trajectory(
            obs, action, next_obs, history_length, remove_incomplete_histories, filter
        ):

            # obs, action, and next_obs must be padded with ihl - 1 zeros in the beginning.
            # So they have a constant shape when passed to the filter.
            obs = jnp.pad(obs, ((ihl - 1, 0), (0, 0)))
            action = jnp.pad(action, ((ihl - 1, 0), (0, 0)))
            next_obs = jnp.pad(next_obs, ((ihl - 1, 0), (0, 0)))

            obs_dim = obs.shape[-1]
            action_dim = action.shape[-1]

            def step_fn(carry, t):
                prev_obs_history, prev_action_history = carry

                o = obs[t]
                a = action[t]
                n = next_obs[t]

                # Filtering
                if filter is not None:
                    # Get obs and action history window (length: fhl) to filter
                    # May not be the same window size as the history_length we want.
                    obs_history_for_filter, action_history_for_filter = (
                        jax.lax.dynamic_slice_in_dim(obs, t - fhl + 1, fhl, axis=0),
                        jax.lax.dynamic_slice_in_dim(action, t - fhl + 1, fhl, axis=0),
                    )
                    o, a = filter(obs_history_for_filter, action_history_for_filter)

                new_obs_history = jnp.roll(prev_obs_history, -1, axis=0).at[-1].set(o)
                new_action_history = (
                    jnp.roll(prev_action_history, -1, axis=0).at[-1].set(a)
                )

                return (new_obs_history, new_action_history), (
                    new_obs_history[-history_length:],
                    new_action_history[-history_length:],
                    n,
                )

            output_obs_dim = filter.filtered_obs_dim if filter is not None else obs_dim
            output_action_dim = (
                filter.filtered_action_dim if filter is not None else action_dim
            )

            init_carry = (
                jnp.zeros((ihl, output_obs_dim)),
                jnp.zeros((ihl, output_action_dim)),
            )
            _, (obs_history, action_history, next_obs_history) = jax.lax.scan(
                step_fn, init_carry, jnp.arange(ihl - 1, obs.shape[0])
            )

            if remove_incomplete_histories:
                obs_history = obs_history[history_length - 1 :]
                action_history = action_history[history_length - 1 :]
                next_obs_history = next_obs_history[history_length - 1 :]

            return obs_history, action_history, next_obs_history

        obs_hist, action_hist, next_obs_hist = jax.vmap(
            _create_history_from_one_trajectory, in_axes=(0, 0, 0, None, None, None)
        )(
            all_obs,
            all_actions,
            all_next_obs,
            memory_settings.history_length,
            remove_incomplete_histories,
            memory_settings.filter,
        )

        # Flattening
        obs_hist = obs_hist.reshape(
            -1, *obs_hist.shape[-2:]
        )  # (N, T, history_length, obs_dim) -> (N*T, history_length, obs_dim)
        action_hist = action_hist.reshape(
            -1, *action_hist.shape[-2:]
        )  # (N, T, history_length, action_dim) -> (N*T, history_length, action_dim)
        next_obs_hist = next_obs_hist.reshape(
            -1, next_obs_hist.shape[-1]
        )  # (N, T, obs_dim) -> (N*T, obs_dim)

        return obs_hist, action_hist, next_obs_hist

    def clone(self) -> "Memory":
        """
        Create a copy of the memory with the same state.
        """

        new_mem = Memory(self.settings)

        new_mem.obs_memory = self.obs_memory.copy()
        new_mem.action_memory = self.action_memory.copy()
        new_mem._filtered_obs_memory = self._filtered_obs_memory.copy()
        new_mem._filtered_action_memory = self._filtered_action_memory.copy()
        new_mem._current_t = self._current_t
        new_mem._current_state = self._current_state
        new_mem._current_obs = self._current_obs
        new_mem._unresolved = self._unresolved

        return new_mem

    def tree_flatten(self):
        return (
            self.obs_memory,
            self.action_memory,
            self._filtered_obs_memory,
            self._filtered_action_memory,
            self._current_t,
            self._current_state,
            self._current_obs,
        ), (self.settings, self._unresolved)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        settings, unresolved = aux_data
        obj = cls(settings)
        (
            obj.obs_memory,
            obj.action_memory,
            obj._filtered_obs_memory,
            obj._filtered_action_memory,
            obj._current_t,
            obj._current_state,
            obj._current_obs,
        ) = children

        obj._unresolved = unresolved
        return obj

    @property
    def env_state(self):
        """
        The current state of the trajectory. This is a tuple of (t, state, obs).
        """
        return (self._current_t, self._current_state, self._current_obs)

    @env_state.setter
    def env_state(self, env_state: Tuple[int, Any, Array]):
        self._current_t, self._current_state, self._current_obs = env_state

    # Spec Passthrough Properties
    @property
    def obs_dim_in(self):
        return self.settings.obs_dim_in

    @property
    def action_dim_in(self):
        return self.settings.action_dim_in

    @property
    def obs_dim_out(self):
        return self.settings.obs_dim_out

    @property
    def action_dim_out(self):
        return self.settings.action_dim_out

    @property
    def history_length(self):
        return self.settings.history_length

    @property
    def t(self):
        """
        The current time step.
        """
        return self._current_t

    @property
    def state(self):
        """
        The current state of the environment.
        """
        return self._current_state

    @property
    def obs(self):
        """
        The current observation of the environment.
        """
        return self._current_obs
