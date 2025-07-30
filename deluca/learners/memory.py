from functools import partial
from typing import Tuple
from jax import Array
import jax
import jax.numpy as jnp

from deluca.filters.core import Filter


class Memory:
    """
    Memory is a class that stores the unfolding of a trajectory, and can be used to generate histories,
    which are arrays of a fixed number of previous observations and actions up to the most recent.

    Memory accepts a filter (eg. spectral filter) that can influence the way the history is generated.
    """

    _history_length: int

    _obs_dim: int
    _action_dim: int

    obs_memory: Array
    action_memory: Array

    _filtered_obs_memory: Array
    _filtered_action_memory: Array

    filter: Filter | None

    def __init__(
        self,
        history_length: int,
        obs_dim: int,
        action_dim: int,
        filter: Filter | None = None,
    ):
        """
        Args:
            memory_capacity: int, the number of observations and actions to store in the Memory
            obs_dim: int, the dimension of the observation
            action_dim: int, the dimension of the action
            filter: Filter | None
        """
        self._history_length = history_length
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self.filter = filter

        real_history_length = history_length

        if filter is not None and filter.filter_history_length > history_length:
            real_history_length = filter.filter_history_length

        self.obs_memory = jnp.zeros((real_history_length, obs_dim))
        self.action_memory = jnp.zeros((real_history_length, action_dim))

        self._filtered_obs_memory = jnp.zeros((real_history_length, filter.filtered_obs_dim if filter is not None else obs_dim))
        self._filtered_action_memory = jnp.zeros((real_history_length, filter.filtered_action_dim if filter is not None else action_dim))

    def store(self, obs: Array, action: Array):
        """
        Store the observation and action in the memory. The most recent observation is stored at the last index of the memory.
        """

        # Remove singleton dimension from obs and action
        obs = jnp.squeeze(obs, -1)
        action = jnp.squeeze(action, -1)

        self.obs_memory = jnp.roll(self.obs_memory, -1, axis=0)
        self.action_memory = jnp.roll(self.action_memory, -1, axis=0)

        self.obs_memory = self.obs_memory.at[-1].set(obs)
        self.action_memory = self.action_memory.at[-1].set(action)

        if self.filter is None:
            self._filtered_obs_memory = self.obs_memory
            self._filtered_action_memory = self.action_memory
        else:
            self._filtered_obs_memory = jnp.roll(self._filtered_obs_memory, -1, axis=0)
            self._filtered_action_memory = jnp.roll(self._filtered_action_memory, -1, axis=0)

            filtered_obs, filtered_action = self.filter(self.obs_memory[-self.filter.filter_history_length:], self.action_memory[-self.filter.filter_history_length:])

            self._filtered_obs_memory = self._filtered_obs_memory.at[-1].set(filtered_obs)
            self._filtered_action_memory = self._filtered_action_memory.at[-1].set(filtered_action)

    def reset(self):
        """
        Reset the memory to the initial state.
        """
        self.obs_memory = jnp.zeros((self._history_length, self._obs_dim))
        self.action_memory = jnp.zeros((self._history_length, self._action_dim))

    def get_history(self):
        """
        Get the history from the memory. If a filter is provided, it will be applied to the history.
        The last index of a history is the most recent observation/action. The history is padded with zeros to the left.

        Returns:
            obs_history: Array
            action_history: Array
        """

        return (
            self._filtered_obs_memory[-self._history_length :],
            self._filtered_action_memory[-self._history_length :],
        )
   
    def from_trajectories(
        self,
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

        fhl = self.filter.filter_history_length if self.filter is not None else 0
        ihl = (
            fhl if fhl > self.history_length else self.history_length
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
            output_action_dim = filter.filtered_action_dim if filter is not None else action_dim

            init_carry = (jnp.zeros((ihl, output_obs_dim)), jnp.zeros((ihl, output_action_dim)))
            _, (obs_history, action_history, next_obs_history) = jax.lax.scan(
                step_fn, init_carry, jnp.arange(ihl - 1, obs.shape[0])
            )

            if remove_incomplete_histories:
                obs_history = obs_history[history_length - 1:]
                action_history = action_history[history_length - 1:]
                next_obs_history = next_obs_history[history_length - 1:]

            return obs_history, action_history, next_obs_history

        obs_hist, action_hist, next_obs_hist = jax.vmap(
            _create_history_from_one_trajectory, in_axes=(0, 0, 0, None, None, None)
        )(
            all_obs,
            all_actions,
            all_next_obs,
            self.history_length,
            remove_incomplete_histories,
            self.filter,
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
    
    @property
    def obs_dim_in(self):
        """
        The dimension of the observations after filtering. Models should take in observations of this dimension.
        """
        return self.filter.filtered_obs_dim if self.filter is not None else self._obs_dim
    
    @property
    def action_dim_in(self):
        """
        The dimension of the actions after filtering. Models should take in actions of this dimension.
        """
        return self.filter.filtered_action_dim if self.filter is not None else self._action_dim


    @property
    def obs_dim_out(self):
        """
        The true dimension of the environment observation space. Models should output predictions in this space.
        """
        return self._obs_dim
    
    @property
    def action_dim_out(self):
        """
        The true dimension of the environment action space. Models should output decisions in this space.
        """
        return self._action_dim
    
    @property
    def history_length(self):
        return self._history_length

