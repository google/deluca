from functools import partial
from typing import Tuple
import warnings
import jax
import jax.numpy as jnp
from jax import Array

from deluca.core import Env, Agent
from deluca.filters.spectral import SpectralFilter
from deluca.memory import MemorySettings


@warnings.deprecated(
    "Works on the old agent interface. New agents use deluca.memory.generate_histories"
)
def generate_trajectories(
    env: Env, agent: Agent, N: int, T: int, rng: Array
) -> Tuple[Array, Array, Array]:
    """Generate N trajectories of length T."""

    @jax.jit
    def generate_trajectory(rng: Array) -> Tuple[Array, Array, Array]:

        def step_fn(carry, rng):
            t, state, obs = carry
            action = agent(obs, rng)
            new_t, new_state, new_obs = env(t, state, action, rng)
            return (new_t, new_state, new_obs), (obs, action, new_obs)

        init_carry = env.reset(rng)
        _, steps = jax.lax.scan(step_fn, init_carry, jax.random.split(rng, T))

        return steps

    return jax.vmap(generate_trajectory)(jax.random.split(rng, N))


def generate_simple_trajectories(N, T, rng):
    """Generates N trajectories of length T, where next_obs = action, and action is normally distributed."""

    @jax.jit
    def generate_trajectory(rng: Array) -> Tuple[Array, Array, Array]:

        def step_fn(carry, rng):
            t, state, obs = carry
            action = jax.random.normal(rng, shape=(2, 1))
            new_t, new_state, new_obs = (
                t + 1,
                obs[1][0],
                jnp.array([[state], [action[0][0]]]),
            )
            return (new_t, new_state, new_obs), (obs, action, new_obs)

        init_carry = (0, 0.32, jnp.array([[1.0], [0.2]]))
        _, steps = jax.lax.scan(step_fn, init_carry, jax.random.split(rng, T))

        return steps

    return jax.vmap(generate_trajectory)(jax.random.split(rng, N))


def from_trajectories(
    memory_settings: MemorySettings,
    trajectories: Tuple[Array, Array, Array],
    remove_incomplete_histories: bool = True,
):
    """
    Convert pure trajectories (obs, action, next_obs) into histories.
    
    A pure trajectory is a tuple of arrays where eg. `obs[n][t]` is the observation at time `t` for trajectory `n`.

    A history is a tuple of arrays where eg. `obs_history[i]` is a record of history_length previous observations
    up the current one at the time step when that history was generated. Histories are not grouped by trajectory.

    `obs` must be of shape (N, T, obs_dim, 1)
    `action` must be of shape (N, T, action_dim, 1)
    `next_obs` must be of shape (N, T, obs_dim, 1)

    Returns:
        obs_histories: (N*T, history_length, filtered_obs_dim)
        action_histories: (N*T, history_length, filtered_action_dim)
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
        fhl if fhl > memory_settings.history_length else memory_settings.history_length
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
            new_action_history = jnp.roll(prev_action_history, -1, axis=0).at[-1].set(a)

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
