from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp
from jax import Array

from deluca.core import Env, Agent
from deluca.filters.spectral import SpectralFilter
from deluca.learners.memory import Memory


def generate_trajectories(
    env: Env, agent: Agent, N: int, T: int, rng: Array
) -> Tuple[Array, Array, Array]:
    """Generate N trajectories of length T."""

    @jax.jit
    def generate_trajectory(rng: Array) -> Tuple[Array, Array, Array]:

        def step_fn(carry, rng):
            t, state, obs = carry
            action = agent((obs, obs), rng) # TODO: update for new agent interface
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
            action = jax.random.normal(rng, shape=(2,1))
            new_t, new_state, new_obs = t + 1, obs[1][0], jnp.array([[state], [action[0][0]]])
            return (new_t, new_state, new_obs), (obs, action, new_obs)

        init_carry = (0, 0.32, jnp.array([[1.0], [0.2]]))
        _, steps = jax.lax.scan(step_fn, init_carry, jax.random.split(rng, T))

        return steps

    return jax.vmap(generate_trajectory)(jax.random.split(rng, N))