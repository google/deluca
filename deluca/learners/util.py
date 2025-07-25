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
            action = jax.random.normal(rng, shape=(2,1))
            new_t, new_state, new_obs = t + 1, obs[1][0], jnp.array([[state], [action[0][0]]])
            return (new_t, new_state, new_obs), (obs, action, new_obs)

        init_carry = (0, 0.32, jnp.array([[1.0], [0.2]]))
        _, steps = jax.lax.scan(step_fn, init_carry, jax.random.split(rng, T))

        return steps

    return jax.vmap(generate_trajectory)(jax.random.split(rng, N))

if __name__ == "__main__":
    trajectories = (
        [
            [
                [[0], [0], [1]],
                [[0], [0], [2]],
                [[0], [0], [3]],
                [[0], [0], [4]],
                [[0], [0], [5]],
            ],
            [
                [[0], [1], [1]],
                [[0], [1], [2]],
                [[0], [1], [3]],
                [[0], [1], [4]],
                [[0], [1], [5]],
            ],
            [
                [[0], [2], [1]],
                [[0], [2], [2]],
                [[0], [2], [3]],
                [[0], [2], [4]],
                [[0], [2], [5]],
            ],
            [
                [[0], [3], [1]],
                [[0], [3], [2]],
                [[0], [3], [3]],
                [[0], [3], [4]],
                [[0], [3], [5]],
            ],
            [
                [[0], [4], [1]],
                [[0], [4], [2]],
                [[0], [4], [3]],
                [[0], [4], [4]],
                [[0], [4], [5]],
            ],
        ],
        [
            [
                [[1], [1], [1]],
                [[1], [1], [11]],
                [[1], [1], [111]],
                [[1], [1], [1111]],
                [[1], [1], [11111]],
            ],
            [
                [[2], [2], [1]],
                [[2], [2], [22]],
                [[2], [2], [222]],
                [[2], [2], [2222]],
                [[2], [2], [22222]],
            ],
            [
                [[3], [3], [1]],
                [[3], [3], [33]],
                [[3], [3], [333]],
                [[3], [3], [3333]],
                [[3], [3], [33333]],
            ],
            [
                [[4], [4], [1]],
                [[4], [4], [44]],
                [[4], [4], [444]],
                [[4], [4], [4444]],
                [[4], [4], [44444]],
            ],
            [
                [[5], [5], [1]],
                [[5], [5], [55]],
                [[5], [5], [555]],
                [[5], [5], [5555]],
                [[5], [5], [55555]],
            ],
        ],
        [
            [
                [[0], [0], [2]],
                [[0], [0], [3]],
                [[0], [0], [4]],
                [[0], [0], [5]],
                [[0], [0], [6]],
            ],
            [
                [[0], [1], [2]],
                [[0], [1], [3]],
                [[0], [1], [4]],
                [[0], [1], [5]],
                [[0], [1], [6]],
            ],
            [
                [[0], [2], [2]],
                [[0], [2], [3]],
                [[0], [2], [4]],
                [[0], [2], [5]],
                [[0], [2], [6]],
            ],
            [
                [[0], [3], [2]],
                [[0], [3], [3]],
                [[0], [3], [4]],
                [[0], [3], [5]],
                [[0], [3], [6]],
            ],
            [
                [[0], [4], [2]],
                [[0], [4], [3]],
                [[0], [4], [4]],
                [[0], [4], [5]],
                [[0], [4], [6]],
            ],
        ],
    )

    memory = Memory(3, 3, 3, filter=SpectralFilter(obs_dim=3, action_dim=3, num_filters=2, spectral_history_length=2))
    obs_hist, action_hist, next_obs_hist = memory.from_trajectories(
        (
            jnp.array(trajectories[0]),
            jnp.array(trajectories[1]),
            jnp.array(trajectories[2]),
        ),
        remove_incomplete_histories=False,
    )

    for i in range(len(obs_hist)):
        print(obs_hist[i])
        print(action_hist[i])
        print(next_obs_hist[i])
        print("--------------------------------")

    # Expected output:
    """
    History 0: DO NOT START WITH FIRST OBSERVATION (array of 0s). Model should get the first observation and the first action,
    not have to guess randomly what they will be.
    obs_history: [[0, 0, 1], [0, 0, 0], [0, 0, 0]]
    action_history: [1, 0, 0]
    next_obs: [0, 0, 2]

    History 1:
    obs_history: [0, 1, 1]
    action_history: [1]
    next_obs_history: [0, 0, 3]
    
    """
