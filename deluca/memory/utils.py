import jax
from jax import Array
from typing import Tuple
from deluca.core import Env
from deluca.agents.new.core import Agent
from .memory import MemorySettings, Memory

def generate_histories(env: Env, agent: Agent, memory_settings: MemorySettings, N: int, T: int, rng: Array, remove_incomplete_histories: bool = True) -> Tuple[Array, Array, Array]:
    """
    Generate histories from N trajectories of length T.

    Args:
        env: Environment to generate trajectories in.
        agent: Agent to generate actions.
        memory_settings: Settings for the memory (ie. history length, filter)
        N: Number of trajectories to generate.
        T: Length of each trajectory.
        rng: JAX PRNG key.

    Returns:
        obs_hist: Array of shape (N*T', history_length, obs_dim)
        action_hist: Array of shape (N*T', history_length, action_dim)
        next_obs_hist: Array of shape (N*T', obs_dim)
    """

    history_len = memory_settings.history_length

    @jax.jit
    def _generate_trajectory(rng: Array) -> Tuple[Array, Array, Array]:

        def step_fn(memory: Memory, rng: Array):
            agent_rng, env_rng = jax.random.split(rng)
            action = agent(memory.get_history(), agent_rng)
            memory = memory.resolve(action)

            # Represents complete history at time t
            obs_history, action_history = memory.get_history()

            t, state, _ = memory.env_state
            new_t, new_state, new_obs = env(t, state, action, env_rng)
            memory = memory.prime(new_obs) # Prime for step t+1. At last step new_obs is T+1 and thus not included in the histories.
            memory.env_state = (new_t, new_state, new_obs)

            new_obs_squeezed = jax.numpy.squeeze(new_obs, -1) # Remove singleton dimension

            return memory, (obs_history, action_history, new_obs_squeezed)

        # Initialise memory with env.reset
        rng, env_rng = jax.random.split(rng)
        init_carry = Memory(memory_settings).reset_env(env, env_rng)
        _, (obs_history, action_history, next_obs_history) = jax.lax.scan(step_fn, init_carry, jax.random.split(rng, T))

        if remove_incomplete_histories:
            obs_history = obs_history[history_len - 1 :]
            action_history = action_history[history_len - 1 :]
            next_obs_history = next_obs_history[history_len - 1 :]

        return obs_history, action_history, next_obs_history

    obs_hist, action_hist, next_obs_hist = jax.vmap(_generate_trajectory)(
        jax.random.split(rng, N)
    )

    # (N, T', H, d) → (N*T', H, d)
    obs_hist = obs_hist.reshape(-1, *obs_hist.shape[-2:])
    action_hist = action_hist.reshape(-1, *action_hist.shape[-2:])
    next_obs_hist = next_obs_hist.reshape(-1, next_obs_hist.shape[-1])

    return obs_hist, action_hist, next_obs_hist