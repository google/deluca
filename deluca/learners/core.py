from typing import Callable, List, Tuple

from deluca.agents._random import SimpleRandom
from deluca.core import Env
from deluca.utils.printing import Task
from abc import abstractmethod
import jax
import jax.numpy as jnp


class Learner:
    env: Env

    def __init__(self, env: Env, rng=jax.random.key(0)):
        self.env = env
        self.rng, subkey = jax.random.split(rng)
        _, sample_obs = env.init(subkey)
        self.obs_dim = (
            sample_obs.shape[-1] if sample_obs.ndim > 1 else sample_obs.shape[0]
        )
        self.action_dim = jnp.zeros(env.action_size).shape[0]

        # Reset the environment to the same state
        self.env.init(subkey)

    def generate_trajectories(
        self, N, T, key=jax.random.key(0)
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Generate N trajectories of length T. Use random policy for now."""

        with Task("Generating trajectories", N) as task:

            def single_trajectory(trajectory_index, key):
                rand_key, env_key = jax.random.split(key)
                agent = SimpleRandom(
                    self.env.action_size, rand_key
                )  # TODO: agent should be passed in
                state, obs = self.env.init(env_key)
                jax.debug.callback(lambda: task.update())

                def step_fn(carry, _):
                    state, obs = carry
                    action = agent(obs)
                    state, new_obs = self.env(state, action[0])
                    return (state, new_obs), (obs, action[0], new_obs)

                init_carry = (state, obs)
                (final_state, final_obs), steps = jax.lax.scan(
                    step_fn, init_carry, None, length=T
                )

                return steps

            out = jax.vmap(single_trajectory)(jnp.arange(N), jax.random.split(key, N))

        return out

    @abstractmethod
    def learn(self, trajectories: Tuple[jax.Array, jax.Array, jax.Array]) -> Tuple[List[float], List[Tuple[float, List[float]]]]:
        """Learn the environment from the trajectories.

        Returns:
            train_losses: List of losses for each epoch
            test_losses: List of (mean_loss, loss_by_coord) for each epoch
        """

    # TODO fix return types depending on if we go with jax or torch
    @abstractmethod
    def predict(self, obs: jax.Array, action: jax.Array) -> List[float]: 
        """Predict the next obs from the current obs and action."""

    @abstractmethod
    def test(
        self, trajectories: Tuple[jax.Array, jax.Array, jax.Array]
    ) -> Tuple[float, List[float]]:
        """Test the learner on a batch of trajectories.

        Returns:
            mean_loss: Mean loss over all trajectories
            loss_by_coord: Loss by coordinate over all trajectories
        """

    @abstractmethod
    def reset_env(self) -> None:
        """Reset the environment to the provided state."""

    def get_learned_env(self):
        return LearnedEnv(
            self.env.init(jax.random.key(0))[1], self.predict, self.reset_env
        )


class LearnedEnv(Env):
    def __init__(
        self,
        init_obs: jax.Array,
        predict: Callable[[jax.Array, jax.Array], List[float]],
        reset_env: Callable[[], None],
    ):
        self.predict = predict
        self.init_obs = init_obs
        self.reset_env = reset_env

    def init(self):
        return self.init_obs

    def __call__(self, state, action):
        return self.predict(state, action)

    def reset(self):
        return self.reset_env()
