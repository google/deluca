from typing import Callable, List, Tuple

from deluca.agents._random import SimpleRandom
from deluca.core import Env
from deluca.utils.printing import Task
from abc import abstractmethod
import jax
import jax.numpy as jnp


class Learner:
    env: Env

    def __init__(self, env: Env):
        self.env = env

    def generate_trajectories(self, N, T, key=jax.random.key(0)) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Generate N trajectories of length T. Use random policy for now."""
        
        with Task("Generating trajectories", N) as task:
            def single_trajectory(trajectory_index, key):
                rand_key, env_key = jax.random.split(key)
                agent = SimpleRandom(self.env.action_size, rand_key) # TODO: agent should be passed in
                state, obs = self.env.init(env_key)
                
                def step_fn(carry, _):
                    state, obs = carry
                    action = agent(obs)
                    state, new_obs = self.env(state, action[0])
                    return (state, new_obs), (obs, action[0], new_obs)
                
                init_carry = (state, obs)
                (final_state, final_obs), steps = jax.lax.scan(step_fn, init_carry, None, length=T)
                
                jax.debug.callback(lambda: task.update())
                
                return steps

            out = jax.vmap(single_trajectory)(jnp.arange(N), jax.random.split(key, N))

        return out


    @abstractmethod
    def learn(self, trajectories) -> Tuple[List[float], List[Tuple[float, jax.Array]]]:
        """Learn the environment from the trajectories.
        
        Returns:
            train_losses: List of losses for each epoch
            test_losses: List of (mean_loss, loss_by_coord) for each epoch
        """

    @abstractmethod
    def predict(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Predict the next obs from the current obs and action."""

    @abstractmethod
    def _test_unpacked(self, obses, actions, next_obses) -> float:
        """Test the learner on a batch of observations, actions, and next observations."""

    def unpack(self, trajectories) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Convert trajectories[i][j] = (obs, action, next_obs) to three arrays of shape (N, T, dim)"""
        obs, actions, next_obs = [], [], []
        for traj in trajectories:
            for obs, action, next_obs in traj:
                obs.append(obs)
                actions.append(action)
                next_obs.append(next_obs)

        return jnp.array(obs), jnp.array(actions), jnp.array(next_obs)
    
    def test(self, trajectories) -> float:
        """Test the learner on a set of trajectories."""
        obses, actions, next_obses = self.unpack(trajectories)
        return self._test_unpacked(obses, actions, next_obses)

    def get_learned_env(self):
        return LearnedEnv(self.env.init(jax.random.key(0))[1], self.predict)

class LearnedEnv(Env):
    def __init__(self, init_obs: jax.Array, predict: Callable[[jax.Array, jax.Array], jax.Array]):
        self.predict = predict
        self.init_obs = init_obs

    def init(self):
        return self.init_obs

    def __call__(self, state, action):
        return self.predict(state, action)