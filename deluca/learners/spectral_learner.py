from typing import List, Tuple, Callable
import jax.numpy as jnp
import jax.random
from jax import Array, jit
import optax
import scipy
from deluca.learners.linear_learner import LinearLearner
from deluca.utils import Task
from deluca import Env
from deluca.learners.core import Learner
from flax import nnx
from functools import partial
from jax.scipy.linalg import eigh as largest_eigh


class SpectralLearner(Learner):
    obs_history: Array
    action_history: Array
    current_history_index = 0

    def __init__(
        self,
        env: Env,
        history_length: int,
        rng: jax.Array,
        spectral_num_filters: int = 24,
        spectral_history_length: int = 30,
        learning_rate: float = 1e-3,
    ):
        """
        Args:
            env: Environment
            history_length: History length for linear learner
            rng: Random number generator
            spectral_num_filters: Number of eigenvectors for spectral filtering
            spectral_history_length: History length for spectral filtering
        """

        super().__init__(env, rng)

        self.spectral_num_filters = spectral_num_filters
        self.spectral_history_length = spectral_history_length
        self.lin_learner = LinearLearner(
            env,
            history_length=history_length,
            expanded_features_dim=spectral_num_filters * self.obs_dim,
            learning_rate=learning_rate,
        )
        self.obs_history = jnp.zeros((spectral_history_length, self.obs_dim))
        self.action_history = jnp.zeros((spectral_history_length, self.action_dim))

    def _apply_spectral_filtering(self, obs: Array) -> Array:
        """Apply spectral filtering to observations before passing to linear learner."""
        N, T, obs_dim = obs.shape

        # Initialize arrays for filtered observations
        filtered_obs = jnp.zeros((N, T, self.spectral_num_filters * obs_dim))
        filter_matrix = get_filters(self.spectral_history_length, self.spectral_num_filters)

        # For each trajectory
        with Task("SpectralLearner: Applying Spectral Filtering", N * T) as task:
            for n in range(N):
                # Apply spectral filtering at each timestep
                for t in range(T):
                    # NOTE: obs_history first entry is least recent (opposite of the linear learner)
                    obs_history = jnp.zeros((self.spectral_history_length, obs_dim))
                    if t < self.spectral_history_length:
                        obs_history = obs_history.at[:t, :].set(obs[n, :t, :])
                    else:
                        obs_history = obs_history.at[
                            : self.spectral_history_length, :
                        ].set(obs[n, t - self.spectral_history_length : t, :])

                   

                    # Get spectral features for current timestep
                    spectral_features = jnp.einsum("mh,md->hd", filter_matrix, obs_history)
                    # Reshape from (h, d, 1) to (h*d,)
                    filtered_obs = filtered_obs.at[n, t, :].set(
                        spectral_features.reshape(-1)
                    )
                    task.update()

        return filtered_obs

    def learn(
        self, trajectories
    ) -> Tuple[List[float], List[Tuple[float, List[float]]]]:
        """Learn using spectral filtering followed by linear learning."""
        obs, actions, next_obs = trajectories

        with Task("SpectralLearner: Learning", 2) as task:
            filtered_obs = self._apply_spectral_filtering(obs)
            task.update()

            return self.lin_learner.learn((filtered_obs, actions, next_obs))

    def test(self, trajectories) -> Tuple[float, List[float]]:
        """Test using spectral filtering followed by linear testing."""
        obs, actions, next_obs = trajectories

        filtered_obs = self._apply_spectral_filtering(obs)

        return self.lin_learner.test((filtered_obs, actions, next_obs))

    def predict(self, obs: Array, action: Array) -> List[float]:
        """Predict using spectral filtering followed by linear prediction."""
        # Update spectral history
        if self.current_history_index < self.spectral_history_length:
            self.obs_history = self.obs_history.at[self.current_history_index, :].set(obs)
            self.action_history = self.action_history.at[self.current_history_index, :].set(action)
            self.current_history_index += 1
        else:
            # roll backwards to free up last index
            self.obs_history = jnp.roll(self.obs_history, shift=-1, axis=0)
            self.action_history = jnp.roll(self.action_history, shift=-1, axis=0)
            self.obs_history = self.obs_history.at[-1, :].set(obs)
            self.action_history = self.action_history.at[-1, :].set(action)
            self.current_history_index = 0

        spectral_obs = self._apply_spectral_filtering(jnp.array([self.obs_history]))[0]

        m = self.lin_learner.history_length

        # Use the linear learner's predict_with_history method ()
        # Must reverse the history because the linear learner expects the most recent history to be at the end
        return self.lin_learner.predict_with_history(
            obs_history=spectral_obs[-m:][::-1],
            action_history=self.action_history[-m:][::-1],  # Take last m timesteps
        )

    def reset_env(self):
        """Reset the spectral history."""
        self.obs_history = jnp.zeros(
            (self.spectral_history_length, self.obs_dim)
        )
        self.action_history = jnp.zeros((self.spectral_history_length, self.action_dim))
        self.current_history_index = 0
        self.lin_learner.reset_env()

def get_filters(spectral_history_length: int, num_filters: int = 24):
    num_filters = min(num_filters, spectral_history_length)
    i = jnp.arange(1, spectral_history_length + 1)
    i_plus_j = i[None] + i[:, None]
    Z = 2 / (i_plus_j**3 - i_plus_j)
    Z = jnp.float64(Z)
    evals, evecs = largest_eigh(Z)
    return evecs[::-1, -num_filters:] * (evals[-num_filters:] ** 0.25)