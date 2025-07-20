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


class SpectralLearner(Learner):
    spectral_obs_history: Array
    action_history: Array

    def __init__(
        self,
        env: Env,
        m: int,
        rng: jax.Array,
        spectral_h: int = 24,
        spectral_m: int = 30,
    ):
        """
        Args:
            env: Environment
            m: History length for linear learner
            rng: Random number generator
            spectral_h: Number of eigenvectors for spectral filtering
            spectral_m: History length for spectral filtering
        """

        super().__init__(env, rng)

        self.spectral_h = spectral_h
        self.spectral_m = spectral_m

        self.sfc_features_fn = get_sfc_features(
            m=spectral_m, d=self.obs_dim, h=spectral_h
        )

        self.lin_learner = LinearLearner(
            env, m=m, expanded_features_dim=spectral_h * self.obs_dim
        )
        self.spectral_obs_history = jnp.zeros((spectral_m, self.obs_dim))
        self.action_history = jnp.zeros((spectral_m, self.action_dim))

    def _apply_spectral_filtering(self, obs: Array) -> Array:
        """Apply spectral filtering to observations before passing to linear learner."""
        N, T, obs_dim = obs.shape

        # Initialize arrays for filtered observations
        filtered_obs = jnp.zeros((N, T, self.spectral_h * obs_dim))

        # For each trajectory
        for n in range(N):
            # Create disturbance history (assuming observations are the disturbances)
            # Pad with zeros at the beginning for history
            dist_history = jnp.concatenate(
                [
                    jnp.zeros((self.spectral_m - 1, obs_dim, 1)),
                    obs[n, :, :, None],  # Add singleton dimension for (d, 1)
                ],
                axis=0,
            )

            # Apply spectral filtering at each timestep
            for t in range(T):
                # Get spectral features for current timestep
                spectral_features = self.sfc_features_fn(t, dist_history)
                # Reshape from (h, d, 1) to (h*d,)
                filtered_obs = filtered_obs.at[n, t, :].set(
                    spectral_features.reshape(-1)
                )

        return filtered_obs

    def learn(
        self, trajectories
    ) -> Tuple[List[float], List[Tuple[float, List[float]]]]:
        """Learn using spectral filtering followed by linear learning."""
        obs, actions, next_obs = trajectories

        filtered_obs = self._apply_spectral_filtering(obs)

        with Task("SpectralLearner: Learning", 1) as task:
            return self.lin_learner.learn((filtered_obs, actions, next_obs))

    def test(self, trajectories) -> Tuple[float, List[float]]:
        """Test using spectral filtering followed by linear testing."""
        obs, actions, next_obs = trajectories

        filtered_obs = self._apply_spectral_filtering(obs)

        return self.lin_learner.test((filtered_obs, actions, next_obs))

    def predict(self, obs: Array, action: Array) -> List[float]:
        """Predict using spectral filtering followed by linear prediction."""
        # Update spectral history
        self.spectral_obs_history = jnp.roll(self.spectral_obs_history, shift=1, axis=0)
        self.action_history = jnp.roll(self.action_history, shift=1, axis=0)
        self.spectral_obs_history = self.spectral_obs_history.at[0].set(obs)
        self.action_history = self.action_history.at[0].set(action)

        # Create spectral features for the linear learner's history
        # The linear learner expects obs_history of shape (m, obs_dim) where m is the history length
        # We need to create spectral features for each timestep in the linear learner's history

        # Get the linear learner's history length
        m = self.lin_learner.m

        # Create spectral features for each timestep in the linear learner's history
        spectral_obs_history = jnp.zeros((m, self.spectral_h * self.obs_dim))

        # For each timestep in the linear learner's history
        for i in range(m):
            # Get the spectral features for timestep i
            # We need to create a disturbance history that includes the current spectral history
            # and pad with zeros for the future timesteps
            dist_history = jnp.concatenate(
                [
                    self.spectral_obs_history[
                        :, :, None
                    ],  # Add singleton dimension: (spectral_m, obs_dim, 1)
                    jnp.zeros(
                        (
                            self.spectral_m - len(self.spectral_obs_history),
                            self.obs_dim,
                            1,
                        )
                    ),  # Pad with zeros
                ],
                axis=0,
            )

            # Get spectral features for this timestep
            spectral_features = self.sfc_features_fn(i, dist_history)

            # Reshape and store
            spectral_obs_history = spectral_obs_history.at[i, :].set(
                spectral_features.reshape(-1)
            )

        # Use the linear learner's predict_with_history method
        return self.lin_learner.predict_with_history(
            obs_history=spectral_obs_history,
            action_history=self.action_history[:m],  # Take first m timesteps
        )

    def reset_env(self):
        """Reset the spectral history."""
        self.spectral_obs_history = jnp.zeros((self.spectral_m, self.obs_dim))
        self.action_history = jnp.zeros((self.spectral_m, self.action_dim))
        self.lin_learner.reset_env()


def compute_filter_matrix(T: int, h: int = 24) -> jnp.ndarray:
    """Compute the spectral filter matrix.

    Args:
        T: Context length
        h: Number of eigenvectors to use

    Returns:
        Filter matrix of shape (h * T)
    """
    # Create the matrix with entries 2/ (i+j-1)^3 - (i+j-1)
    i = jnp.arange(1, T + 1)
    j = jnp.arange(1, T + 1)
    I, J = jnp.meshgrid(i, j)
    matrix = 2 / ((I + J - 1) ** 3 - (I + J - 1))
    matrix = scipy.linalg.hilbert(T)
    # Compute eigenvalues and eigenvectors
    matrix = jnp.float64(matrix)
    eigenvalues, eigenvectors = scipy.linalg.eigh(matrix)

    # Sort in descending order and take top h
    idx = jnp.argsort(eigenvalues)[::-1][:h]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Create filter matrix: eigenvalue^{1/4} * eigenvector
    filter_matrix = eigenvectors * (jnp.abs(eigenvalues) ** 0.25)[None, :]

    return filter_matrix


def get_sfc_features(
    m: int,
    d: int,
    h: int,
) -> Callable[[int, jnp.ndarray], jnp.ndarray]:
    """Get a function that computes SFC features from disturbance history.

    Args:
        m: History length
        d: State dimension
        h: Number of eigenvectors
        gamma: Decay rate

    Returns:
        A function that takes (offset, dist_history) and returns filtered features
    """
    # Compute filter matrix
    filter_matrix = compute_filter_matrix(m, h)

    @jit
    def _get_sfc_features(
        offset: int,
        dist_history: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute SFC features from a window of disturbance history.

        Args:
            offset: Current time offset
            dist_history: Full disturbance history of shape (m+hist, d, 1)
            filter_matrix: Pre-computed filter matrix of shape (m, h)

        Returns:
            Filtered features of shape (h, d, 1)
        """
        window = jax.lax.dynamic_slice(dist_history, (offset, 0, 0), (m, d, 1))
        return jnp.einsum("mh,md1->hd1", filter_matrix, window)

    return _get_sfc_features
