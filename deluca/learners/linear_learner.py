from typing import List, Tuple, Callable
import jax.numpy as jnp
import jax.random
from jax import Array
import optax
from deluca.utils import Task
from deluca import Env
from deluca.learners.core import Learner
from flax import nnx
from functools import partial


class _LinearEnvironmentPredictor(nnx.Module):
    """LinearEnvironmentPredictor.
    This learner predicts the next observation from a previous observations and actions.
    It uses a linear model to predict the next observation.
    """

    def __init__(self, history_length: int, obs_dim: int, action_dim: int, expanded_features_dim = 0):
        super().__init__()
        self.history_length = history_length
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        obs_dim_in = expanded_features_dim if expanded_features_dim > 0 else obs_dim

        self.M = nnx.Param(jnp.zeros((history_length, obs_dim, obs_dim_in)))
        self.N = nnx.Param(jnp.zeros((history_length, obs_dim, action_dim)))
        self.b = nnx.Param(jnp.zeros(obs_dim))

    def __call__(self, obs_history: Array, action_history: Array) -> Array:
        # y_t+1 = sum_i=0^h M_i * y_t-i + sum_i=0^h N_i * u_t-i + b

        original_shape = obs_history.shape
        if len(obs_history.shape) == 2:
            obs_history = obs_history[None, :, :]
            action_history = action_history[None, :, :]

        # y_out = (
        #     jnp.sum(self.M[None, :, :] * obs_history, axis=1)
        #     + jnp.sum(self.N[None, :, :] * action_history, axis=1)
        #     + self.b
        # )

        # y_out = (
        #     jnp.tensordot(self.M, obs_history, axes=[[0, 2], [0, 1]]) # type: ignore
        #     + jnp.tensordot(self.N, action_history, axes=[[0, 2], [0, 1]]) # type: ignore
        #     + self.b
        # )

        obs_contribution = jnp.einsum("ijk,bik->bj", self.M, obs_history)

        # N_i * u_{t-i} summed over i
        action_contribution = jnp.einsum("ijk,bik->bj", self.N, action_history)

        y_out = obs_contribution + action_contribution + self.b[jnp.newaxis, :]

        # If input was 2D, squeeze the output back to 2D
        if len(original_shape) == 2:
            y_out = y_out.squeeze(axis=0)

        return y_out
    
    def train_call(self, obs_history: Array, action_history: Array, beta):
        original_shape = obs_history.shape
        if len(obs_history.shape) == 2:
            obs_history = obs_history[None, :, :]
            action_history = action_history[None, :, :]

        rho = 2 / self.history_length
        
        # Apply beta and rho factors to the matrices
        M_weighted = beta * self.M * (1 - rho) ** jnp.arange(self.history_length)[:, None, None]
        N_weighted = beta * self.N * (1 - rho) ** jnp.arange(self.history_length)[:, None, None]

        # Use the same einsum approach as __call__
        obs_contribution = jnp.einsum("ijk,bik->bj", M_weighted, obs_history)
        action_contribution = jnp.einsum("ijk,bik->bj", N_weighted, action_history)
        
        y_out = obs_contribution + action_contribution + self.b[jnp.newaxis, :]

        # If input was 2D, squeeze the output back to 2D
        if len(original_shape) == 2:
            y_out = y_out.squeeze(axis=0)

        return y_out

def _prepare_histories_static(
    obs: Array, actions: Array, next_obs: Array, h: int
) -> Tuple[Array, Array, Array]:
    N, T, obs_dim = obs.shape
    _, _, action_dim = actions.shape
    _, _, next_obs_dim = next_obs.shape

    # Create index arrays for gathering
    n_idx = jnp.arange(N)[:, None, None]  # (N, 1, 1)
    t_idx = jnp.arange(T)[None, :, None]  # (1, T, 1)
    h_idx = jnp.arange(h)[None, None, :]  # (1, 1, h)

    # Calculate source indices for history
    source_idx = t_idx - h_idx  # (1, T, h)

    # Pad the trajectories with zeros at the beginning
    padded_obs = jnp.concatenate(
        [jnp.zeros((N, h - 1, obs_dim)), obs], axis=1
    )  # (N, T+h-1, obs_dim)

    padded_actions = jnp.concatenate(
        [jnp.zeros((N, h - 1, action_dim)), actions], axis=1
    )  # (N, T+h-1, action_dim)

    # Adjust indices for padding
    source_idx_adjusted = source_idx + (h - 1)  # (1, T, h)

    # Gather using advanced indexing
    obs_histories = padded_obs[n_idx, source_idx_adjusted]  # (N, T, h, obs_dim)
    action_histories = padded_actions[
        n_idx, source_idx_adjusted
    ]  # (N, T, h, action_dim)

    # Flatten
    obs_histories = obs_histories.reshape(N * T, h, obs_dim)
    action_histories = action_histories.reshape(N * T, h, action_dim)

    # Use the original next_obs_dim, since obs_dim may be expanded for spectral filtering
    next_obs_histories = next_obs.reshape(N * T, next_obs_dim)

    return obs_histories, action_histories, next_obs_histories


num_zero = 0
num_high = 0


class LinearLearner(Learner):
    num_histories_in_epoch: int = 150
    holdout_ratio: float = 0.8
    batch_size: int = 32
    has_learned = False
    normalizer_functions: Tuple[
        Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
        Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
        Callable[[jax.Array], Tuple[jax.Array, jax.Array]],
    ] = (lambda x, mean, var: x, lambda x, mean, var: x, lambda x: (x, x))

    prediction_obs_history: jax.Array
    prediction_action_history: jax.Array

    learn_on_incomplete_histories: bool = False
    """If True, the learner will learn on histories that are shorter than h."""
    
    matrix_norm_target: float = 10.0
    """Target norm for M and N matrices after each training step."""

    def __init__(
        self,
        env: Env,
        learning_rate: float = 1e-3,
        momentum: float = 0.9,
        history_length: int = 30,
        expanded_features_dim: int = 0,
        matrix_norm_target: float = 10.0,
    ):
        super().__init__(env)

        self.expanded_obs_dim = expanded_features_dim + self.obs_dim

        # Allow for overriding the obs and action dimensions for spectral filtering

        self.model = _LinearEnvironmentPredictor(
            history_length=history_length, obs_dim=self.obs_dim, action_dim=self.action_dim, expanded_features_dim=expanded_features_dim
        )
        self.optimizer = nnx.Optimizer(
            self.model, optax.chain(
                optax.sgd(learning_rate, momentum=momentum),
                optax.clip_by_global_norm(10.0)
            )
        )

        self.history_length = history_length
        self.matrix_norm_target = matrix_norm_target

        self.prediction_obs_history = jnp.zeros((self.history_length, self.expanded_obs_dim))
        self.prediction_action_history = jnp.zeros((self.history_length, self.action_dim))

        # Cache for JIT compiled functions with different shapes
        self._prepare_histories_cache = {}

    def _get_prepare_histories_fn(self, shape_key):
        """Get or create a JIT compiled prepare_histories function for given shape."""
        if shape_key not in self._prepare_histories_cache:
            prepare_fn = partial(_prepare_histories_static, h=self.history_length)
            self._prepare_histories_cache[shape_key] = jax.jit(prepare_fn)
        return self._prepare_histories_cache[shape_key]

    def prepare_histories(
        self, obs: Array, actions: Array, next_obs: Array
    ) -> Tuple[Array, Array, Array]:
        """
        Wrapper method that handles task updates and calls the JIT compiled function.
        """
        shape_key = (obs.shape, actions.shape, next_obs.shape)
        prepare_fn = self._get_prepare_histories_fn(shape_key)
        obs_histories, action_histories, next_obs_histories = prepare_fn(
            obs, actions, next_obs
        )

        if not self.learn_on_incomplete_histories:
            # Create mask for complete histories (timesteps >= h-1)
            # For each trajectory, the first h-1 timesteps have incomplete histories
            timestep_indices = jnp.tile(
                jnp.arange(obs.shape[1]), obs.shape[0]
            )  # Flattened timestep indices
            valid_mask = timestep_indices >= (self.history_length - 1)

            # Apply mask to filter out incomplete histories
            obs_histories = obs_histories[valid_mask]
            action_histories = action_histories[valid_mask]
            next_obs_histories = next_obs_histories[valid_mask]

        return obs_histories, action_histories, next_obs_histories

    def learn(
        self, trajectories
    ) -> Tuple[List[float], List[Tuple[float, List[float]]]]:

        # Define learning functions outside of the main learn method
        @nnx.jit
        def _test(
            model, obs_histories, action_histories, next_obs_histories
        ) -> Tuple[float, Array]:
            pred = model(obs_histories, action_histories)
            error = optax.squared_error(pred, next_obs_histories)
            loss_by_coord = error.mean(axis=0)
            return loss_by_coord.mean(), loss_by_coord # type: ignore

        @nnx.jit
        def _learn_step(
            model: _LinearEnvironmentPredictor, optimizer, obs_history, action_history, next_obs
        ) -> Tuple[float, nnx.Module, nnx.Optimizer]:
            def loss_fn(model):
                pred = model.train_call(obs_history, action_history, 10.0)

                loss = optax.squared_error(pred, next_obs).mean()
                return loss

            loss, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(grads)
            return loss, model, optimizer

        obs, actions, next_obs = trajectories

        assert (
            obs.shape[0] == actions.shape[0] == next_obs.shape[0]
        ), "Observations, actions, and next observations must have the same length"

        # Normalization
        normalize, _, compute_normalization = self.normalizer_functions

        self.obs_mean, self.obs_var = compute_normalization(obs)
        self.action_mean, self.action_var = compute_normalization(actions)

        obs = normalize(obs, self.obs_mean, self.obs_var)
        actions = normalize(actions, self.action_mean, self.action_var)
        next_obs = normalize(next_obs, self.obs_mean, self.obs_var)

        obs_histories, action_histories, next_obs_histories = self.prepare_histories(
            obs, actions, next_obs
        )

        # Shuffle trajectories
        self.rng, subkey = jax.random.split(self.rng)
        perm = jax.random.permutation(subkey, len(obs_histories))
        obs_histories = obs_histories[perm]
        action_histories = action_histories[perm]
        next_obs_histories = next_obs_histories[perm]

        # Split train/test
        holdout_index = int(len(obs_histories) * self.holdout_ratio)
        train_obs_histories = obs_histories[:holdout_index]
        train_action_histories = action_histories[:holdout_index]
        train_next_obses = next_obs_histories[:holdout_index]

        test_obs_histories = obs_histories[holdout_index:]
        test_action_histories = action_histories[holdout_index:]
        test_next_obses = next_obs_histories[holdout_index:]

        train_losses = []
        test_losses = []

        # Training loop (not JIT compiled due to task updates)
        with Task("LinearLearner: Learning", len(train_obs_histories)) as task:
            for i in range(len(train_obs_histories)):
                # Learn step
                train_loss, self.model, self.optimizer = _learn_step(
                    self.model,
                    self.optimizer,
                    train_obs_histories[i],
                    train_action_histories[i],
                    train_next_obses[i],
                )

                train_losses.append(train_loss)

                # Test periodically
                if i % self.num_histories_in_epoch == 0 and len(test_obs_histories) > 0:
                    test_loss = _test(
                        self.model,
                        test_obs_histories,
                        test_action_histories,
                        test_next_obses,
                    )
                    test_losses.append(test_loss)
                    task.update(
                        increment=0,
                        text=f"Epoch {i//self.num_histories_in_epoch} Test loss: {test_loss[0]:.4f}",
                    )

                task.update()

        self.has_learned = True

        return train_losses, test_losses

    def test(self, trajectories) -> Tuple[float, List[float]]:
        if not self.has_learned:
            raise ValueError("LinearLearner has not been trained yet")

        @nnx.jit
        def _test(
            model, obs_histories, action_histories, next_obs_histories
        ) -> Tuple[float, List[float]]:
            pred = model(obs_histories, action_histories)
            error = optax.squared_error(pred, next_obs_histories)
            loss_by_coord = error.mean(axis=0)
            return loss_by_coord.mean(), loss_by_coord, (pred, next_obs_histories) # type: ignore

        obs, actions, next_obs = trajectories

        obs = jnp.array(obs)
        actions = jnp.array(actions)
        next_obs = jnp.array(next_obs)

        normalize, _, _ = self.normalizer_functions

        obs = normalize(obs, self.obs_mean, self.obs_var)
        actions = normalize(actions, self.action_mean, self.action_var)
        next_obs = normalize(next_obs, self.obs_mean, self.obs_var)

        obs_histories, action_histories, next_obs_histories = self.prepare_histories(
            obs, actions, next_obs
        )

        with Task("LinearLearner: Testing", 1) as task:
            loss = _test(
                self.model, obs_histories, action_histories, next_obs_histories
            )
            task.update()

        return loss

    def reset_env(self):
        """
        Reset the prediction history to the provided initial obs and action.
        """
        self.prediction_obs_history = jnp.zeros((self.history_length, self.obs_dim))
        self.prediction_action_history = jnp.zeros((self.history_length, self.action_dim))

    def predict(self, obs: Array, action: Array) -> List[float]:
        """
        Predict the next observation from the obs and action. Provided obs and action
        will be stored in the history.
        """
        normalize, denormalize, _ = self.normalizer_functions
        obs_norm = normalize(obs, self.obs_mean, self.obs_var)
        action_norm = normalize(action, self.action_mean, self.action_var)

        self.prediction_obs_history = jnp.roll(
            self.prediction_obs_history, shift=1, axis=0
        )
        self.prediction_action_history = jnp.roll(
            self.prediction_action_history, shift=1, axis=0
        )
        self.prediction_obs_history = self.prediction_obs_history.at[0].set(obs_norm)
        self.prediction_action_history = self.prediction_action_history.at[0].set(
            action_norm
        )

        pred_norm = self.model(
            self.prediction_obs_history, self.prediction_action_history
        )
        prediction = denormalize(pred_norm, self.obs_mean, self.obs_var)

        return prediction  # type: ignore

    def predict_with_history(
        self, obs_history: Array, action_history: Array
    ) -> List[float]:
        """
        Predict the next observation from the obs and action. Provided obs and action
        will be stored in the history.
        """
        # Normalize inputs
        normalize, denormalize, _ = self.normalizer_functions

        obs_history_norm = normalize(obs_history, self.obs_mean, self.obs_var)
        action_history_norm = normalize(
            action_history, self.action_mean, self.action_var
        )

        # Predict and denormalize
        pred_norm = self.model(obs_history_norm, action_history_norm)
        prediction = denormalize(pred_norm, self.obs_mean, self.obs_var)

        return prediction  # type: ignore

    @staticmethod
    def default_normalizers():
        return (
            _default_normalizer,
            _default_denormalizer,
            _default_compute_normalization,
        )


@nnx.jit
def _default_normalizer(x, mean, var):
    return nnx.standardize(x, mean=mean, variance=var + 1e-8)


@nnx.jit
def _default_denormalizer(x, mean, var):
    return x * jnp.sqrt(var) + mean


@nnx.jit
def _default_compute_normalization(x):
    return jnp.mean(x, axis=(0, 1)), jnp.var(x, axis=(0, 1))


if __name__ == "__main__":
    trajectories = (
        [
            [
                [0, 0, 1],
                [0, 0, 2],
                [0, 0, 3],
                [0, 0, 4],
                [0, 0, 5],
            ],
            [
                [0, 1, 1],
                [0, 1, 2],
                [0, 1, 3],
                [0, 1, 4],
                [0, 1, 5],
            ],
            [
                [0, 2, 1],
                [0, 2, 2],
                [0, 2, 3],
                [0, 2, 4],
                [0, 2, 5],
            ],
            [
                [0, 3, 1],
                [0, 3, 2],
                [0, 3, 3],
                [0, 3, 4],
                [0, 3, 5],
            ],
            [
                [0, 4, 1],
                [0, 4, 2],
                [0, 4, 3],
                [0, 4, 4],
                [0, 4, 5],
            ],
        ],
        [
            [
                [1],
                [11],
                [111],
                [1111],
                [11111],
            ],
            [
                [2],
                [22],
                [222],
                [2222],
                [22222],
            ],
            [
                [3],
                [33],
                [333],
                [3333],
                [33333],
            ],
            [
                [4],
                [44],
                [444],
                [4444],
                [44444],
            ],
            [
                [5],
                [55],
                [555],
                [5555],
                [55555],
            ],
        ],
        [
            [
                [0, 0, 2],
                [0, 0, 3],
                [0, 0, 4],
                [0, 0, 5],
                [0, 0, 6],
            ],
            [
                [0, 1, 2],
                [0, 1, 3],
                [0, 1, 4],
                [0, 1, 5],
                [0, 1, 6],
            ],
            [
                [0, 2, 2],
                [0, 2, 3],
                [0, 2, 4],
                [0, 2, 5],
                [0, 2, 6],
            ],
            [
                [0, 3, 2],
                [0, 3, 3],
                [0, 3, 4],
                [0, 3, 5],
                [0, 3, 6],
            ],
            [
                [0, 4, 2],
                [0, 4, 3],
                [0, 4, 4],
                [0, 4, 5],
                [0, 4, 6],
            ],
        ],
    )
    obs_hist, action_hist, next_obs_hist = _prepare_histories_static(
        jnp.array(trajectories[0]),
        jnp.array(trajectories[1]),
        jnp.array(trajectories[2]),
        3,
    )

    # Create mask for complete histories (timesteps >= h-1)
    # For each trajectory, the first h-1 timesteps have incomplete histories
    timestep_indices = jnp.tile(jnp.arange(5), 5)  # Flattened timestep indices
    valid_mask = timestep_indices >= (3 - 1)
    # print(timestep_indices)
    # print(valid_mask)

    # Apply mask to filter out incomplete histories
    # obs_hist = obs_hist[valid_mask]
    # action_hist = action_hist[valid_mask]
    # next_obs_hist = next_obs_hist[valid_mask]

    # for i in range(len(obs_hist)):
    #     print(obs_hist[i])
    #     print(action_hist[i])
    #     print(next_obs_hist[i])
    #     print("--------------------------------")

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
