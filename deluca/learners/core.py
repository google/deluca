from typing import Callable, Dict, List, Tuple, Any

import flax.struct
import jax
import chex

from deluca.agents._random import SimpleRandom
from deluca.core import Env
from deluca.learners.memory import MemorySettings
from deluca.normalizers.core import Normalizers, WithoutNormalization
from deluca.utils.printing import Task
from abc import abstractmethod
import jax.numpy as jnp
from jax import Array
import flax.nnx as nnx
import optax

Histories = Tuple[Array, Array, Array]


@flax.struct.dataclass
class LearnerSettings:
    learn_on_incomplete_histories: bool

    batch_size: int
    batches_per_test: int
    num_epochs: int
    holdout_ratio: float

    optimizer_fn: optax.GradientTransformation
    loss_fn: Callable[[Array, Array], Array | chex.Array]
    train_kwargs: Dict[str, Any] | None


DefaultSettings = LearnerSettings(
    learn_on_incomplete_histories=False,
    holdout_ratio=0.8,
    batch_size=32,
    batches_per_test=10,
    num_epochs=5,
    optimizer_fn=optax.adam(learning_rate=1e-3),
    loss_fn=lambda pred, next_obs: optax.squared_error(pred, next_obs),
    train_kwargs=None,
)


class LearnerModel(nnx.Module):
    @abstractmethod
    def __call__(self, obs: Array, action: Array, rng: Array) -> Array:
        """Predict the next observation given the history of most recent observations and actions."""

    @abstractmethod
    def _train_call(self, obs: Array, action: Array, rng: Array) -> Array:
        """Separate function if needed for training."""


class Learner:
    name: str

    model: LearnerModel

    obs_dim_in: int
    action_dim_in: int
    obs_dim_out: int
    action_dim_out: int
    history_length: int

    normalizers: Normalizers
    optimizer: nnx.Optimizer
    loss_fn: Callable[
        [Array, Array], Array | chex.Array
    ]  # Optax functions return chex.Array which works with jax.Array, but Python typing doesn't realize it lol

    def __init__(
        self,
        memory_settings: MemorySettings,
        settings: LearnerSettings,
        rng: Array,
        normalizers: Normalizers | None = None,
    ):
        self.obs_dim_in = memory_settings.obs_dim_in
        self.action_dim_in = memory_settings.action_dim_in
        self.obs_dim_out = memory_settings.obs_dim_out
        self.action_dim_out = memory_settings.action_dim_out
        self.history_length = memory_settings.history_length

        self.settings = settings

        self.normalizers = normalizers or WithoutNormalization()
        self.optimizer = nnx.Optimizer(
            self.model,
            self.settings.optimizer_fn,
        )

    def train(self, histories: Histories, rng: Array) -> Tuple[Array, Array]:
        """Train the learned environment from the histories. Histories are a tuple of (obs, action, next_obs)
        where obs is of shape (H, history_length, obs_dim) and action is of shape (H, history_length, action_dim)
        and next_obs is of shape (H, obs_dim)

        Returns:
            train_losses: List of losses for each epoch
            test_losses: List of (mean_loss, loss_by_coord) for each epoch
        """
        assert (
            len(histories) == 3
        ), "Histories must be a tuple of (obs, actions, next_obs)"

        obs, actions, next_obs = histories

        # For training, we need to compute the normalization of the observations and actions (e.g. mean and std)
        # These values will be stored and used for denormalization during prediction.
        self.normalizers.compute_normalization(obs, actions)

        histories = self._preprocess_histories(histories)

        # Splitting
        held_index = int(self.settings.holdout_ratio * len(obs))

        train_obs = obs[:held_index]
        train_actions = actions[:held_index]
        train_next_obs = next_obs[:held_index]

        test_obs = obs[held_index:]
        test_actions = actions[held_index:]
        test_next_obs = next_obs[held_index:]

        # Batching
        # Add a batch dimension to the histories (N*T, history_length, obs_dim) -> (N*T/batch_size, batch_size, history_length, obs_dim)
        num_histories = train_obs.shape[0]
        padding = self.settings.batch_size - num_histories % self.settings.batch_size
        num_batches = (num_histories + padding) // self.settings.batch_size
        if padding > 0:
            train_obs = jnp.concatenate(
                [train_obs, jnp.zeros((padding, obs.shape[1], obs.shape[2]))]
            )
            train_actions = jnp.concatenate(
                [
                    train_actions,
                    jnp.zeros((padding, actions.shape[1], actions.shape[2])),
                ]
            )
            train_next_obs = jnp.concatenate(
                [train_next_obs, jnp.zeros((padding, next_obs.shape[1]))]
            )

        train_obs = train_obs.reshape(
            num_batches, self.settings.batch_size, -1, train_obs.shape[-1]
        )
        train_actions = train_actions.reshape(
            num_batches, self.settings.batch_size, -1, train_actions.shape[-1]
        )
        train_next_obs = train_next_obs.reshape(
            num_batches, self.settings.batch_size, train_next_obs.shape[-1]
        )

        # Exposing settings for closure with jitted functions
        loss_fn = self.settings.loss_fn
        batches_per_test = self.settings.batches_per_test

        @nnx.jit
        def _test_step(model, test_obs, test_actions, test_next_obs, rng):
            test_pred = model(test_obs, test_actions, rng)
            test_loss = jnp.mean(loss_fn(test_pred, test_next_obs))
            return test_loss

        @nnx.jit
        def _train_batch(model, optimizer, batch, rng):
            obs_batch, actions_batch, next_obs_batch = batch

            def loss_step(model: LearnerModel):
                pred = model._train_call(obs_batch, actions_batch, rng)
                loss = loss_fn(pred, next_obs_batch)
                return jnp.mean(loss)

            train_loss, grads = nnx.value_and_grad(loss_step)(model)
            optimizer.update(grads)

            return train_loss

        train_losses = jnp.empty((self.settings.num_epochs, num_batches))
        test_losses = jnp.empty(
            (self.settings.num_epochs, num_batches // batches_per_test)
        )

        with Task(
            f"{self.name} Training for {self.settings.num_epochs} epochs",
            self.settings.num_epochs,
        ) as e_task:
            for e in range(self.settings.num_epochs):

                # Shuffle the batches
                rng, shuffle_key = jax.random.split(rng)
                batch_indices = jax.random.permutation(shuffle_key, num_batches)
                train_obs = train_obs[batch_indices]
                train_actions = train_actions[batch_indices]
                train_next_obs = train_next_obs[batch_indices]

                with Task(
                    f"Training with {num_batches} batches", num_batches
                ) as b_task:
                    for b, train_key in enumerate(jax.random.split(rng, num_batches)):
                        train_loss = _train_batch(
                            self.model,
                            self.optimizer,
                            (train_obs[b], train_actions[b], train_next_obs[b]),
                            train_key,
                        )
                        train_losses = train_losses.at[e, b].set(train_loss)

                        b_task.update()

                        if b % batches_per_test == 0:
                            test_loss = _test_step(
                                self.model,
                                test_obs[b],
                                test_actions[b],
                                test_next_obs[b],
                                train_key,
                            )
                            test_losses = test_losses.at[e, b // batches_per_test].set(
                                test_loss
                            )
                            b_task.update(
                                increment=0,
                                text=f"Batch {b} Losses: Train {train_loss:.5f} | Test {test_loss:.5f}",
                            )

                e_task.update(
                    text=f"Epoch {e}: Train Loss Avg {jnp.mean(train_losses[e]):.5f}"
                )

        return train_losses, test_losses

    def _preprocess_histories(self, histories: Histories) -> Histories:
        """
        Preprocess a collection of histories for training.
        """

        assert (
            len(histories) == 3
        ), "Histories must be a tuple of (obs, actions, next_obs)"

        obs, actions, next_obs = histories

        assert obs.ndim == actions.ndim == 3, "Histories must be 3D arrays"
        assert next_obs.ndim == 2, "next_obs must be a 2D array"
        assert (
            obs.shape[0] == actions.shape[0] == next_obs.shape[0]
        ), "obs, actions, and next_obs must have the same number of histories"
        assert (
            obs.shape[1] == actions.shape[1] == self.history_length
        ), "All histories must have length history_length"
        assert (
            obs.shape[2] == self.obs_dim_in
        ), "Observations must have obs_dim_in dimensions"
        assert (
            actions.shape[2] == self.action_dim_in
        ), "Actions must have action_dim_in dimensions"
        assert (
            next_obs.shape[1] == self.obs_dim_out
        ), "Next observations must have obs_dim_out dimensions"

        # Normalization
        obs = self.normalizers.normalize_obs(obs)
        actions = self.normalizers.normalize_action(actions)
        next_obs = self.normalizers.normalize_obs(next_obs)

        return obs, actions, next_obs

    def _preprocess_history(self, history: Tuple[Array, Array]) -> Tuple[Array, Array]:
        """
        Preprocess a single history for prediction.
        For prediction, we expect a history to be a tuple of (obs_history, action_history)
        """

        assert (
            len(history) == 2
        ), "History must be a tuple of (obs_history, action_history)"

        obs_history, action_history = history

        assert obs_history.ndim == action_history.ndim and obs_history.ndim in [
            2,
            3,
        ], "Histories must be 2D or 3D (batched) arrays"

        hist_index = 0 if obs_history.ndim == 2 else 1

        assert (
            obs_history.shape[hist_index]
            == action_history.shape[hist_index]
            == self.history_length
        ), "obs and actions must have length history_length"
        assert (
            obs_history.shape[hist_index + 1] == self.obs_dim_in
        ), "obs must have obs_dim_in dimensions"
        assert (
            action_history.shape[hist_index + 1] == self.action_dim_in
        ), "actions must have action_dim_in dimensions"

        obs_history = self.normalizers.normalize_obs(obs_history)
        action_history = self.normalizers.normalize_action(action_history)

        return obs_history, action_history

    def _postprocess_prediction(self, pred_obs: Array) -> Array:
        """
        Postprocess the prediction (e.g. denormalize and add back singleton dimension)
        """

        pred_obs = self.normalizers.denormalize_obs(pred_obs)

        return jnp.expand_dims(pred_obs, -1)

    def predict(self, obs_history: Array, action_history: Array, rng: Array) -> Array:
        """Predict the next observation given the history of most recent observations and actions.
        Note: if histories are longer than history_length, the oldest observations and actions are ignored.
        """

        obs_history, action_history = self._preprocess_history(
            (obs_history, action_history)
        )

        pred_obs = self.model(obs_history, action_history, rng)

        return self._postprocess_prediction(pred_obs)

    @abstractmethod
    def reset_env(self) -> None:
        """Reset the environment to the provided state."""

    def get_learned_env(self, initial_obs: Array):
        """
        Returns a learned environment that can be used to simulate the system.
        The learned environment is a wrapper around the predictor, and conforms to the Deluca Env interface.

        Args:
            initial_obs: The initial observation of the environment.

        Returns:
            LearnedEnv: The learned environment.
        """
        return LearnedEnv(
            initial_obs, self.action_dim_out, self.predict, self.reset_env
        )


class LearnedEnv(Env):
    def __init__(
        self,
        init_obs: Array,
        action_dim: int,
        predict: Callable[[Array, Array, Array], Array],
        reset_env: Callable[[], None],
    ):
        self.predict = predict
        self.init_obs = init_obs
        self.action_dim = action_dim
        self.reset_env = reset_env

    def init(self):
        return self.init_obs

    def __call__(self, t, obs, action, rng):
        """
        Args:
            t: This parameter is included to conform to the Deluca Env interface. It is not used.
            obs: The current observation of the environment. Note: normal environments take the entire state, but we only need the observation.
            action: The current action of the environment.
            rng: The random key for the predictor.

        Returns:
            t: The next time step.
            new_state: Since a learned environment trains only with observations, it has no conception of the latent state. Instead, we return the observation as the state.
            new_obs: The next observation of the environment.
        """
        new_obs = self.predict(obs, action, rng)
        return t + 1, new_obs, new_obs

    def reset(self, rng):
        self.reset_env()
        return (
            0,
            Array([]),
            self.predict(self.init_obs, jnp.zeros((self.action_dim,)), rng),
        )
