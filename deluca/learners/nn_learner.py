from deluca.learners.core import Learner
from deluca.core import Env
from deluca.utils.printing import Task

import jax
import jax.numpy as jnp
from flax.training import train_state
import flax.linen as nn
from typing import Callable, List, Tuple
import optax


class _EnvironmentPredictor(nn.Module):
    """Neural network with two hidden layers to predict next observation from current observation and action."""

    hidden_size: int = 64
    obs_dim: int = 3

    @nn.compact
    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        # Concatenate observation and action along the last dimension
        x = jnp.concatenate([obs, action], axis=-1)

        # First hidden layer
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)

        # Second hidden layer
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)

        # Output layer - predict next observation
        next_obs = nn.Dense(self.obs_dim)(x)

        return next_obs


class NNLearner(Learner):
    num_trajectories_in_epoch: int = 10
    holdout_ratio: float = 0.8
    batch_size: int = 32  # Add batch size parameter
    has_learned = False
    normalizer_functions: Tuple[
        Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
        Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
        Callable[[jax.Array], Tuple[jax.Array, jax.Array]],
    ] = (lambda x, mean, var: x, lambda x, mean, var: x, lambda x: (x, x))

    mean: jax.Array
    var: jax.Array

    def __init__(self, env: Env, hidden_size: int = 64, learning_rate: float = 1e-3, rng=jax.random.key(0)):
        superkey, rng = jax.random.split(rng)
        
        super().__init__(env, superkey)
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Get sample observation to determine dimensions
        self.model = _EnvironmentPredictor(hidden_size=hidden_size, obs_dim=self.obs_dim)


        # Initialize model parameters
        self.params = self.model.init(rng, jnp.zeros((1, self.obs_dim)), jnp.zeros((1, self.action_dim)))["params"]

        self.tx = optax.sgd(learning_rate, momentum=0.9)
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=self.params,
            tx=self.tx,
        )

        self.loss_fn = lambda pred, next_obs: optax.squared_error(pred, next_obs).mean()
        self.loss_fn_by_coord = lambda pred, next_obs: optax.squared_error(
            pred, next_obs
        )

    def learn_step_batched(self, state, obs_batch, action_batch, next_obs_batch):
        """Optimized batch learning step."""

        def loss_fn(params):
            pred = state.apply_fn({"params": params}, obs_batch, action_batch)
            return self.loss_fn(pred, next_obs_batch)

        loss = loss_fn(state.params)
        grads = jax.grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def learn(
        self,
        trajectories: Tuple[jax.Array, jax.Array, jax.Array],
    ) -> Tuple[List[float], List[Tuple[float, List[float]]]]:
        """
        Learn from trajectories with batched training.

        Args:
            trajectories: List of trajectories, each trajectory is a list of (obs, action, next_obs) tuples

        Returns:
            List of losses
        """
        train_losses = []
        test_losses = []

        normalize, _, compute_normalization = self.normalizer_functions

        with Task("NNLearner: Preparing to learn", 4) as task:

            task.update(increment=0, text="Flattening")

            # Flatten all training data into single arrays
            all_obs, all_actions, all_next_obs = trajectories

            all_obs = all_obs.reshape(-1, all_obs.shape[-1])
            all_actions = all_actions.reshape(-1, all_actions.shape[-1])
            all_next_obs = all_next_obs.reshape(-1, all_next_obs.shape[-1])

            task.update(increment=0, text="Shuffling")

            heldout_index = int(self.holdout_ratio * len(all_obs))

            indices = jax.random.permutation(jax.random.key(0), len(all_obs))

            all_obs = all_obs[indices]
            all_actions = all_actions[indices]
            all_next_obs = all_next_obs[indices]

            task.update(text="Normalizing")

            self.obs_mean, self.obs_var = compute_normalization(all_obs)
            self.action_mean, self.action_var = compute_normalization(all_actions)

            all_obs = normalize(all_obs, self.obs_mean, self.obs_var)
            all_actions = normalize(all_actions, self.action_mean, self.action_var)
            all_next_obs = normalize(all_next_obs, self.obs_mean, self.obs_var)

            task.update(text="Batching")

            train_obs = all_obs[:heldout_index]
            train_actions = all_actions[:heldout_index]
            train_next_obs = all_next_obs[:heldout_index]

            test_obs = all_obs[heldout_index:]
            test_actions = all_actions[heldout_index:]
            test_next_obs = all_next_obs[heldout_index:]

            num_samples = len(train_obs)
            num_batches = (num_samples + self.batch_size - 1) // self.batch_size
            num_epochs = num_batches // self.num_trajectories_in_epoch

            task.update()

        with Task("NNLearner: Learning", num_batches) as task:
            for epoch in range(0, num_epochs):
                epoch_losses = []

                # Process batches within this epoch
                for batch_idx in range(
                    epoch * self.num_trajectories_in_epoch,
                    min(
                        epoch * self.num_trajectories_in_epoch
                        + self.num_trajectories_in_epoch,
                        num_batches,
                    ),
                ):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(start_idx + self.batch_size, num_samples)

                    obs_batch = train_obs[start_idx:end_idx]
                    action_batch = train_actions[start_idx:end_idx]
                    next_obs_batch = train_next_obs[start_idx:end_idx]

                    self.state, loss = self.learn_step_batched(
                        self.state, obs_batch, action_batch, next_obs_batch
                    )
                    epoch_losses.append(loss)
                    task.update()

                # Compute average loss for this epoch
                avg_train_loss = jnp.mean(jnp.array(epoch_losses))
                train_losses.append(avg_train_loss)

                test_pred = self.state.apply_fn(
                    {"params": self.state.params}, test_obs, test_actions
                )
                test_loss = self.loss_fn_by_coord(test_pred, test_next_obs)
                test_loss_mean = jnp.mean(test_loss)
                test_losses.append((test_loss_mean, test_loss.mean(axis=0)))

                task.update(
                    increment=0,
                    text=f"Epoch {epoch} Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss_mean:.4f}"
                )

        self.has_learned = True

        return train_losses, test_losses

    def test(self, trajectories: Tuple[jax.Array, jax.Array, jax.Array]) -> Tuple[float, List[float]]:
        if not self.has_learned:
            raise ValueError("NNLearner has not been trained yet")
        
        obs, actions, next_obs = trajectories

        normalize, _, _ = self.normalizer_functions

        obs = normalize(obs, self.obs_mean, self.obs_var)
        actions = normalize(actions, self.action_mean, self.action_var)
        next_obs = normalize(next_obs, self.obs_mean, self.obs_var)

        pred = self.state.apply_fn({"params": self.state.params}, obs, actions)
        error = self.loss_fn_by_coord(pred, next_obs)
        loss_by_coord = error.mean(axis=(0, 1)) 

        return loss_by_coord.mean(), loss_by_coord, (pred, next_obs) # type: ignore

    def predict(self, obs: jax.Array, action: jax.Array) -> List[float]:
        """
        Predict the next observation given current observation and action.

        Args:
            obs: Current observation
            action: Current action

        Returns:
            Predicted next observation
        """
        if not self.has_learned:
            raise ValueError("NNLearner has not been trained yet")

        normalize, denormalize, _ = self.normalizer_functions

        # Ensure inputs have batch dimension
        if obs.ndim == 1:
            obs_batch = obs[None, :]
            action_batch = action[None, :]
        else:
            obs_batch = obs
            action_batch = action

        obs_batch = normalize(obs_batch, self.obs_mean, self.obs_var)
        action_batch = normalize(action_batch, self.action_mean, self.action_var)

        result = self.state.apply_fn({"params": self.state.params}, obs_batch, action_batch)  # type: ignore
        result = denormalize(result, self.obs_mean, self.obs_var)

        # Remove batch dimension if input was single observation
        if obs.ndim == 1:
            return result[0] # type: ignore
        return result # type: ignore


    def reset_env(self):
        """Reset the environment to the provided state."""
        pass

    @staticmethod
    def default_normalizers():
        """
        Returns (normalize, denormalize, compute_normalization)
        normalize: Normalize the data to have mean 0 and variance 1.
        denormalize: Denormalize the data to the original scale.
        compute_normalization: Compute the mean and variance of the data.
        """
        return (
            _default_normalizer,
            _default_denormalizer,
            _default_compute_normalization,
        )

@jax.jit
def _default_normalizer(x, mean, var):
    return nn.standardize(x, mean=mean, variance=var)

@jax.jit
def _default_denormalizer(x, mean, var):
    std = jnp.sqrt(var)
    return x * std + mean

@jax.jit
def _default_compute_normalization(x):
    return (
        jnp.mean(x, axis=0),
        jnp.var(x, axis=0),
    )
