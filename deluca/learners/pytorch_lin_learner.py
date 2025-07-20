import torch
import torch.nn as nn
from typing import Callable, Sequence, Tuple, List

from deluca.core import Env
from deluca.learners.core import Learner
from deluca.utils.printing import Task


_device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"  # type: ignore


class _LinearEnvironmentPredictor(nn.Module):
    def __init__(self, h: int, obs_dim: int, action_dim: int):
        super().__init__()
        self.h = h
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # M_i and N_i are learned vectors for each history position
        self.M = nn.Parameter(torch.zeros((h, obs_dim), device=_device))
        self.N = nn.Parameter(torch.zeros((h, action_dim), device=_device))
        self.b = nn.Parameter(torch.zeros((obs_dim), device=_device))

    def forward(
        self, obs_history: torch.Tensor, action_history: torch.Tensor
    ) -> torch.Tensor:
        # Add batch dimension if not present (2D -> 3D)
        original_shape = obs_history.shape
        if len(obs_history.shape) == 2:
            obs_history = obs_history[None, :, :]
            action_history = action_history[None, :, :]

        # y_out = sum_i=0^h M_i^T * y_t-i + sum_i=0^h N_i^T * u_t-i + b
        obs_effect = torch.einsum("hi,bhi->bi", self.M, obs_history)
        action_effect = torch.einsum("hi,bhi->bi", self.N, action_history)

        y_out = obs_effect + action_effect + self.b

        # If input was 2D, squeeze the output back to 2D
        if len(original_shape) == 2:
            y_out = y_out.squeeze(0)

        return y_out


class Pytorch_LinearLearner(Learner):

    normalizer_functions: (
        Tuple[
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
            Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        ]
        | None
    ) = None
    num_histories_in_epoch: int = 150
    holdout_ratio: float = 0.8
    has_learned = False
    learn_on_incomplete_histories: bool = False

    def __init__(
        self, env: Env, learning_rate: float = 1e-3, momentum: float = 0.9, h: int = 30
    ):
        super().__init__(env)

        self.model = _LinearEnvironmentPredictor(
            h=h, obs_dim=self.obs_dim, action_dim=self.action_dim
        )
        self.h = h

        self.prediction_obs_history = torch.zeros(
            (self.h, self.obs_dim), device=_device
        )
        self.prediction_action_history = torch.zeros(
            (self.h, self.action_dim), device=_device
        )
        self.loss_fn = nn.MSELoss(reduction="none")
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=learning_rate, momentum=momentum
        )

    def _learn_step(
        self,
        obs_history: torch.Tensor,
        action_history: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> Tuple[float, torch.Tensor]:
        pred = self.model(obs_history, action_history)
        loss = self.loss_fn(pred, next_obs).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _test(
        self,
        obs_histories: torch.Tensor,
        action_histories: torch.Tensor,
        next_obs_histories: torch.Tensor,
    ) -> Tuple[float, List[float]]:
        pred = self.model(obs_histories, action_histories)
        loss = self.loss_fn(pred, next_obs_histories)
        return loss.mean().item(), loss.mean(dim=0).tolist()

    def learn(
        self, trajectories
    ) -> Tuple[List[float], List[Tuple[float, List[float]]]]:
        obs, actions, next_obs = trajectories

        assert (
            obs.shape[0] == actions.shape[0] == next_obs.shape[0]
        ), "Observations, actions, and next observations must have the same length"

        obs = torch.tensor(obs, device=_device)
        actions = torch.tensor(actions, device=_device)
        next_obs = torch.tensor(next_obs, device=_device)

        # Normalization
        if self.normalizer_functions is not None:
            normalize, _, compute_normalization = self.normalizer_functions
            self.obs_mean, self.obs_var = compute_normalization(obs)
            self.action_mean, self.action_var = compute_normalization(actions)
            obs = normalize(obs, self.obs_mean, self.obs_var)
            actions = normalize(actions, self.action_mean, self.action_var)
            next_obs = normalize(next_obs, self.obs_mean, self.obs_var)

        obs_histories, action_histories, next_obs_histories = self.prepare_histories(
            obs, actions, next_obs
        )

        # Shuffling
        perm = torch.randperm(obs_histories.shape[0])
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

        with Task("Pytorch_LinearLearner: Learning", len(train_obs_histories)) as task:
            for i in range(len(train_obs_histories)):
                train_loss = self._learn_step(
                    train_obs_histories[i],
                    train_action_histories[i],
                    train_next_obses[i],
                )
                train_losses.append(train_loss)
                task.update()

                if i % self.num_histories_in_epoch == 0 and len(test_obs_histories) > 0:
                    test_loss = self._test(
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

    def test(
        self, trajectories
    ):
        if not self.has_learned:
            raise ValueError("Pytorch_LinearLearner has not been trained yet")
        
        obs, actions, next_obs = trajectories

        obs = torch.tensor(obs, device=_device)
        actions = torch.tensor(actions, device=_device)
        next_obs = torch.tensor(next_obs, device=_device)

        obs_histories, action_histories, next_obs_histories = self.prepare_histories(
            obs, actions, next_obs
        )

        pred = self.model(obs_histories, action_histories)
        error = self.loss_fn(pred, next_obs_histories)
        loss_by_coord = error.mean(dim=0)
        return loss_by_coord.mean().item(), loss_by_coord.tolist()
    
    def predict(self, obs, action) -> List[float]:
        obs = torch.tensor(obs, device=_device)
        action = torch.tensor(action, device=_device)
        
        # Update history buffers (shift and add new obs/action)
        self.prediction_obs_history = torch.cat([self.prediction_obs_history[1:], obs.unsqueeze(0)], dim=0)
        self.prediction_action_history = torch.cat([self.prediction_action_history[1:], action.unsqueeze(0)], dim=0)
        
        # Pass history to model
        pred = self.model(self.prediction_obs_history, self.prediction_action_history)
        return pred.tolist()
    
    def reset_env(self):
        """Reset the prediction history buffers."""
        self.prediction_obs_history = torch.zeros(
            (self.h, self.obs_dim), device=_device
        )
        self.prediction_action_history = torch.zeros(
            (self.h, self.action_dim), device=_device
        )
        
    def prepare_histories(
        self, obs: torch.Tensor, actions: torch.Tensor, next_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare histories for training by creating sliding windows of observations and actions.
        """

        N, T, obs_dim = obs.shape
        _, _, action_dim = actions.shape

        # Create index arrays for gathering
        n_idx = torch.arange(N, device=obs.device)[:, None, None]  # (N, 1, 1)
        t_idx = torch.arange(T, device=obs.device)[None, :, None]  # (1, T, 1)
        h_idx = torch.arange(self.h, device=obs.device)[None, None, :]  # (1, 1, h)

        # Calculate source indices for history
        source_idx = t_idx - h_idx  # (1, T, h)

        # Pad the trajectories with zeros at the beginning
        padded_obs = torch.cat(
            [torch.zeros((N, self.h - 1, obs_dim), device=obs.device), obs], dim=1
        )  # (N, T+h-1, obs_dim)

        padded_actions = torch.cat(
            [torch.zeros((N, self.h - 1, action_dim), device=actions.device), actions],
            dim=1,
        )  # (N, T+h-1, action_dim)

        # Adjust indices for padding
        source_idx_adjusted = source_idx + (self.h - 1)  # (1, T, h)

        # Gather using advanced indexing
        obs_histories = padded_obs[n_idx, source_idx_adjusted]  # (N, T, h, obs_dim)
        action_histories = padded_actions[
            n_idx, source_idx_adjusted
        ]  # (N, T, h, action_dim)

        # Flatten
        obs_histories = obs_histories.reshape(N * T, self.h, obs_dim)
        action_histories = action_histories.reshape(N * T, self.h, action_dim)
        next_obs_histories = next_obs.reshape(N * T, obs_dim)

        if not self.learn_on_incomplete_histories:
            # Create mask for complete histories (timesteps >= h-1)
            # For each trajectory, the first h-1 timesteps have incomplete histories
            timestep_indices = torch.arange(obs.shape[1], device=obs.device).repeat(
                obs.shape[0]
            )  # Flattened timestep indices
            valid_mask = timestep_indices >= (self.h - 1)

            # Apply mask to filter out incomplete histories
            obs_histories = obs_histories[valid_mask]
            action_histories = action_histories[valid_mask]
            next_obs_histories = next_obs_histories[valid_mask]

        return obs_histories, action_histories, next_obs_histories
    
    @classmethod
    def default_normalizers(cls):
        return (
            cls._default_normalizer,
            cls._default_denormalizer,
            cls._default_compute_normalization,
        )


    @staticmethod
    def _default_normalizer(
        x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor
    ) -> torch.Tensor:
        return (x - mean) / (var + 1e-8)

    @staticmethod
    def _default_denormalizer(
        x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor
    ) -> torch.Tensor:
        return x * torch.sqrt(var) + mean

    @staticmethod
    def _default_compute_normalization(
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return x.mean(dim=0), x.var(dim=0)
