# Learned Environments

import numpy as np
import jax
from deluca.envs import Pendulum2D, MountainCar, Pendulum
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from deluca.learners.pytorch_lin_learner import Pytorch_LinearLearner
from deluca.utils.printing import Task

# Test the NNLearner

seed = 2


def print_test_losses(axes, test_losses, name):
    loss_x, loss_y, loss_z = [], [], []
    for loss, loss_by_coord in test_losses:
        # Handle both torch tensors and regular numbers
        if hasattr(loss_by_coord[0], 'item'):
            loss_x.append(loss_by_coord[0].item())
            loss_y.append(loss_by_coord[1].item())
            loss_z.append(loss_by_coord[2].item())
        else:
            loss_x.append(loss_by_coord[0])
            loss_y.append(loss_by_coord[1])
            loss_z.append(loss_by_coord[2])

    axes.plot(loss_x, label="cos(theta)")
    axes.plot(loss_y, label="sin(theta)")
    axes.plot(loss_z, label="theta_dot")
    axes.set_title(name)
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Loss")
    axes.legend()


def compare_learned_env_with_true_env(env, learner, name, axes, num_epochs=200):
    global seed
    
    lenv = learner.get_learned_env()

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    t_state, t_obs = env.init(jax.random.key(seed))
    learner.reset_env()
    
    seed += 1

    
    # Convert to PyTorch tensors and use torch operations
    def get_rollout(init_state, init_obs, seed):
        torch.manual_seed(seed)
        key = jax.random.key(seed)
        
        l_obses = []
        t_obses = []
        t_state = init_state
        t_obs = init_obs
        
        with Task("Comparing Learned and True Environments", num_epochs) as task:
            for i in range(num_epochs):
                action_key, key = jax.random.split(key)
                action = jax.random.uniform(action_key, shape=(1,), minval=-2.0, maxval=2.0)

                l_obs = learner.predict(t_obs, action)
                
                new_t_state, new_t_obs = env(t_state, action)

                l_obses.append(l_obs)
                t_obses.append(new_t_obs)
                task.update()

                t_state = new_t_state
                t_obs = new_t_obs


        return l_obses, t_obses

    # Get the rollout
    l_obses, t_obses = get_rollout(t_state, t_obs, seed)
    seed += 1

    # Convert to numpy for plotting
    l_obses_np = np.array(l_obses)
    t_obses_np = np.array(t_obses)

    # Plot scatter of predicted vs true for all observation dimensions
    axes.scatter(t_obses_np[:, 0], l_obses_np[:, 0], alpha=0.3, label="cos(theta)")
    axes.scatter(t_obses_np[:, 1], l_obses_np[:, 1], alpha=0.3, label="sin(theta)")
    axes.scatter(t_obses_np[:, 2], l_obses_np[:, 2], alpha=0.3, label="theta_dot")

    # Add y=x line for reference
    lims = [
        np.min(np.stack([t_obses, l_obses])),
        np.max(np.stack([t_obses, l_obses])),
    ]
    axes.plot(lims, lims, "r--", alpha=0.75, zorder=0, label="y=x")

    axes.set_aspect("equal", adjustable="box")
    axes.set_title(f"{name} trained: pred vs true")
    axes.set_xlabel("True Observation")
    axes.set_ylabel("Predicted Observation")
    axes.legend()


fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

with Task("Testing LinearLearner", 4) as task:
    env = Pendulum2D().create()

    learner = Pytorch_LinearLearner(env)
    #learner.normalizer_functions = Pytorch_LinearLearner.default_normalizers()
    trajectories = learner.generate_trajectories(2500, 100)

    task.update()

    train_losses, test_losses = learner.learn((trajectories[0][:10], trajectories[1][:10], trajectories[2][:10]))

    # Plot losses for 10x10 trajectories
    axes[0].plot(train_losses)
    axes[0].set_title("Losses for 10x100 trajectories")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    print_test_losses(axes[6], test_losses, "10x100")

    compare_learned_env_with_true_env(env, learner, "10x100", axes[3])

    task.update()

    learner = Pytorch_LinearLearner(env)
    #learner.normalizer_functions = Pytorch_LinearLearner.default_normalizers()
    train_losses, test_losses = learner.learn((trajectories[0][:101], trajectories[1][:101], trajectories[2][:101]))

    axes[1].plot(train_losses)
    axes[1].set_title("Losses for 100x100 trajectories")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")

    print_test_losses(axes[7], test_losses, "100x100")

    compare_learned_env_with_true_env(env, learner, "100x100", axes[4])

    task.update()

    learner = Pytorch_LinearLearner(env)
    #learner.normalizer_functions = Pytorch_LinearLearner.default_normalizers()
    train_losses, test_losses = learner.learn(trajectories)

    axes[2].plot(train_losses)
    axes[2].set_title("Losses for 2500x100 trajectories")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")

    print_test_losses(axes[8], test_losses, "2500x100")

    compare_learned_env_with_true_env(env, learner, "2500x100", axes[5])

    task.update()

plt.tight_layout()
plt.show()
