# Learned Environments

from deluca.learners import LinearLearner
from deluca.envs import Pendulum2D, MountainCar, Pendulum
import jax
import jax.numpy as jnp
from deluca.agents import SimpleRandom
import matplotlib.pyplot as plt

from deluca.learners.spectral_learner import SpectralLearner
from deluca.utils.printing import Task

# Test the NNLearner

seed = 2


def print_test_losses(axes, test_losses, name):
    loss_x, loss_y, loss_z = [], [], []
    for loss, loss_by_coord in test_losses:
        loss_x.append(jnp.mean(loss_by_coord[0]))
        loss_y.append(jnp.mean(loss_by_coord[1]))
        loss_z.append(jnp.mean(loss_by_coord[2]))

    axes.plot(loss_x, label="cos(theta)")
    axes.plot(loss_y, label="sin(theta)")
    axes.plot(loss_z, label="theta_dot")
    axes.set_title(name)
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Loss")
    axes.legend()


def compare_learned_env_with_true_env(env, agent, learner, name, axes, num_epochs=200):
    global seed


    lenv = learner.get_learned_env()

    t_state, t_obs = env.init(jax.random.key(seed))
    learner.reset_env()
    
    seed += 1

    l_obses, t_obses = [], []
    key = jax.random.key(seed)

    with Task(f"{name} trained: pred vs true", num_epochs) as task:
        for i in range(num_epochs):
            # Generate random action
            key, action_key = jax.random.split(key)
            # Assuming action space is a scalar between -2.0 and 2.0 for Pendulum
            # This is a common action space for this environment.
            action = agent(t_obs)
            action = jnp.squeeze(action, axis=-1)

            # Get learned and true next observation
            l_obs = lenv(t_obs, action)
            new_t_state, new_t_obs = env(t_state, action)

            l_obses.append(l_obs)
            t_obses.append(new_t_obs)

            task.update()

            t_state, t_obs = new_t_state, new_t_obs

    l_obses = jnp.array(l_obses)
    t_obses = jnp.array(t_obses)
    seed += 1

    # Plot scatter of predicted vs true for all observation dimensions
    axes.scatter(t_obses[:, 0], l_obses[:, 0], alpha=0.3, label="cos(theta)")
    axes.scatter(t_obses[:, 1], l_obses[:, 1], alpha=0.3, label="sin(theta)")
    axes.scatter(t_obses[:, 2], l_obses[:, 2], alpha=0.3, label="theta_dot")

    # Add y=x line for reference
    lims = [
        jnp.min(jnp.array([t_obses, l_obses])),
        jnp.max(jnp.array([t_obses, l_obses])),
    ]
    axes.plot(lims, lims, "r--", alpha=0.75, zorder=0, label="y=x")

    axes.set_aspect("equal", adjustable="box")
    axes.set_title(f"{name} trained: pred vs true")
    axes.set_xlabel("True Observation")
    axes.set_ylabel("Predicted Observation")
    axes.legend()


fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

with Task("Testing Spectral Learner", 4) as task:
    env = Pendulum2D().create()
    agent = SimpleRandom(env.action_size, jax.random.key(18))

    learner = SpectralLearner(env, m=30, rng=jax.random.key(29))
    # learner.normalizer_functions = LinearLearner.default_normalizers()
    trajectories = learner.generate_trajectories(1250, 100)

    task.update()

    train_losses, test_losses = learner.learn((trajectories[0][:10], trajectories[1][:10], trajectories[2][:10]))

    # Plot losses for 10x10 trajectories
    axes[0].plot(train_losses)
    axes[0].set_title("Losses for 10x100 trajectories (Spectral)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    print_test_losses(axes[6], test_losses, "10x100")

    compare_learned_env_with_true_env(env, agent, learner, "10x100", axes[3])

    task.update()

    learner = LinearLearner(env)
    # learner.normalizer_functions = LinearLearner.default_normalizers()
    train_losses, test_losses = learner.learn((trajectories[0][:101], trajectories[1][:101], trajectories[2][:101]))

    axes[1].plot(train_losses)
    axes[1].set_title("Losses for 100x100 trajectories (Spectral)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")

    print_test_losses(axes[7], test_losses, "100x100")

    compare_learned_env_with_true_env(env, agent, learner, "100x100", axes[4])

    task.update()

    learner = LinearLearner(env)
    # learner.normalizer_functions = LinearLearner.default_normalizers()
    train_losses, test_losses = learner.learn(trajectories)

    axes[2].plot(train_losses)
    axes[2].set_title("Losses for 2500x100 trajectories (Spectral)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")

    print_test_losses(axes[8], test_losses, "2500x100")

    compare_learned_env_with_true_env(env, agent, learner, "2500x100", axes[5])

    task.update()

plt.tight_layout()
plt.show()
