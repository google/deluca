# Learned Environments

from deluca.envs._brax import BraxEnv
from deluca.envs.classic._mountain_car import MountainCar
from deluca.learners.nn_learner import NNLearner
from deluca.envs import Pendulum2D
import jax
import jax.numpy as jnp
from deluca.agents import SimpleRandom
import matplotlib.pyplot as plt

from brax.envs import inverted_double_pendulum

from deluca.utils.printing import Task

# Test the NNLearner

seed = 2


def print_test_losses(axes, test_losses, name):
    loss_x, loss_y, loss_z = [], [], []

    # Convert test_losses = [(loss, loss_by_coord), (loss, loss_by_coord), ...]
    # to loss_x = [loss_by_coord[0], loss_by_coord[0], ...]
    # loss_y = [loss_by_coord[1], loss_by_coord[1], ...]
    # loss_z = [loss_by_coord[2], loss_by_coord[2], ...]
    for loss, loss_by_coord in test_losses:
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


def compare_learned_env_with_true_env(env, learner, name, axes, num_epochs=1000):
    global seed

    lenv = learner.get_learned_env()

    t_state, t_obs = env.init(jax.random.key(seed))
    seed += 1

    learner.reset_env()

    # Speed up with jax.jit and jax.lax.scan
    @jax.jit
    def get_rollout(init_state, init_obs, key):
        def step_fn(carry, _):
            t_state, t_obs, key = carry

            # Generate random action
            key, action_key = jax.random.split(key)
            # Assuming action space is a scalar between -2.0 and 2.0 for Pendulum
            # This is a common action space for this environment.
            action = jax.random.uniform(action_key, shape=(1,), minval=-2.0, maxval=2.0)

            # Get learned and true next observation
            l_obs = learner.predict(t_obs, action)
            new_t_state, new_t_obs = env(t_state, action)

            # jax.debug.print("@ On action {} predicted {} and true {} (diff {})", action[0], l_obs, new_t_obs, l_obs - new_t_obs)

            return (new_t_state, new_t_obs, key), (l_obs, new_t_obs)

        init_carry = (init_state, init_obs, key)
        _, (l_obses, t_obses) = jax.lax.scan(
            step_fn, init_carry, None, length=num_epochs
        )
        return l_obses, t_obses

    # Get the rollout
    l_obses, t_obses = get_rollout(t_state, t_obs, jax.random.key(seed))
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

with Task("Testing NNLearner", 4) as task:
    env = BraxEnv.from_env(inverted_double_pendulum.InvertedDoublePendulum()) #MountainCar().create()
    agent = SimpleRandom(env.action_size, jax.random.key(18))

    learner = NNLearner(env)
    learner.normalizer_functions = NNLearner.default_normalizers()
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

    learner = NNLearner(env)
    learner.normalizer_functions = NNLearner.default_normalizers()
    train_losses, test_losses = learner.learn((trajectories[0][:100], trajectories[1][:100], trajectories[2][:100]))

    axes[1].plot(train_losses)
    axes[1].set_title("Losses for 100x100 trajectories")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")

    print_test_losses(axes[7], test_losses, "100x100")

    compare_learned_env_with_true_env(env, learner, "100x100", axes[4])

    task.update()

    learner = NNLearner(env)
    learner.normalizer_functions = NNLearner.default_normalizers()
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
