from deluca.agents._grc import GRC
from deluca.agents._random import SimpleRandom
from deluca.agents._zero import Zero
from deluca.envs.brax._pendulum2d import Pendulum2D
from deluca.filters.spectral import SpectralFilter
from deluca.learners.ff_learner import (
    FFLearner,
    FFLearnerSettings,
    DefaultSettings as FFLearnerDefaultSettings,
)
from deluca.learners.core import Learner
from deluca.learners.n_linear_learner import (
    LinearLearner,
    LinearLearnerSettings,
    DefaultSettings as LinearDefaultSettings,
)
from deluca.learners.util import generate_simple_trajectories, generate_trajectories
from deluca.learners.memory import Memory
from deluca.envs._lds import LDS, SinusDisturbance, ZeroDisturbance
import jax
import numpy as np

import flax.nnx as nnx

import matplotlib
import matplotlib.pyplot as plt


def pred_vs_true_plot(learner: Learner, histories, ax):
    """
    Plot the predicted vs true next observations as a scatter plot on the given axis.
    """
    obs, actions, next_obs = histories

    # Get predictions from the learner
    pred = learner.predict(obs, actions)

    assert (
        pred.shape[-1] == next_obs.shape[-1]
    ), "Predictions and next observations must have the same number of dimensions"

    true = next_obs.reshape(-1, next_obs.shape[-1])
    pred = pred.reshape(-1, pred.shape[-1])

    hfi = learner.memory.history_length
    colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]
    alphas = np.linspace(0.2, 0.85, true.shape[0] - hfi).tolist()

    for i in range(true.shape[-1]):
        ax.scatter(
            true[:hfi, i],
            pred[:hfi, i],
            alpha=0.7,
            label=None,
            s=25,
            color=colors[i],
            marker="x",
        )
        ax.scatter(
            true[hfi:, i],
            pred[hfi:, i],
            alpha=alphas,
            label=f"y_{i}",
            s=20,
            color=colors[i],
            marker="o",
        )

    # Add diagonal line for perfect predictions
    min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        alpha=0.5,
        label="Perfect prediction",
    )

    ax.set_xlabel("True next observations")
    ax.set_ylabel("Predicted next observations")
    ax.set_title(f"Predicted vs True: {learner.name}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

def plot_losses(learner: Learner, train_losses, test_losses, batches_per_test, ax):
    """Plot losses on the given axis."""

    train_losses = train_losses.reshape(-1)
    test_losses = test_losses.reshape(-1)

    ax.plot(train_losses, label="Train")
    ax.plot(np.arange(0, len(test_losses)) * batches_per_test, test_losses, label="Test")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Losses for {learner.name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

d_obs = 2
d_action = 3
d_hidden = 10
disturbance = SinusDisturbance()
disturbance.init(d_hidden)

rng = jax.random.key(5)

# Set up environment and agent
rng, env_key, agent_key = jax.random.split(rng, 3)
env = LDS(env_key, d_action, d_hidden, d_obs, disturbance=disturbance)
agent = SimpleRandom(env.action_size, rng=agent_key) # GRC(env.A, env.B, env.C, rng=agent_key)

# Give agent a chance to learn
rng, env_key, agent_key = jax.random.split(rng, 3)
t, state, obs = env.reset(env_key)

for i in range(100):  # TODO: This seems to do nothing -- agent problems
    action = agent(obs, agent_key)
    t, state, obs = env(t, state, action, env_key)

# Generate trajectories
rng, histories_key = jax.random.split(rng)
N = 500
T = 250
obs, actions, next_obs = generate_trajectories(
    env, agent, N, T, rng=histories_key
)  # generate_simple_trajectories(N, T, rng=histories_key)

# TODO: generate trajectories corrupts the agent since it is stateful and not callable in jitted functions
# Agents must be updated.

# Split histories into train and test
train_amt = N - 50

train_obs = obs[:train_amt]
train_actions = actions[:train_amt]
train_next_obs = next_obs[:train_amt]

test_obs = obs[train_amt:]
test_actions = actions[train_amt:]
test_next_obs = next_obs[train_amt:]

spectral_filter = SpectralFilter(
    obs_dim=env.observation_size,
    action_dim=env.action_size,
    num_filters=24,
    spectral_history_length=100,
)

# Set up learner
settings = FFLearnerDefaultSettings
memory = Memory(30, env.observation_size, env.action_size)

train_histories = memory.from_trajectories((train_obs, train_actions, train_next_obs))
test_histories = memory.from_trajectories((test_obs, test_actions, test_next_obs))

rng, learner_key = jax.random.split(rng)
ff_learner = FFLearner(memory, settings, rng)
ff_train_losses, ff_test_losses = ff_learner.train(train_histories, learner_key)

settings = LinearDefaultSettings
linear_learner = LinearLearner(memory, settings, rng)
linear_train_losses, linear_test_losses = linear_learner.train(train_histories, learner_key)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

if memory.filter is not None:
    fig.suptitle(f"With filter: {memory.filter.__class__.__name__}")
else:
    fig.suptitle("No Filter")

plot_losses(ff_learner, ff_train_losses, ff_test_losses, settings.batches_per_test, axes[0, 0])
plot_losses(linear_learner, linear_train_losses, linear_test_losses, settings.batches_per_test, axes[0, 1])

pred_vs_true_plot(ff_learner, test_histories, axes[1, 0])
pred_vs_true_plot(linear_learner, test_histories, axes[1, 1])

plt.tight_layout()
plt.show()
