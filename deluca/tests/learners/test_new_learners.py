from deluca.agents.new.linear import LinearAgent, DefaultSettings as DefaultLinearAgentSettings
from deluca.envs._brax import BraxEnv
from deluca.agents._grc import GRC
from deluca.agents.new.random import SimpleRandom
from deluca.agents._zero import Zero
from deluca.envs.brax._pendulum2d import Pendulum2D
from deluca.filters.spectral import SpectralFilter
from deluca.learners.feedforward import (
    FFLearner,
    FFLearnerSettings,
    DefaultSettings as FFLearnerDefaultSettings,
)
from deluca.learners.core import Learner
from deluca.learners.linear import (
    LinearLearner,
    LinearLearnerSettings,
    DefaultSettings as LinearDefaultSettings,
)
from deluca.memory import Memory, MemorySettings
from deluca.envs._lds import LDS, SinusDisturbance, ZeroDisturbance
import jax
import jax.numpy as jnp
import numpy as np

import flax.nnx as nnx

import matplotlib
import matplotlib.pyplot as plt

from deluca.memory.utils import generate_histories
from deluca.utils.printing import progress


def pred_vs_true_plot(learner: Learner, history_length: int, histories, ax, rng):
    """
    Plot the predicted vs true next observations as a scatter plot on the given axis.
    """
    obs, actions, next_obs = histories

    # Get predictions from the learner
    pred = learner.predict(obs, actions, rng)

    pred = jnp.squeeze(pred, -1)

    assert (
        pred.shape[-1] == next_obs.shape[-1]
    ), "Predictions and next observations must have the same number of dimensions"

    true = next_obs.reshape(-1, next_obs.shape[-1])
    pred = pred.reshape(-1, pred.shape[-1])

    hfi = history_length
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
        "lime",
        "teal",
        "indigo",
        "violet",
        "coral",
        "maroon",
        "navy",
        "gold",
        "silver",
        "black",
        "white",
        "coral",
        "maroon",
        "navy",
    ]
    alphas = np.linspace(0.1, 0.85, true.shape[0] - hfi).tolist()

    for i in range(true.shape[-1]):
        ax.scatter(
            true[:hfi, i],
            pred[:hfi, i],
            alpha=0.8,
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

d_obs = 5
d_action = 3
d_hidden = 10
disturbance = SinusDisturbance()
disturbance.init(d_hidden)

rng = jax.random.key(5)

# Set up environment and agent
rng, env_key, agent_key = jax.random.split(rng, 3)
env = LDS(env_key, d_action, d_hidden, d_obs, disturbance=disturbance)
# env = BraxEnv(env_key, "inverted_double_pendulum")
# agent = GRC(env.A, env.B, env.C, rng=agent_key)

spectral_filter = SpectralFilter(
    obs_dim=env.observation_size,
    action_dim=env.action_size,
    num_filters=24,
    spectral_history_length=100,
)

N = 1000
TEST_AMT = 50
T = 100

rng, agent_rng = jax.random.split(rng)
memory_settings = MemorySettings.from_env(env, 30, spectral_filter)
agent = LinearAgent(memory_settings, DefaultLinearAgentSettings, rng=agent_rng)
# agent = SimpleRandom(memory_settings, DefaultLinearAgentSettings, agent_rng)

# Train agent slightly
# memory = Memory(memory_settings).reset_env(env, rng)

# rng, agent_rng = jax.random.split(rng)
# i = 0
# for irng in progress(jax.random.split(agent_rng, 2), "Training agent"):
#     loss, memory = agent.train_step(env, memory, irng)
#     i += 1
#     if i % 10 == 0:
#         print(f"Loss: {loss}")
#         print(f"Loss: {loss}")
        

print("Generating trajectories...")
rng, h1_rng, h2_rng = jax.random.split(rng, 3)
train_histories = generate_histories(env, agent, memory_settings, N - TEST_AMT, T, rng=h1_rng)
test_histories = generate_histories(env, agent, memory_settings, TEST_AMT, T, rng=h2_rng)
print("Done generating trajectories.")

# Set up learner
settings = FFLearnerDefaultSettings

rng, learner_key = jax.random.split(rng)
ff_learner = FFLearner(memory_settings, settings, rng)
ff_train_losses, ff_test_losses = ff_learner.train(train_histories, learner_key)

settings = LinearDefaultSettings
linear_learner = LinearLearner(memory_settings, settings, rng)
linear_train_losses, linear_test_losses = linear_learner.train(train_histories, learner_key)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

fig.suptitle(f"{env.__class__.__name__} {env.env.env.env.__class__.__name__ if hasattr(env, 'env') else ""} with {memory_settings.filter.__class__.__name__ if memory_settings.filter else "No Filter"} (d_hidden={d_hidden})") # type: ignore

plot_losses(ff_learner, ff_train_losses, ff_test_losses, settings.batches_per_test, axes[0, 0])
plot_losses(linear_learner, linear_train_losses, linear_test_losses, settings.batches_per_test, axes[0, 1])

pred_vs_true_plot(ff_learner, memory_settings.history_length, test_histories, axes[1, 0], rng)
pred_vs_true_plot(linear_learner, memory_settings.history_length, test_histories, axes[1, 1], rng)

plt.tight_layout()
plt.show()
