from typing import Any
from deluca.agents.new.feedforward import FFAgent, DefaultSettings as FFAgentDefaultSettings
from deluca.agents.new.linear import LinearAgent, DefaultSettings as LinearAgentDefaultSettings
from deluca.envs._brax import BraxEnv
from deluca.envs.classic._mountain_car import MountainCar
from deluca.envs.classic._pendulum import Pendulum
from deluca.filters.spectral import SpectralFilter
from deluca.learners.memory import Memory, MemorySettings
from deluca.envs._lds import LDS, SinusDisturbance, ZeroDisturbance
from deluca.learners.util import generate_trajectories
from deluca.normalizers.core import DefaultNormalizers
from deluca.agents._random import SimpleRandom
import jax
import jax.numpy as jnp
from jax import Array
import numpy as np
import flax.nnx as nnx

import matplotlib.pyplot as plt

from deluca.utils.printing import Task, progress

d_obs = 2
d_action = 3
d_hidden = 10
disturbance = SinusDisturbance()
disturbance.init(d_hidden)

rng = jax.random.key(3)

# Set up environment and agent
rng, env_key, agent_key, random_key = jax.random.split(rng, 4)
# env = LDS(env_key, d_action, d_hidden, d_obs, disturbance=disturbance, x0=jax.random.normal(env_key, (d_hidden, 1)))
# env = MountainCar(env_key)
# env = Pendulum(env_key)
# env = PlaygroundEnv(env_key, "Go1Getup")
env = BraxEnv(env_key, "inverted_double_pendulum")
filter = SpectralFilter(obs_dim=env.observation_size, action_dim=env.action_size, num_filters=24, spectral_history_length=100)

"""
>>> Initializing a Normalizer
memory = Memory(1, env.observation_size, env.action_size, filter=filter)
normalizer = DefaultNormalizers()
random_agent = SimpleRandom(env.action_size, random_key)

obs_history, action_history, _ = memory.from_trajectories(generate_trajectories(env, random_agent, 100, 100, rng=env_key))
normalizer.compute_normalization(obs_history, action_history)

print(f"Normalization Primed with: ")
print("--------------------------------")
print("Variable \t \t | mean \t \t | std")
print("--------------------------------")
print(f"Obs \t \t | {obs_history.mean()} \t | {obs_history.std()}")
print(f"Action \t \t | {action_history.mean()} \t | {action_history.std()}")
print("--------------------------------")
"""

# def loss_fn(action: Array, next_state: Any, next_obs: Array) -> Array:
#     """Inverted Double Pendulum Loss

#     Observation Space
#     - 0: Position of Cart on Linear Surface
#     - 1: Vertical Angle of Pole on the Cart (goal: keep straight up)
#     - 2: Linear Velocity of the Cart (goal: minimize)
#     - 3: Angular Velocity of the pole on the cart (goal: minimize)"""

#     # penalties = jnp.array([0, 10, 0.1, 0.7])
#     # return jnp.sum(jnp.square(next_obs) * penalties) + next_state.done * 50

#     # Test loss: encourage cart to move to position 0.4
#     return jnp.square(next_obs[0] - 0.4)

def loss_fn(action: Array, next_state: Any, next_obs: Array) -> Array:
    """
    Computes a complex differentiable loss for Brax's inverted_double_pendulum env.
    Encourages upright stability, stillness, center balance, and smoothness.
    
    Args:
        state: brax.physics.base.State (after action applied)
        action: jnp.ndarray of shape [batch_size, action_dim]
        
    Returns:
        scalar loss (batch-mean)
    """

    qp = next_state.qp

    # Unpack key components
    pos = qp.pos        # [n_bodies, 3]
    rot = qp.rot        # [n_bodies, 4] quaternion
    vel = qp.vel        # [n_bodies, 3]
    ang = qp.ang        # [n_bodies, 3]

    # Body indices (based on Brax default)
    cart_idx = 0
    pole1_idx = 1
    pole2_idx = 2

    # 1. Pole orientation: get up direction using quaternion
    def get_up_vector(q):
        # rotate [0, 1, 0] by quaternion to get "up" vector
        # since Brax default pole points in Y direction
        # use formula for rotating vector by quaternion
        qx, qy, qz, qw = q
        # Up vector rotated = q * [0,1,0,0] * q_conj
        up = jnp.array([
            2 * (qx * qy + qw * qz),
            1 - 2 * (qx**2 + qz**2),
            2 * (qy * qz - qw * qx)
        ])
        return up

    pole1_up = get_up_vector(rot[pole1_idx])
    pole2_up = get_up_vector(rot[pole2_idx])

    # We want the up vector to point as close to +Y as possible
    upright_loss = (
        (1.0 - pole1_up[1])**2 + 
        (1.0 - pole2_up[1])**2
    )

    # 2. Pole tip height (encourages vertical pose)
    # Higher tip means more vertical (tip is at pos + rotated pole length)
    pole1_pos = pos[pole1_idx]
    pole2_pos = pos[pole2_idx]

    pole1_height = pole1_pos[1]
    pole2_height = pole2_pos[1]

    height_reward = pole1_height + pole2_height  # want to maximize

    # 3. Angular velocity loss (encourages stability)
    angular_loss = (
        jnp.sum(ang[pole1_idx] ** 2) +
        jnp.sum(ang[pole2_idx] ** 2)
    )

    # 4. Linear velocity loss
    linear_loss = (
        jnp.sum(vel[pole1_idx] ** 2) +
        jnp.sum(vel[pole2_idx] ** 2)
    )

    # 5. Cart position drift
    cart_x_pos = pos[cart_idx][0]
    cart_pos_loss = cart_x_pos ** 2

    # 6. Action magnitude penalty (smoothness)
    action_loss = jnp.sum(action ** 2)

    # Total loss: lower is better
    total_loss = (
        10.0 * upright_loss -
        5.0 * height_reward +
        1.0 * angular_loss +
        0.1 * linear_loss +
        1.0 * cart_pos_loss +
        0.01 * action_loss
    )

    return total_loss


env.reset(env_key)
memory_settings = MemorySettings(20, env.observation_size, env.action_size, filter=filter)
agent_settings = FFAgentDefaultSettings.replace(learning_rate=1e-3) # type: ignore
agent = LinearAgent(memory_settings, agent_settings, rng=agent_key)


N = 200
T = 2

losses = []

# Slowly build up the agent's knowledge, first with one step, then working up to multiple steps.
training_plan = [
    (1, 2, 100), # First train for 100 iterations with 1 step averaged over 2 episodes
    (2, 2, 500), # Then train for 500 iterations with 2 steps averaged over 2 episodes
    (3, 3, 500), # ...
    (4, 3, 500),
    (5, 3, 500),
    (6, 3, 500),
    (7, 3, 300),
    (8, 3, 300),
    (10, 5, 500)
]

for T, N, iters in progress(training_plan, f"Training Agent according to plan"):
    rng, train_key = jax.random.split(rng)
    for i in progress(range(iters), f"Training with {N} episodes of length {T}"):
        losses_out = agent.train(env, N, T, train_key)
        losses.append(losses_out)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(losses)

# Plot vertical lines at the end of each training plan
cum_iters = 0
for T, N, iters in training_plan:
    cum_iters += iters
    plt.axvline(x=cum_iters, color='r', linestyle='--')


plt.title('Training Loss')
plt.xlabel('Step')
plt.ylabel('Loss')

plt.tight_layout()


rng, reset_key = jax.random.split(rng)
memory = Memory(memory_settings).reset_env(env, reset_key)

states = [memory.state]
# Run the trained model for 100 steps
for rng in jax.random.split(rng, 100):
    agent_key, env_key = jax.random.split(rng)
    action = agent(memory.get_history(), rng)
    memory = memory.act(env, action, env_key)
    states.append(memory.state)

with open("out.html", "w") as f:
    f.write(env.render(states))
    print("saved.")


plt.show()
