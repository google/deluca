from typing import Any
from deluca.agents.new.feedforward import FFAgent, DefaultSettings as FFAgentDefaultSettings
from deluca.agents.new.linear import LinearAgent, DefaultSettings as LinearAgentDefaultSettings
from deluca.envs._brax import BraxEnv
from deluca.envs.classic._mountain_car import MountainCar
from deluca.envs.classic._pendulum import Pendulum
from deluca.filters.spectral import SpectralFilter
from deluca.learners.memory import Memory
from deluca.envs._lds import LDS, SinusDisturbance, ZeroDisturbance
from deluca.normalizers.core import DefaultNormalizers
from deluca.agents._random import SimpleRandom
import jax
import jax.numpy as jnp
from jax import Array
import numpy as np
import flax.nnx as nnx

import matplotlib.pyplot as plt

from deluca.utils.printing import Task

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

memory = Memory(1, env.observation_size, env.action_size)
# normalizer = DefaultNormalizers()
# random_agent = SimpleRandom(env.action_size, random_key)

# # Prime the normalizer with random actions
# t, state, obs = env.reset(env_key)
# for i in range(1000):
#     action = random_agent(obs, jax.random.split(random_key, i + 1)[-1])
#     t, state, obs = env(t, state, action, random_key)
#     memory.store(obs, action)

# obs_history, action_history = memory.get_history()
# normalizer.compute_normalization(obs_history, action_history)

# def loss_fn(action: Array, next_state: Any, next_obs: Array) -> Array:
#     obs = next_state.obs
#     # obs[0:2] = hand position
#     # obs[2:4] = target position
#     hand_pos = obs[0:2]
#     target_pos = obs[2:4]

#     # Distance between hand and goal
#     dist = jnp.linalg.norm(hand_pos - target_pos)

#     # Control cost: penalize large actions
#     control_cost = 0.01 * jnp.sum(jnp.square(action))

#     # Convert reward into loss
#     return dist + control_cost

def loss_fn(action: Array, next_state: Any, next_obs: Array) -> Array:
    # The pendulum should be upright + penalize large actions
    return jnp.sum(jnp.square(action)) + jnp.sum(jnp.square(next_state.obs))

agent_settings = FFAgentDefaultSettings.replace(learning_rate=1e-3, loss_fn=loss_fn) # type: ignore
agent = FFAgent(memory, agent_settings, rng=agent_key)

# Give agent a chance to learn
rng, env_key, agent_key = jax.random.split(rng, 3)
t, state, obs = env.reset(env_key)

losses = []
actions = []
obses = []
states = [state]

N = 1000 

with Task("Training agent", N) as task:
    for i, key in enumerate(jax.random.split(agent_key, N)):
        agent_key, env_key = jax.random.split(agent_key)
        
        loss, (new_t, new_state, new_obs), action = agent.step_and_update(env, t, state, memory.get_history(), agent_key)
        
        if (new_t == 0):
            memory.reset() # Episode reset

        memory.store(obs, action)

        losses.append(loss)
        actions.append(action)
        obses.append(new_obs)
        states.append(new_state)

        t, state, obs = new_t, new_state, new_obs
        task.update(text=f"Loss: {loss:.4f}")

print(actions[-10:])
print(obses[-10:])

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Step')
plt.ylabel('Loss')

plt.tight_layout()

env.render(states)
plt.show()

