# Copyright 2022 The Deluca Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ppo_run."""
from typing import Sequence

from absl import app
from absl import flags
from brax import envs
from deluca.core import Agent
from deluca.envs import BraxEnv
from deluca.lung.controllers import DeepAC
from deluca.lung.core import BreathWaveform
from deluca.lung.envs import BalloonLung
from deluca.training import ppo
import jax.numpy as jnp
import ml_collections
import dill as pickle

flags.DEFINE_string(
    "name", "balloon", "Name for this experiment.", short_name="n")


def get_config():
  """Get the default configuration.

  The default hyperparameters originate from PPO paper arXiv:1707.06347
  and openAI baselines 2::
  https://github.com/openai/baselines/blob/master/baselines/ppo2/defaults.py
  """
  config = ml_collections.ConfigDict()
  # The learning rate for the Adam optimizer.
  config.learning_rate = 2.5e-4
  # Batch size used in training.
  config.batch_size = 28
  # Number of agents playing in parallel.
  config.num_agents = 2
  # Number of steps each agent performs in one policy unroll.
  config.actor_steps = 28
  # Number of training epochs per each unroll of the policy.
  config.num_epochs = 3
  # Number of episodes for training
  config.num_episodes = 10
  # RL discount parameter.
  config.gamma = 0.99
  # Generalized Advantage Estimation parameter.
  config.lambda_ = 0.95
  # The PPO clipping parameter used to clamp ratios in loss function.
  config.clip_param = 0.1
  # Weight of value function loss in the total loss.
  config.vf_coeff = 0.5
  # Weight of entropy bonus in the total loss.
  config.entropy_coeff = 0.01
  # Linearly decay learning rate and clipping parameter to zero during
  # the training.
  config.decaying_lr_and_clip_param = True
  config.optimization_params = {"learning_rate": config.learning_rate}
  return config


def run_balloon_lung():
  """Run function."""
  waveform = BreathWaveform.create()

  def rewards_func(state, obs, action, env, counter):
    del state, action, env, counter
    return -(obs.predicted_pressure - waveform.at(obs.time))**2

  ppo.train(
      BalloonLung.create(),
      DeepAC.create(),
      rewards_func,
      horizon=29,
      config=get_config())


class DummyReacherAgent(Agent):
  """Dummy PPO agent."""

  def init(self):
    return {}

  def __call__(self, state, obs, *args, **kwargs):
    return {}, jnp.array([1., 0., 0.])

  def derive_prob_and_value(self, cont_state):
    pass

  def log_prob_acftion(self, log_probs, action):
    pass


def run_reacher():
  """Run reacher example."""
  reacher = BraxEnv.create(env=envs.reacher.Reacher())
  reacher_state, obs = reacher.init()
  agent = DummyReacherAgent.create()
  agent_state = agent.init()

  def rewards_func(state, obs, action, env, counter):
    """Reacher reward fn."""
    del state, env, counter
    reward_dist = -jnp.norm(obs[-3:])
    reward_ctrl = -jnp.square(action).sum()
    return reward_dist + reward_ctrl

  ppo.train(
      reacher, agent.create(), rewards_func, horizon=10, config=get_config())
  for _ in range(10):
    agent_state, action = agent(agent_state, obs)
    reacher_state, obs = reacher(reacher_state, action)
    print(obs[-3:], action)


def run_learned(f):
  env = pickle.load(f)["model_ckpt"]

  agent = DeepAC.create()
  waveform = BreathWaveform.create()

  def rewards_func(state, obs, action, env, counter):
    del state, action, env, counter
    return -(obs.predicted_pressure - waveform.at(obs.time))**2

  ppo.train(
      env=env,
      agent=agent,
      reward_fn=rewards_func,
      horizon=29,
      config=get_config())


def main(argv):
  if flags.FLAGS.name == "balloon":
    run_balloon_lung()
  elif flags.FLAGS.name == "reacher":
    run_reacher()
  elif flags.FLAGS.name == "learned":
    run_learned()


if __name__ == "__main__":
  app.run(main)
