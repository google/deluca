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

"""Brax implementation of a pendulum."""
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
import jax.numpy as jp

from deluca.envs._brax import BraxEnv


class Pendulum2D(PipelineEnv):
    """2D Pendulum environment using Brax PipelineEnv."""

    XML_CONFIG = """
<mujoco model="pendulum2d">
  <option timestep="0.01" />
  <worldbody>
    <body name="anchor" pos="0 0 0">
      <geom type="capsule" size="0.5 0.5" mass="1" />
      <joint type="hinge" name="joint1" axis="0 1 0" pos="0 0 -2.5" range="-180 180" stiffness="10000" damping="20" />
      <body name="middle" pos="0 0 -2.5">
        <geom type="capsule" size="0.5 1" mass="1" />
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="joint1" joint="joint1" gear="1" />
  </actuator>
</mujoco>
"""

    @classmethod
    def create(cls):
        env = BraxEnv.from_env(cls())
        env.unfreeze()
        return env

    def __init__(self, backend="generalized", **kwargs):
        """Initialize the Pendulum2D environment."""
        sys = mjcf.loads(self.XML_CONFIG)
        super().__init__(sys=sys, backend=backend, **kwargs)

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        _, rng1, rng2 = jax.random.split(rng, 3)

        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=-0.01, maxval=0.01
        )
        qd = jax.random.normal(rng2, (self.sys.qd_size(),)) * 0.01
        pipeline_state = self.pipeline_init(q, qd)

        obs = self._get_obs(pipeline_state)
        reward, done = jp.zeros(2)
        metrics = {}

        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state = self.pipeline_step(state.pipeline_state, action)

        obs = self._get_obs(pipeline_state)
        # Simple reward: negative distance from upright position
        angle = pipeline_state.q[0]  # Assuming first joint is the pendulum angle
        reward = -jp.abs(angle)  # Reward for being close to 0 (upright)
        done = jp.float32(0)  # No termination condition for now

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    @property
    def action_size(self):
        return 1

    def _get_obs(self, pipeline_state) -> jax.Array:
        """Observe pendulum state."""
        return jp.concatenate(
            [
                jp.sin(pipeline_state.q),  # sin of angle
                jp.cos(pipeline_state.q),  # cos of angle
                jp.clip(pipeline_state.qd, -10, 10),  # angular velocity
            ]
        )
