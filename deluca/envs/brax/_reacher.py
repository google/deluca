# Copyright 2021 The Deluca Authors.
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
import jax
import jax.numpy as jnp

from deluca.envs._brax import BraxEnv


class Reacher(BraxEnv):
  """Brax reacher with random initialization."""

  def init(self, key: jnp.ndarray = None):
    """Init function."""
    key, key1, key2 = jnp.random_split(key, 3)
    qpos = self.sys.default_angle() + jnp.random_uniform(
        key1, (self.sys.num_joint_dof,), -.1, .1)
    qvel = jnp.random_uniform(key2, (self.sys.num_joint_dof,), -.005, .005)
    qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
    _, target = self._random_target(key)
    pos = jnp.index_update(qp.pos, self.sys.body.index['target'], target)
    qp = qp.replace(pos=pos)

    obs = qp if self.env is None else self.env._get_obs(
        qp, self.sys.info(qp))

    return qp, obs

  def _random_target(self, rng: jnp.ndarray):
    """Returns a target location in a random circle slightly above xy plane."""
    rng, rng1, rng2 = jnp.random_split(rng, 3)
    dist = .2 * jnp.random_uniform(rng1)
    ang = jnp.pi * 2. * jnp.random_uniform(rng2)
    target_x = dist * jnp.cos(ang)
    target_y = dist * jnp.sin(ang)
    target_z = .01
    target = jnp.array([target_x, target_y, target_z]).transpose()
    return rng, target
