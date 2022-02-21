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
import brax

from deluca.envs._brax import BraxEnv


class Pendulum2D(BraxEnv):
  """Example here:

  https://colab.sandbox.google.com/drive/1iBumRjVMfsIjnGrzsmOvGypSjWsdsupC
  """

  def setup(self):
    """setup."""
    pendulum = brax.Config(dt=0.01, substeps=100)

    # start with a frozen anchor at the root of the pendulum
    anchor = pendulum.bodies.add(name='anchor', mass=1.0)
    anchor.frozen.all = True
    anchor.inertia.x, anchor.inertia.y, anchor.inertia.z = 1, 1, 1

    ball = pendulum.bodies.add(name='ball', mass=1)
    ball.inertia.x, ball.inertia.y, ball.inertia.z = 1, 1, 1
    cap = ball.colliders.add().capsule
    cap.radius, cap.length = 0.5, 1

    # now add a middle and bottom ball to the pendulum
    pendulum.bodies.append(ball)
    pendulum.bodies[1].name = 'middle'

    # connect anchor to middle
    joint = pendulum.joints.add(
        name='joint1',
        parent='anchor',
        child='middle',
        stiffness=10000,
        angular_damping=20)
    joint.angle_limit.add(min=-180, max=180)
    joint.child_offset.z = -2.5
    joint.rotation.z = 90

    # gravity is -9.8 m/s^2 in z dimension
    pendulum.gravity.z = -9.8

    # ignore collisions
    pendulum.collide_include.add()

    self.sys = brax.System(pendulum)
