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


class Pendulum3D(BraxEnv):
  """Example here:
  https://colab.sandbox.google.com/drive/1rlIZUTFc5B2R_mYT7Ukf4Y4Q3-BGQDrN
  """

  def setup(self):
    """setup."""
    pendulum = brax.Config(dt=0.01, substeps=100)

    rod_length = 5
    rod_radius = 0.1
    pen_length = 5
    pen_radius = 0.1

    # start with a frozen anchor at the root of the pendulum
    anchor = pendulum.bodies.add(name='anchor', mass=1.0)
    anchor.frozen.all = True
    anchor.inertia.x, anchor.inertia.y, anchor.inertia.z = 1, 1, 1
    anchor.colliders.add()
    anchor.colliders[0].capsule.radius = 0.5
    anchor.colliders[0].capsule.length = 1.0

    rod = pendulum.bodies.add(name='rod', mass=1.0)
    rod.inertia.x, rod.inertia.y, rod.inertia.z = 1, 1, 1
    cap = rod.colliders.add().capsule
    cap.radius, cap.length = rod_radius, rod_length
    rod.colliders[0].rotation.y = 90

    pen = pendulum.bodies.add(name='pen', mass=1.0)
    pen.inertia.x, pen.inertia.y, pen.inertia.z = 1, 1, 1
    cap = pen.colliders.add().capsule
    cap.radius, cap.length = pen_radius, pen_length

    joint = pendulum.joints.add(
        name='joint1',
        parent='anchor',
        child='rod',
        stiffness=10000,
        angular_damping=0)
    joint.angle_limit.add(min=-180, max=180)
    joint.angle_limit.add(min=-180, max=180)
    joint.child_offset.x = -rod_length / 2
    joint.rotation.y = 90

    joint = pendulum.joints.add(
        name='joint2',
        parent='rod',
        child='pen',
        stiffness=10000,
        angular_damping=0)
    joint.angle_limit.add(min=-180, max=180)
    joint.child_offset.z = -pen_length / 2
    joint.parent_offset.x = pen_length / 2
    joint.rotation.x = 90

    # gravity is -9.8 m/s^2 in z dimension
    pendulum.gravity.z = -9.8

    # ignore collisions
    pendulum.collide_include.add()

    self.sys = brax.System(pendulum)
