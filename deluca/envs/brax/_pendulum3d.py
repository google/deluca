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
from brax.io import mjcf
from deluca.envs._brax import BraxEnv


class Pendulum3D(BraxEnv):
    """3D Pendulum environment using Brax."""

    XML_CONFIG = """
<mujoco model="pendulum3d">
    <option timestep="0.01" />
    <worldbody>
    <body name="anchor" pos="0 0 0">
    <geom type="capsule" size="0.5 0.5" mass="1" />
    <joint type="ball" name="joint1" pos="0 0 0" range="-180 180" stiffness="10000" damping="0" />
    <body name="rod" pos="-2.5 0 0">
        <geom type="capsule" size="0.1 5" mass="1" euler="0 90 0" />
        <joint type="ball" name="joint2" pos="2.5 0 -2.5" range="-180 180" stiffness="10000" damping="0" />
        <body name="pen" pos="2.5 0 -5">
        <geom type="capsule" size="0.1 5" mass="1" />
        </body>
    </body>
    </body>
    </worldbody>
    <actuator>
    <motor name="joint1" joint="joint1" gear="1" />
    <motor name="joint2" joint="joint2" gear="1" />
    </actuator>
</mujoco>
"""

    def __init__(self):
        """Load the XML config in the constructor."""
        self.sys = mjcf.loads(self.XML_CONFIG)
