# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
@author: Olivier Sigaud
A merge between two sources:
* Adaptation of the MountainCar Environment from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0), adapted by  'Tom Schaul, tom@idsia.ch'
and then modified by Arnaud de Broissia
* the OpenAI/gym MountainCar environment
itself from
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
from gym import spaces

from deluca.envs.core import Env
from deluca.utils import Random


class MountainCar(Env):
    def __init__(self, goal_velocity=0, seed=0, horizon=50):
        self.viewer = None
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.goal_velocity = goal_velocity
        self.power = 0.0015
        self.H = horizon
        self.action_dim = 1
        self.random = Random(seed)

        self.low_state = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high_state = np.array([self.max_position, self.max_speed], dtype=np.float32)

        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )
        self.nsamples = 0
        
        # @jax.jit
        def _dynamics(state, action):
            self.nsamples += 1
            position = state[0]
            velocity = state[1]

            force = jnp.minimum(jnp.maximum(action, self.min_action), self.max_action)

            velocity += force * self.power - 0.0025 * jnp.cos(3 * position)
            velocity = jnp.clip(velocity, -self.max_speed, self.max_speed)

            position += velocity
            position = jnp.clip(position, self.min_position, self.max_position)
            reset_velocity = (position == self.min_position) & (velocity < 0)
            # print('state.shape = ' + str(state.shape))
            # print('position.shape = ' + str(position.shape))
            # print('velocity.shape = ' + str(velocity.shape))
            # print('reset_velocity.shape = ' + str(reset_velocity.shape))
            velocity = jax.lax.cond(reset_velocity[0], velocity, lambda x: jnp.zeros((1,)), velocity, lambda x: x)
            # print('velocity.shape AFTER = ' + str(velocity.shape))
            return jnp.reshape(jnp.array([position, velocity]), (2,))
        
        @jax.jit
        def c(x, u):
            position, velocity = self.state[0], self.state[1]
            done = (position >= self.goal_position) & (velocity >= self.goal_velocity)
            return -100.0 * done + 0.1*(u[0]+1)**2
        self.reward_fn = c
        self.dynamics = _dynamics
        self.f, self.f_x, self.f_u = (
                _dynamics,
                jax.jacfwd(_dynamics, argnums=0),
                jax.jacfwd(_dynamics, argnums=1),
            )
        self.c, self.c_x, self.c_u, self.c_xx, self.c_uu = (
                c,
                jax.grad(c, argnums=0),
                jax.grad(c, argnums=1),
                jax.hessian(c, argnums=0),
                jax.hessian(c, argnums=1),
            )
                           

        self.reset()

    def step(self, action):
        self.state = self.dynamics(self.state, action)
        position = self.state[0]
        velocity = self.state[1]

        # Convert a possible numpy bool to a Python bool.
        done = (position >= self.goal_position) & (velocity >= self.goal_velocity)

        # reward = 100.0 * done
        # reward -= jnp.power(action, 2) * 0.1
        reward = self.reward_fn(self.state, action)

        return self.state, reward, done, {}

    def reset(self):
        self.state = jnp.array(
            [jax.random.uniform(self.random.generate_key(), minval=-0.6, maxval=0.4), 0]
        )
        return self.state

    def _height(self, xs):
        return jnp.sin(3 * xs) * 0.45 + 0.55

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(0.5, 0.5, 0.5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth / 4, clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth / 4, clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position - self.min_position) * scale
            flagy1 = self._height(self.goal_position) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(0.8, 0.8, 0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation((pos - self.min_position) * scale, self._height(pos) * scale)
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array=mode == "rgb_array")
