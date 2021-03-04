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
import jax
import jax.numpy as jnp
import numpy as np
from deluca.envs.core import Env
from deluca.utils import Random
from os import path

class LDS(Env):
    def __init__(self, state_size=1, action_size=1, A=None, B=None, C=None, seed=0, noise="normal"):

        if A is not None:
            assert (
                A.shape[0] == state_size and A.shape[1] == state_size
            ), "ERROR: Your input dynamics matrix does not have the correct shape."

        if B is not None:
            assert (
                B.shape[0] == state_size and B.shape[1] == action_size
            ), "ERROR: Your input dynamics matrix does not have the correct shape."

        self.viewer = None
        self.random = Random(seed)
        self.noise = noise
            
        self.state_size, self.action_size = state_size, action_size
        self.proj_vector1 = jax.random.normal(self.random.generate_key(), (state_size,))
        self.proj_vector1 = self.proj_vector1/jnp.linalg.norm(self.proj_vector1)
        self.proj_vector2 = jax.random.normal(self.random.generate_key(), (state_size,))
        self.proj_vector2 = self.proj_vector2/jnp.linalg.norm((self.proj_vector2))
        print('self.proj_vector1:' + str(self.proj_vector1))
        print('self.proj_vector2:' + str(self.proj_vector2))

        self.A = (
            A
            if A is not None
            else jax.random.normal(self.random.generate_key(), shape=(state_size, state_size))
        )

        self.B = (
            B
            if B is not None
            else jax.random.normal(self.random.generate_key(), shape=(state_size, action_size))
        )

        self.C = (
            C
            if C is not None
            else jax.numpy.identity(self.state_size)
        )

        self.t = 0

        self.reset()

    def step(self, action):
        self.state = self.A @ self.state + self.B @ action
        if self.noise == "normal":
            self.state += np.random.normal(0, 0.1, size=(self.state_size,self.action_size))
        self.obs = self.C @ self.state

    @jax.jit
    def dynamics(self, state, action):
        new_state = self.A @ state + self.B @ action
        return new_state

    def reset(self):
        self.state = jax.random.normal(self.random.generate_key(), shape=(self.state_size, 1))
        self.obs = self.C @ self.state

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(1000, 1000)
            self.viewer.set_bounds(-2.5, 2.5, -2.5, 2.5)
            fname = path.dirname(__file__) + "/classic/assets/lds_arrow.png"
            self.img = rendering.Image(fname, 0.35, 0.35)
            self.img.set_color(1.0, 1.0, 1.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)
            fnamewind = path.dirname(__file__) + "/classic/assets/lds_grid.png"
            self.imgwind = rendering.Image(fnamewind, 5.0, 5.0)
            self.imgwind.set_color(0.5, 0.5, 0.5)
            self.imgtranswind = rendering.Transform()
            self.imgwind.add_attr(self.imgtranswind)

        self.viewer.add_onetime(self.imgwind)
        self.viewer.add_onetime(self.img)
        cur_x, cur_y = self.imgtrans.translation
        # new_x, new_y = self.state[0], self.state[1]
        new_x, new_y = jnp.dot(self.state.squeeze(), self.proj_vector1), jnp.dot(self.state.squeeze(), self.proj_vector2)
        diff_x = new_x - cur_x
        diff_y = new_y - cur_y
        new_rotation = jnp.arctan2(diff_y, diff_x)
        self.imgtrans.set_translation(new_x, new_y)
        self.imgtrans.set_rotation(new_rotation)
                    
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
