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
from os import path

import jax
import jax.numpy as jnp
import numpy as ojnp
from scipy.optimize._numdiff import approx_derivative as scipy_approx_derivative

from deluca.envs.core import Env


def approx_derivative(f, x, method):
    if "alg" in method:
        if method["alg"] in ["2-point", "3-point", "cs"]:  # grid-based finite differences
            if "rel_step" in method:
                rel_step = method["rel_step"]
            else:
                rel_step = None
            # print(jnp.linalg.norm(x))
            return scipy_approx_derivative(f, x, method["alg"], rel_step=rel_step)

        elif method["alg"] == "es":  # Monte Carlo Gaussian smoothing derivative
            n = method["n"]
            sigma = method["sigma"]

            d_in = len(x)
            G = ojnp.random.randn(n, d_in)
            J_est = None

            for g in G:
                J_samp = ojnp.outer(f(x + sigma * g) - f(x - sigma * g), g) / 2
                if J_est is None:
                    J_est = J_samp
                else:
                    J_est += J_samp

            return J_est / (n * sigma)

    raise NotImplementedError


class PlanarQuadrotor(Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, placebo=0, autodiff=True, method="3-point"):
        self.nsamples = 0
        self.m, self.l, self.g, self.dt, self.H = 0.1, 0.2, 9.81, 0.05, 100
        self.initial_state, self.goal_state, self.goal_action = (
            jnp.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            jnp.array([self.m * self.g / 2.0, self.m * self.g / 2.0]),
        )
        self.goal_action = jnp.hstack((self.goal_action, jnp.zeros(placebo)))

        self.viewer = None
        self.action_dim, self.state_dim = 2 + placebo, 6
        self.last_u = jnp.zeros((2,))

        def f(x, u):
            self.nsamples += 1
            state = x
            x, y, th, xdot, ydot, thdot = state
            u1, u2 = u[:2]
            m, g, l, dt = self.m, self.g, self.l, self.dt
            xddot = -(u1 + u2) * jnp.sin(th) / m
            yddot = (u1 + u2) * jnp.cos(th) / m - g
            thddot = l * (u2 - u1) / (m * l ** 2)
            state_dot = jnp.array([xdot, ydot, thdot, xddot, yddot, thddot])
            new_state = state + state_dot * dt
            return new_state

        def c(x, u):
            return 0.1 * (u - self.goal_action) @ (u - self.goal_action) + (x - self.goal_state) @ (
                x - self.goal_state
            )

        def f_x(x, u):
            return approx_derivative(lambda x: f(x, u), x, method=method)

        def f_u(x, u):
            return approx_derivative(lambda u: f(x, u), u, method=method)

        def c_x(x, u):
            return 2 * (x - self.goal_state)

        def c_u(x, u):
            return 2 * 0.1 * (u - self.goal_action)

        def c_xx(x, u):
            return 2 * jnp.eye(self.state_dim)

        def c_uu(x, u):
            return 2 * 0.1 * jnp.eye(self.action_dim)

        if autodiff:
            self.f, self.f_x, self.f_u = (
                f,
                jax.jacfwd(f, argnums=0),
                jax.jacfwd(f, argnums=1),
            )
            self.c, self.c_x, self.c_u, self.c_xx, self.c_uu = (
                c,
                jax.grad(c, argnums=0),
                jax.grad(c, argnums=1),
                jax.hessian(c, argnums=0),
                jax.hessian(c, argnums=1),
            )
        else:
            self.f, self.f_x, self.f_u = f, f_x, f_u
            self.c, self.c_x, self.c_u, self.c_xx, self.c_uu = c, c_x, c_u, c_xx, c_uu

    def reset(self):
        self.state, self.last_u, self.h = self.initial_state, None, 0
        return self.state

    def step(self, u):
        self.last_u, self.h = u, self.h + 1
        self.state = self.f(self.state, u)
        return self.state, self.c(self.state, u)

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(1000, 1000)
            self.viewer.set_bounds(-1.2, 1.2, -1.2, 1.2)
            fname = path.dirname(__file__) + "/assets/drone.png"
            self.img = rendering.Image(fname, 0.4, 0.17)
            self.img.set_color(1.0, 1.0, 1.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)
            fnamewind = path.dirname(__file__) + "/assets/wind.png"
            self.imgwind = rendering.Image(fnamewind, 2.4, 2.4)
            self.imgwind.set_color(0.5, 0.5, 0.5)
            self.imgtranswind = rendering.Transform()
            self.imgwind.add_attr(self.imgtranswind)

        self.viewer.add_onetime(self.imgwind)
        self.viewer.add_onetime(self.img)
        self.imgtrans.set_translation(self.state[0], self.state[1] + 0.04)
        self.imgtrans.set_rotation(self.state[2])
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
