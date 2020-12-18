import jax
import jax.numpy as np
from os import path
from deluca.envs.core import Env


###### THIS ENIVRONMENT WILL NOT EXPOSE DERIVATIVES OF DYNAMICS EVER ###########
class PlanarQuadrotorLambdaWind(Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, wind_func=lambda x, y, t: [0.0, 0.0]):
        self.m, self.l, self.g, self.dt, self.H, self.wind_func = (
            0.1,
            0.2,
            9.81,
            0.05,
            100,
            wind_func,
        )
        self.initial_state, self.goal_state, self.goal_action = (
            np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([self.m * self.g / 2.0, self.m * self.g / 2.0]),
        )

        self.viewer = None
        self.action_dim, self.state_dim = 2, 6

        def f(x, u, t=None):
            state = x
            x, y, th, xdot, ydot, thdot = state
            u1, u2 = u
            m, g, l, dt = self.m, self.g, self.l, self.dt
            wind = self.wind_func(x, y, t)
            xddot = -(u1 + u2) * np.sin(th) / m + wind[0] / m
            yddot = (u1 + u2) * np.cos(th) / m - g + wind[1] / m
            thddot = l * (u2 - u1) / (m * l ** 2)
            state_dot = np.array([xdot, ydot, thdot, xddot, yddot, thddot])
            new_state = state + state_dot * dt
            return new_state

        @jax.jit
        def c(x, u):
            return 0.1 * (u - self.goal_action) @ (u - self.goal_action) + (
                x - self.goal_state
            ) @ (x - self.goal_state)

        self.f, self.f_x, self.f_u = (
            f,
            jax.jit(jax.jacfwd(f, argnums=0)),
            jax.jit(jax.jacfwd(f, argnums=1)),
        )
        self.c, self.c_x, self.c_u, self.c_xx, self.c_uu = (
            c,
            jax.jit(jax.grad(c, argnums=0)),
            jax.jit(jax.grad(c, argnums=1)),
            jax.jit(jax.hessian(c, argnums=0)),
            jax.jit(jax.hessian(c, argnums=1)),
        )

    def reset(self, wind_func=None):
        if wind_func:
            self.wind_func = wind_func
        self.state, self.last_u, self.h = self.initial_state, None, 0

        return self.state

    def step(self, u):
        self.last_u, self.h = u, self.h + 1
        self.state = self.f(self.state, u)
        return self.state, self.c(self.state, u), False, {}

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(1000, 1000)
            self.viewer.set_bounds(-1.2, 1.2, -1.2, 1.2)
            fname = path.dirname(__file__) + "/drone.png"
            self.img = rendering.Image(fname, 0.4, 0.17)
            self.img.set_color(1.0, 1.0, 1.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)
            fnamewind = path.dirname(__file__) + "/wind.png"
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


class PlanarQuadrotorSimWind(Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, wind=0.0, wind_mode="DISSIPATIVE"):
        self.m, self.l, self.g, self.dt, self.H, self.wind, self.wind_mode = (
            0.1,
            0.2,
            9.81,
            0.05,
            100,
            wind,
            wind_mode,
        )
        self.initial_state, self.goal_state, self.goal_action = (
            np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([self.m * self.g / 2.0, self.m * self.g / 2.0]),
        )

        self.viewer = None
        self.action_dim, self.state_dim = 2, 6

        @jax.jit
        def wind_field(x, y):
            if self.wind_mode == "DISSIPATIVE":
                return [self.wind * x, self.wind * y]
            elif self.wind_mode == "TORNADOES":
                pos1 = [1.0, 0.0]
                pos2 = [0.5, 0.5]
                pos3 = [0.0, 1.0]
                return [
                    self.wind * (pos1[0] + pos2[0] + pos3[0] - 3 * x),
                    self.wind * (pos1[0] + pos2[0] + pos3[0] - 3 * y),
                ]
            elif self.wind_mode == "CONSTANT":
                theta = np.pi / 4.0
                return [self.wind * np.cos(theta), self.wind * np.sin(theta)]
            else:
                print("WIND MODE NOT RECOGNIZED")
                exit()

        @jax.jit
        def f(x, u):
            state = x
            x, y, th, xdot, ydot, thdot = state
            u1, u2 = u
            m, g, l, dt = self.m, self.g, self.l, self.dt
            wind = wind_field(x, y)
            xddot = -(u1 + u2) * np.sin(th) / m + wind[0] / m
            yddot = (u1 + u2) * np.cos(th) / m - g + wind[1] / m
            thddot = l * (u2 - u1) / (m * l ** 2)
            state_dot = np.array([xdot, ydot, thdot, xddot, yddot, thddot])
            new_state = state + state_dot * dt
            return new_state

        @jax.jit
        def c(x, u):
            return 0.1 * (u - self.goal_action) @ (u - self.goal_action) + (
                x - self.goal_state
            ) @ (x - self.goal_state)

        self.f, self.f_x, self.f_u = (
            f,
            jax.jit(jax.jacfwd(f, argnums=0)),
            jax.jit(jax.jacfwd(f, argnums=1)),
        )
        self.c, self.c_x, self.c_u, self.c_xx, self.c_uu = (
            c,
            jax.jit(jax.grad(c, argnums=0)),
            jax.jit(jax.grad(c, argnums=1)),
            jax.jit(jax.hessian(c, argnums=0)),
            jax.jit(jax.hessian(c, argnums=1)),
        )

    def reset(self):
        self.state, self.last_u, self.h = self.initial_state, None, 0
        return self.state

    def step(self, u):
        self.last_u, self.h = u, self.h + 1
        self.state = self.f(self.state, u)
        return self.state, self.c(self.state, u), False, {}

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(1000, 1000)
            self.viewer.set_bounds(-1.2, 1.2, -1.2, 1.2)
            fname = path.dirname(__file__) + "/drone.png"
            self.img = rendering.Image(fname, 0.4, 0.17)
            self.img.set_color(1.0, 1.0, 1.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)
            fnamewind = path.dirname(__file__) + "/wind.png"
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