import jax.numpy as np
import jax
from deluca.envs.core import Env

class Reacher(Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self, parameter=0.0):
        self.H, self.dt, self.state_dim, self.action_dim = 200, 0.01, 6, 2
        self.m1, self.m2, self.l1, self.l2, self.g = 1.0, 1.0, 1.0, 1.0, parameter
        self.initial_th, self.goal_coord = (np.pi / 4, np.pi / 2), np.array([0.0, 1.8])
        self.initial_state = np.array(
            [
                *self.initial_th,
                0.0,
                0.0,
                self.l1 * np.cos(self.initial_th[0])
                + self.l2 * np.cos(self.initial_th[0] + self.initial_th[1])
                - self.goal_coord[0],
                self.l1 * np.sin(self.initial_th[0])
                + self.l2 * np.sin(self.initial_th[0] + self.initial_th[1])
                - self.goal_coord[1],
            ]
        )
        self.viewer, self.state, self.h = None, self.initial_state, 0
        self.nsamples = 0
        @jax.jit
        def f(x, u):
            self.nsamples += 1
            m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g
            th1, th2, dth1, dth2, Dx, Dy = x
            t1, t2 = u

            a11 = (m1 + m2) * l1 ** 2 + m2 * l2 ** 2 + 2 * m2 * l1 * l2 * np.cos(th2)
            a12 = m2 * l2 ** 2 + m2 * l1 * l2 * np.cos(th2)
            a22 = m2 * l2 ** 2
            b1 = (
                t1
                + m2 * l1 * l2 * (2 * dth1 + dth2) * dth2 * np.sin(th2)
                - m2 * l2 * g * np.sin(th1 + th2)
                - (m1 + m2) * l1 * g * np.sin(th1)
            )
            b2 = (
                t2
                - m2 * l1 * l2 * dth1 ** 2 * np.sin(th2)
                - m2 * l2 * g * np.sin(th1 + th2)
            )
            A, b = np.array([[a11, a12], [a12, a22]]), np.array([b1, b2])
            ddth1, ddth2 = np.linalg.inv(A) @ b

            th1, th2 = th1 + dth1 * self.dt, th2 + dth2 * self.dt
            dth1, dth2 = dth1 + ddth1 * self.dt, dth2 + ddth2 * self.dt
            Dx, Dy = (
                l1 * np.cos(th1) + l2 * np.cos(th1 + th2) - self.goal_coord[0],
                l1 * np.sin(th1) + l2 * np.sin(th1 + th2) - self.goal_coord[1],
            )

            return np.array([th1, th2, dth1, dth2, Dx, Dy])

        @jax.jit
        def c(x, u):
            # coord = np.array(
            #     [
            #         self.l1 * np.cos(x[0]) + self.l2 * np.cos(x[0] + x[1]),
            #         self.l1 * np.sin(x[0]) + self.l2 * np.sin(x[0] + x[1]),
            #     ]
            # )
            return 0.1 * u @ u + x[4:] @ x[4:]

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
        self.state, self.h = self.initial_state, 0
        return self.state

    def step(self, u):
        self.h += 1
        self.state = self.f(self.state, u)
        return self.state, self.c(self.state, u), False, {}

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)

            rod1 = rendering.make_capsule(self.l1, 0.2)
            rod1.set_color(0.8, 0.3, 0.3)
            self.pole_transform1 = rendering.Transform()
            rod1.add_attr(self.pole_transform1)
            self.viewer.add_geom(rod1)

            rod2 = rendering.make_capsule(self.l2, 0.2)
            rod2.set_color(0.3, 0.3, 0.8)
            self.pole_transform2 = rendering.Transform()
            rod2.add_attr(self.pole_transform2)
            self.viewer.add_geom(rod2)

            axle1 = rendering.make_circle(0.05)
            axle1.set_color(0, 0, 0)
            self.viewer.add_geom(axle1)

            axle2 = rendering.make_circle(0.05)
            axle2.set_color(0, 0, 0)
            self.pole_transform3 = rendering.Transform()
            axle2.add_attr(self.pole_transform3)
            self.viewer.add_geom(axle2)

            axle3 = rendering.make_circle(0.1)
            axle3.set_color(0, 0.5, 0)
            self.pole_transform4 = rendering.Transform()
            axle3.add_attr(self.pole_transform4)
            self.viewer.add_geom(axle3)

        self.pole_transform1.set_rotation(self.state[0])
        E1X, E1Y = self.l1 * np.cos(self.state[0]), self.l1 * np.sin(self.state[0])
        self.pole_transform2.set_translation(E1X, E1Y)
        self.pole_transform2.set_rotation(self.state[0] + self.state[1])
        self.pole_transform3.set_translation(E1X, E1Y)
        self.pole_transform4.set_translation(self.goal_coord[0], self.goal_coord[1])
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
