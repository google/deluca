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
"""classic Acrobot task"""
import jax
import jax.numpy as jnp
import numpy as np
from gym import spaces
from jax.numpy import cos
from jax.numpy import pi
from jax.numpy import sin
from jax.numpy import arcsin

from deluca.envs.core import Env
from deluca.utils import Random

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = [
    "Alborz Geramifard",
    "Robert H. Klein",
    "Christoph Dann",
    "William Dabney",
    "Jonathan P. How",
]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann <cdann@cdann.de>"

# SOURCE:
# https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py


class Acrobot(Env):

    """
    Acrobot is a 2-link pendulum with only the second joint actuated.
    Initially, both links point downwards. The goal is to swing the
    end-effector at a height at least the length of one link above the base.
    Both links can swing freely and can pass by each other, i.e., they don't
    collide when they have the same angle.
    **STATE:**
    The state consists of the sin() and cos() of the two rotational joint
    angles and the joint angular velocities :
    [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].
    For the first link, an angle of 0 corresponds to the link pointing downwards.
    The angle of the second link is relative to the angle of the first link.
    An angle of 0 corresponds to having the same angle between the two links.
    A state of [1, 0, 1, 0, ..., ...] means that both links point downwards.
    **ACTIONS:**
    The action is either applying +1, 0 or -1 torque on the joint between
    the two pendulum links.
    **REFERENCE:**
    .. warning::
        This version of the domain uses the Runge-Kutta method for integrating
        the system dynamics and is more realistic, but also considerably harder
        than the original version which employs Euler integration,
        see the AcrobotLegacy class.
    """

    dt = 0.2

    LINK_LENGTH_1 = 1.0  # [m]
    LINK_LENGTH_2 = 1.0  # [m]
    LINK_MASS_1 = 1.0  #: [kg] mass of link 1
    LINK_MASS_2 = 1.0  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
    LINK_MOI = 1.0  #: moments of inertia for both links

    MAX_VEL_1 = 4 * pi
    MAX_VEL_2 = 9 * pi

    AVAIL_TORQUE = jnp.array([-1.0, 0.0, +1])

    torque_noise_max = 0.0

    #: use dynamics equations from the nips paper or the book
    book_or_nips = "book"
    action_arrow = None
    domain_fig = None
    actions_num = 3

    def __init__(self, seed=0, horizon=50):
        high = np.array([1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2], dtype=np.float32)
        low = -high
        self.random = Random(seed)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.action_dim = 1
        self.H = horizon
        self.nsamples = 0

        # @jax.jit
        def _dynamics(state, action):
            self.nsamples += 1
            # Augment the state with our force action so it can be passed to _dsdt
            augmented_state = jnp.append(state, action)

            new_state = rk4(self._dsdt, augmented_state, [0, self.dt])
            # only care about final timestep of integration returned by integrator
            new_state = new_state[-1]
            new_state = new_state[:4]  # omit action
            # ODEINT IS TOO SLOW!
            # ns_continuous = integrate.odeint(self._dsdt, self.s_continuous, [0, self.dt])
            # self.s_continuous = ns_continuous[-1] # We only care about the state
            # at the ''final timestep'', self.dt
                        
            new_state = jax.ops.index_update(new_state, 0, wrap(new_state[0], -pi, pi))
            new_state = jax.ops.index_update(new_state, 1, wrap(new_state[1], -pi, pi))
            new_state = jax.ops.index_update(
                new_state, 2, bound(new_state[2], -self.MAX_VEL_1, self.MAX_VEL_1)
            )
            new_state = jax.ops.index_update(
                new_state, 3, bound(new_state[3], -self.MAX_VEL_2, self.MAX_VEL_2)
            )
            
            return new_state
        
        # @jax.jit
        def c(x, u):
            return u[0]**2 + 1 - jnp.exp(0.5*cos(x[0]) + 0.5*cos(x[1]) - 1)
        self.dynamics = _dynamics
        self.reward_fn = c
        
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

    def reset(self):
        self.state = jax.random.uniform(
            self.random.generate_key(), shape=(4,), minval=-0.1, maxval=0.1
            # self.random.generate_key(), shape=(6,), minval=-0.1, maxval=0.1
        )
        return self.state

    def step(self, action):
        # torque = self.AVAIL_TORQUE[action] # discrete action space
        torque = bound(action, -1.0, 1.0) # continuous action space

        # Add noise to the force action
        if self.torque_noise_max > 0:
            torque += self.np_random.uniform(-self.torque_noise_max, self.torque_noise_max)
        self.state = self.dynamics(self.state, torque)
        terminal = self._terminal()
        # reward = -1.0 + terminal # openAI cost function
        reward = self.reward_fn(self.state, action)

        # TODO: should this return self.state (dim 4) or self.observation (dim 6)?
        return self.state, reward, terminal, {}

    @property
    def observation(self):
        return jnp.array(
            [
                cos(self.state[0]),
                sin(self.state[0]),
                cos(self.state[1]),
                sin(self.state[1]),
                self.state[2],
                self.state[3],
            ]
        )

    def _terminal(self):
        return -cos(self.state[0]) - cos(self.state[1] + self.state[0]) > 1.0
    

    def _dsdt(self, augmented_state, t):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = 9.8
        a = augmented_state[-1]
        s = augmented_state[:-1]
        
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
        
        d1 = m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * cos(theta2)) + I1 + I2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * cos(theta2)) + I2
        phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.0)
        phi1 = (
            -m2 * l1 * lc2 * dtheta2 ** 2 * sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2)
            + phi2
        )
        if self.book_or_nips == "nips":
            # the following line is consistent with the description in the
            # paper
            ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        else:
            # the following line is consistent with the java implementation and the
            # book
            ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * sin(theta2) - phi2) / (
                m2 * lc2 ** 2 + I2 - d2 ** 2 / d1
            )
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return (dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0) # 4-state version    


def wrap(x, m, M):
    """Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.
    Args:
        x: a scalar
        m: minimum possible value in range
        M: maximum possible value in range
    Returns:
        x: a scalar, wrapped
    """
    diff = M - m

    if diff == 0:
        return x

    to_subtract = jnp.ceil(jnp.maximum(0, x - M) / diff)
    x -= to_subtract * diff

    to_add = jnp.ceil(jnp.maximum(0, m - x) / diff)
    x += to_add * diff

    return x


def bound(x, m, M=None):
    """Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].
    Args:
        x: scalar
    Returns:
        x: scalar, bound between min (m) and Max (M)
    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return jnp.minimum(jnp.maximum(x, m), M)


def rk4(derivs, y0, t, *args, **kwargs):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.
    Args:
        derivs: the derivative of the system and has the signature ``dy = derivs(yi, ti)``
        y0: initial state vector
        t: sample times
        args: additional arguments passed to the derivative function
        kwargs: additional keyword arguments passed to the derivative function
    Example 1 ::
        ## 2D system
        def derivs6(x,t):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)
    Example 2::
        ## 1D system
        alpha = 2
        def derivs(x,t):
            return -alpha*x + exp(-t)
        y0 = 1
        yout = rk4(derivs, y0, t)
    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    Returns:
        yout: Runge-Kutta approximation of the ODE
    """

    try:
        Ny = len(y0)
    except TypeError:
        yout = jnp.zeros((len(t),), np.float_)
    else:
        yout = jnp.zeros((len(t), Ny), np.float_)

    yout = jax.ops.index_update(yout, 0, y0)

    for i in np.arange(len(t) - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = jnp.asarray(derivs(y0, thist, *args, **kwargs))
        # print('dt2.shape = ' + str(dt2.shape))
        k2 = jnp.asarray(derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
        k3 = jnp.asarray(derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
        k4 = jnp.asarray(derivs(y0 + dt * k3, thist + dt, *args, **kwargs))
        yout = jax.ops.index_update(yout, i + 1, y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4))

    return yout

    def render(self, mode="human"):
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
            self.viewer.set_bounds(-bound, bound, -bound, bound)

        if s is None:
            return None

        p1 = [-self.LINK_LENGTH_1 * cos(s[0]), self.LINK_LENGTH_1 * sin(s[0])]

        p2 = [
            p1[0] - self.LINK_LENGTH_2 * cos(s[0] + s[1]),
            p1[1] + self.LINK_LENGTH_2 * sin(s[0] + s[1]),
        ]

        xys = np.array([[0, 0], p1, p2])[:, ::-1]
        thetas = [s[0] - pi / 2, s[0] + s[1] - pi / 2]
        link_lengths = [self.LINK_LENGTH_1, self.LINK_LENGTH_2]

        self.viewer.draw_line((-2.2, 1), (2.2, 1))
        for ((x, y), th, llen) in zip(xys, thetas, link_lengths):
            l, r, t, b = 0, llen, 0.1, -0.1
            jtransform = rendering.Transform(rotation=th, translation=(x, y))
            link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
            link.add_attr(jtransform)
            link.set_color(0, 0.8, 0.8)
            circ = self.viewer.draw_circle(0.1)
            circ.set_color(0.8, 0.8, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")
