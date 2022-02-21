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

"""Acrobot."""
from deluca.core import Env
from deluca.core import field
import jax
import jax.numpy as jnp

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
  """Acrobot."""
  key: jnp.ndarray = field(jaxed=False)
  dt: float = field(0.2, jaxed=False)

  LINK_LENGTH_1: float = field(1.0, jaxed=False)
  LINK_LENGTH_2: float = field(1.0, jaxed=False)
  LINK_MASS_1: float = field(1.0, jaxed=False)
  LINK_MASS_2: float = field(1.0, jaxed=False)
  LINK_COM_POS_1: float = field(0.0, jaxed=False)
  LINK_COM_POS_2: float = field(0.0, jaxed=False)
  LINK_MOI: float = field(1.0)

  MAX_VEL_1: float = field(4 * jnp.pi, jaxed=False)
  MAX_VEL_2: float = field(9 * jnp.pi, jaxed=False)

  AVAIL_TORQUE: jnp.ndarray = field(jaxed=False)

  torque_noise_max: float = field(0.0, jaxed=False)

  high: jnp.ndarray = field(jaxed=False)
  low: jnp.ndarray = field(jaxed=False)

  def setup(self):
    self.high = jnp.array([1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2])
    self.low = -self.high
    if self.key is None:
      self.key = jax.random.PRNGKey(0)
    if self.AVAIL_TORQUE is None:
      self.AVAIL_TORQUE = jnp.array([-1.0, 0.0, +1])

  def init(self):
    # TODO(dsuo): to implement
    pass

  def __call__(self, state, action):
    augmented_state = jnp.append(state, action)

    new_state = rk4(self._dsdt, augmented_state, [0, self.dt])
    # only care about final timestep of integration returned by integrator
    new_state = new_state[-1]
    new_state = new_state[:4]  # omit action
    # ODEINT IS TOO SLOW!
    # ns_continuous = integrate.odeint(self._dsdt, self.s_continuous, [0, self.dt])
    # self.s_continuous = ns_continuous[-1] # We only care about the state
    # at the ''final timestep'', self.dt

    new_state = new_state.at[0].set(wrap(new_state[0], -jnp.pi, jnp.pi))
    new_state = new_state.at[1].set(wrap(new_state[1], -jnp.pi, jnp.pi))
    new_state = new_state.at[2].set(
        bound(new_state[2], -self.MAX_VEL_1, self.MAX_VEL_1))
    new_state = new_state.at[3].set(
        bound(new_state[3], -self.MAX_VEL_2, self.MAX_VEL_2))

    return (
        new_state,
        jnp.array([
            jnp.cos(new_state[0]),
            jnp.sin(new_state[0]),
            jnp.cos(new_state[1]),
            jnp.sin(new_state[1]),
            new_state[2],
            new_state[3],
        ]),
    )

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

    d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 +
                             2 * l1 * lc2 * jnp.cos(theta2)) + I1 + I2
    d2 = m2 * (lc2**2 + l1 * lc2 * jnp.cos(theta2)) + I2
    phi2 = m2 * lc2 * g * jnp.cos(theta1 + theta2 - jnp.pi / 2.0)
    phi1 = (-m2 * l1 * lc2 * dtheta2**2 * jnp.sin(theta2) -
            2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * jnp.sin(theta2) +
            (m1 * lc1 + m2 * l1) * g * jnp.cos(theta1 - jnp.pi / 2) + phi2)
    ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * lc2**2 + I2 - d2**2 / d1)
    # if self.book_or_nips == "nips":
    # the following line is consistent with the description in the
    # paper
    # ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
    # else:
    # the following line is consistent with the java implementation and the
    # book
    # ddtheta2 = (
    # a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * jnp.sin(theta2) - phi2
    # ) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
    return (dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0)  # 4-state version


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
        derivs: the derivative of the system and has the signature ``dy =
          derivs(yi, ti)``
        y0: initial state vector
        t: sample times
        args: additional arguments passed to the derivative function
        kwargs: additional keyword arguments passed to the derivative function
    Example 1 :: ## 2D system
        def derivs6(x,t): d1 =  x[0] + 2*x[1] d2 =  -3*x[0] + 4*x[1] return (d1,
          d2) dt = 0.0005 t = arange(0.0, 2.0, dt) y0 = (1,2) yout =
          rk4(derivs6, y0, t)
    Example 2:: ## 1D system alpha = 2
        def derivs(x,t): return -alpha*x + exp(-t) y0 = 1 yout = rk4(derivs, y0,
          t) If you have access to scipy, you should probably be using jnp.sing
          the scipy.integrate tools rather than this function.

    Returns:
        yout: Runge-Kutta approximation of the ODE
    """

  try:
    Ny = len(y0)
  except TypeError:
    yout = jnp.zeros((len(t),))
  else:
    yout = jnp.zeros((len(t), Ny))

  yout = yout.at[0].set(y0)

  for i in jnp.arange(len(t) - 1):

    thist = t[i]
    dt = t[i + 1] - thist
    dt2 = dt / 2.0
    y0 = yout[i]

    k1 = jnp.asarray(derivs(y0, thist, *args, **kwargs))
    k2 = jnp.asarray(derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
    k3 = jnp.asarray(derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
    k4 = jnp.asarray(derivs(y0 + dt * k3, thist + dt, *args, **kwargs))
    yout = yout.at[i + 1].set(y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4))

  return yout
