# Copyright 2023 The Deluca Authors.
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

"""Deep Actor Critic."""
import deluca.core
import flax.linen as nn
import jax
import jax.numpy as jnp

from deluca.lung.core import BreathWaveform
from deluca.lung.core import Controller
from deluca.lung.core import ControllerState
from deluca.lung.core import DEFAULT_DT
from deluca.lung.core import proper_time
from deluca.lung.controllers._expiratory import Expiratory
from deluca.lung.utils.data.transform import ShiftScaleTransform


# TODO(wenhanxia): modify arguments to config dict of model specification.
class ActorCritic(nn.Module):
  """Class defining the actor-critic model."""

  num_actions: int = 100
  chan1: int = 32
  chan2: int = 64
  dtype = jnp.float32

  @nn.compact
  def __call__(self, x):
    """Define the model architecture.

    Network is used to both estimate policy (logits) and expected state value;
    in other words, hidden layers' params are shared between policy and value
    networks.

    Args:
      x: input of shape N, H, W(1)

    Returns:
      policy_log_probabilities: logits
      value: state value
    """
    x = x.astype(self.dtype)
    x = nn.Conv(
        features=self.chan1,
        kernel_size=[3],
        strides=1,
        name='conv1',
        dtype=self.dtype)(
            x)
    x = nn.relu(x)
    x = nn.Conv(
        features=self.chan2,
        kernel_size=[3],
        strides=1,
        name='conv2',
        dtype=self.dtype)(
            x)
    x = nn.relu(x)

    x = x.reshape((x.shape[0], -1))  # flatten
    logits = nn.Dense(
        features=self.num_actions, name='logits', dtype=self.dtype)(
            x)
    policy_log_probabilities = nn.log_softmax(logits)
    value = nn.Dense(features=1, name='value', dtype=self.dtype)(x)
    return policy_log_probabilities, value


class DeepACControllerState(deluca.Obj):
  errs: jnp.ndarray
  key: int
  time: float = float('inf')
  steps: int = 0
  dt: float = DEFAULT_DT


class DeepAC(Controller):
  """ Class defining the conroller based on the actor critic model """

  params: list = deluca.field(jaxed=True)
  model: nn.module = deluca.field(ActorCritic, jaxed=False)
  featurizer: jnp.ndarray = deluca.field(jaxed=False)
  H: int = deluca.field(100, jaxed=False)
  input_dim: int = deluca.field(1, jaxed=False)
  history_len: int = deluca.field(10, jaxed=False)
  kernel_size: int = deluca.field(5, jaxed=False)
  clip: float = deluca.field(40.0, jaxed=False)
  normalize: bool = deluca.field(False, jaxed=False)
  u_scaler: ShiftScaleTransform = deluca.field(jaxed=False)
  p_scaler: ShiftScaleTransform = deluca.field(jaxed=False)
  waveform: deluca.Obj = deluca.field(
      jaxed=False)  # moving waveform here at the cost of generality

  # bptt: int = deluca.field(1, jaxed=False) not used right now
  # TODO: add analogue of activation=torch.nn.ReLU

  def setup(self, waveform=None):
    self.model = ActorCritic()
    if self.params is None:
      self.params = self.model.init(
          jax.random.PRNGKey(0),
          jnp.expand_dims(jnp.ones([self.history_len]), axis=(0, 1)))['params']

    # linear feature transform:
    # errs -> [average of last h errs, ..., average of last 2 errs, last err]
    # emulates low-pass filter bank

    self.featurizer = jnp.tril(jnp.ones((self.history_len, self.history_len)))
    self.featurizer /= jnp.expand_dims(
        jnp.arange(self.history_len, 0, -1), axis=0)

    if waveform is None:
      self.waveform = BreathWaveform.create()

    if self.normalize:
      self.u_scaler = u_scaler
      self.p_scaler = p_scaler

  # TODO(dsuo): Handle dataclass initialization of jax objects
  def init(self, key):
    errs = jnp.array([0.0] * self.history_len)
    state = DeepACControllerState(errs=errs, key=key)
    return state

  def derive_prob_and_value(self, cont_state):
    # expects a batch dimension
    # adds the spatial dimension
    errs = jax.tree_map(lambda x: jnp.expand_dims(x, axis=(1)), cont_state.errs)
    log_prob, value = self.model.apply({'params': self.params},
                                       (errs @ self.featurizer))
    return log_prob.squeeze(), value.squeeze()

  def log_prob_action(self, log_probs, action):
    return log_probs[action[0]]

  def __call__(self, controller_state, obs):
    state, t = obs.predicted_pressure, obs.time
    errs, waveform = controller_state.errs, self.waveform
    target = waveform.at(t)
    if self.normalize:
      target_scaled = self.p_scaler(target).squeeze()
      state_scaled = self.p_scaler(state).squeeze()
      next_errs = jnp.roll(errs, shift=-1)
      next_errs = next_errs.at[-1].set(target_scaled - state_scaled)
    else:
      next_errs = jnp.roll(errs, shift=-1)
      next_errs = next_errs.at[-1].set(target - state)

    current_key, next_key = jax.random.split(controller_state.key)
    ## adding a batch dimension ##
    next_errs_expanded = jnp.expand_dims(next_errs, axis=(0))

    controller_state_big = controller_state.replace(
        errs=next_errs_expanded, key=current_key)
    decay = self.decay(waveform, t)
    log_prob, value = self.derive_prob_and_value(controller_state_big)
    # value = value[0]
    prob = jnp.exp(log_prob)
    # environment step
    u_in = jax.random.choice(current_key, prob.shape[0], p=prob)

    # changed decay compare from None to float(inf) due to cond requirements
    ## TODO(@namanagarwal) - Need to do this
    # u_in, log_prob, value = jax.lax.cond(
    #     jnp.isinf(decay), true_func, lambda x: (jnp.array(decay), None, None), None)

    # TODO (@namanagarwal) : Figure out what to do with clamping
    # u_in = jax.lax.clamp(0.0, u_in.astype(jnp.float64), self.clip).squeeze()

    # update controller_state

    new_dt = jnp.max(
        jnp.array([DEFAULT_DT, t - proper_time(controller_state.time)]))
    new_time = t
    new_steps = controller_state.steps + 1
    controller_state = controller_state.replace(
        time=new_time, steps=new_steps, dt=new_dt, key=next_key, errs=next_errs)
    return controller_state, (u_in, 0), log_prob[u_in], value
