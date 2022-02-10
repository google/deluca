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

"""learned_lung."""
import functools
from typing import Any, Dict

import deluca.core
from deluca.lung.core import LungEnv
from deluca.lung.utils.data.transform import ShiftScaleTransform
from deluca.lung.utils.nn import MLP
from deluca.lung.utils.nn import ShallowBoundaryModel
import flax.linen as nn
import jax
import jax.numpy as jnp

# pylint: disable=unused-argument
# pylint: disable=g-bare-generic
# pylint: disable=g-complex-comprehension


class LearnedLungObservation(deluca.Obj):
  predicted_pressure: float = 0.0
  time: float = 0.0


# Note: can't inherit from core because jax arrays not allowed to be manipulated
# as default values, and non-default args can't follow inherited default args
class SimulatorState(deluca.Obj):
  u_history: jnp.ndarray
  p_history: jnp.ndarray
  t_in: int = 0
  steps: int = 0
  predicted_pressure: float = 0.0


def call_if_t_in_positive(state, reset):
  new_state, _ = reset()
  return new_state


def true_func(state, reset):
  """True branch of jax.lax.cond in call function."""
  state = jax.lax.cond(state.t_in > 0,
                       functools.partial(call_if_t_in_positive, reset=reset),
                       lambda x: x, state)
  return state


def false_func(state, u_in, model_idx, u_normalizer, update_history,
               u_history_len, p_history_len, u_window, p_window,
               sync_unnormalized_pressure, num_boundary_models, funcs):
  """False branch of jax.lax.cond in call function."""
  u_in_normalized = u_normalizer(u_in).squeeze()
  state = state.replace(
      u_history=update_history(state.u_history, u_in_normalized),
      t_in=state.t_in + 1)
  boundary_features = jnp.hstack(
      [state.u_history[-u_history_len:], state.p_history[-p_history_len:]])
  default_features = jnp.hstack(
      [state.u_history[-u_window:], state.p_history[-p_window:]])
  default_pad_len = (u_history_len + p_history_len) - (u_window + p_window)
  default_features = jnp.hstack(
      [jnp.zeros((default_pad_len,)), default_features])
  features = jax.lax.cond(
      model_idx == num_boundary_models,
      lambda x: default_features,
      lambda x: boundary_features,
      None,
  )
  normalized_pressure = jax.lax.switch(model_idx, funcs, features)
  normalized_pressure = normalized_pressure.astype(jnp.float64)
  state = state.replace(predicted_pressure=normalized_pressure)
  new_p_history = update_history(state.p_history, normalized_pressure)
  state = state.replace(p_history=new_p_history
                       )  # just for inference. This is undone during training
  state = sync_unnormalized_pressure(state)  # unscale predicted pressure
  return state


class LearnedLung(LungEnv):
  """Learned lung."""
  params: list = deluca.field(jaxed=True)
  init_rng: jnp.array = deluca.field(jaxed=False)
  u_window: int = deluca.field(5, jaxed=False)
  p_window: int = deluca.field(3, jaxed=False)
  u_history_len: int = deluca.field(5, jaxed=False)
  p_history_len: int = deluca.field(5, jaxed=False)
  u_normalizer: ShiftScaleTransform = deluca.field(jaxed=False)
  p_normalizer: ShiftScaleTransform = deluca.field(jaxed=False)
  seed: int = deluca.field(0, jaxed=False)
  flow: int = deluca.field(0, jaxed=False)
  transition_threshold: int = deluca.field(0, jaxed=False)

  default_model_parameters: Dict[str, Any] = deluca.field(jaxed=False)
  num_boundary_models: int = deluca.field(5, jaxed=False)
  boundary_out_dim: int = deluca.field(1, jaxed=False)
  boundary_hidden_dim: int = deluca.field(100, jaxed=False)
  reset_normalized_peep: float = deluca.field(0.0, jaxed=False)

  default_model_name: str = deluca.field("MLP", jaxed=False)
  default_model: nn.module = deluca.field(jaxed=False)
  boundary_models: list = deluca.field(default_factory=list, jaxed=False)
  ensemble_models: list = deluca.field(default_factory=list, jaxed=False)

  def setup(self):
    self.u_history_len = max(self.u_window, self.num_boundary_models)
    self.p_history_len = max(self.p_window, self.num_boundary_models)
    self.default_model = MLP(
        hidden_dim=self.default_model_parameters["hidden_dim"],
        out_dim=self.default_model_parameters["out_dim"],
        n_layers=self.default_model_parameters["n_layers"],
        droprate=self.default_model_parameters["droprate"],
        activation_fn=self.default_model_parameters["activation_fn"])
    default_params = self.default_model.init(
        jax.random.PRNGKey(0),
        jnp.ones([self.u_history_len + self.p_history_len]))["params"]
    self.boundary_models = [
        ShallowBoundaryModel(
            out_dim=self.boundary_out_dim,
            hidden_dim=self.boundary_hidden_dim,
            model_num=i + 1) for i in range(self.num_boundary_models)
    ]
    boundary_params = [
        self.boundary_models[i].init(
            jax.random.PRNGKey(0),
            jnp.ones([self.u_history_len + self.p_history_len]))["params"]
        for i in range(self.num_boundary_models)
    ]

    self.ensemble_models = self.boundary_models + [self.default_model]
    if self.params is None:
      self.params = boundary_params + [default_params]
    print("TREE MAP")
    print(jax.tree_map(lambda x: x.shape, self.params))

  def init(self, key):
    return self.reset()


  def reset(self):
    normalized_peep = self.reset_normalized_peep
    state = SimulatorState(
        u_history=jnp.zeros([self.u_history_len]),
        p_history=jnp.hstack(
            [jnp.zeros([self.p_history_len - 1]),
             jnp.array([normalized_peep])]),
        t_in=0,
        steps=0,
        predicted_pressure=normalized_peep,
    )
    state = self.sync_unnormalized_pressure(state)
    obs = LearnedLungObservation(
        predicted_pressure=state.predicted_pressure, time=self.time(state))
    return state, obs

  def update_history(self, history, value):
    history = jnp.roll(history, shift=-1)
    history = history.at[-1].set(value)
    return history

  def sync_unnormalized_pressure(self, state):
    normalized_pressure = self.p_normalizer.inverse(
        state.predicted_pressure).squeeze()
    return state.replace(predicted_pressure=normalized_pressure)

  def __call__(self, state, action):
    u_in, u_out = action
    model_idx = jnp.min(jnp.array([state.t_in, self.num_boundary_models]))
    funcs = [
        functools.partial(self.ensemble_models[i].apply,
                          {"params": self.params[i]})
        for i in range(self.num_boundary_models + 1)
    ]
    # partial_false_func = functools.partial(
    #     false_func, u_in=u_in, model_idx=model_idx, funcs=funcs)
    partial_false_func = functools.partial(
        false_func,
        u_in=u_in,
        model_idx=model_idx,
        u_normalizer=self.u_normalizer,
        update_history=self.update_history,
        u_history_len=self.u_history_len,
        p_history_len=self.p_history_len,
        u_window=self.u_window,
        p_window=self.p_window,
        sync_unnormalized_pressure=self.sync_unnormalized_pressure,
        num_boundary_models=self.num_boundary_models,
        funcs=funcs)
    partial_true_func = functools.partial(true_func, reset=self.reset)
    state = jax.lax.cond(u_out == 1, partial_true_func, partial_false_func,
                         state)
    state = state.replace(steps=state.steps + 1)
    obs = LearnedLungObservation(
        predicted_pressure=state.predicted_pressure, time=self.time(state))
    return state, obs
