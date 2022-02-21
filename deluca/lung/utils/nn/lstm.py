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

"""lstm module."""
import functools
from typing import Callable
import flax.linen as nn
import jax
import jax.numpy as jnp


# pylint: disable=g-bare-generic


class LSTM(nn.Module):
  """LSTM encoder, returning state after EOS is input."""
  n_layers: int = 2  # or 4
  hidden_dim: int = 20
  out_dim: int = 1
  bptt: int = 3  # or 10 or 29
  activation_fn: Callable = nn.tanh

  @nn.compact
  def __call__(self, inputs):
    # inputs.shape = (batch_size, seq_length, vocab_size).
    batch_size = (inputs.shape[0],)
    lstm = LSTMNetwork(
        n_layers=self.n_layers, activation_fn=self.activation_fn)
    init_lstm_state = lstm.initialize_carry(batch_size, self.hidden_dim)
    init_lstm_state = (init_lstm_state[0].astype(jnp.float64),
                       init_lstm_state[1].astype(jnp.float64))
    _, y = lstm(init_lstm_state, inputs)
    y = nn.Dense(features=self.out_dim, use_bias=True, name='lstm_fc')(y)
    return y[0, -1, 0]  # squeeze for consistent shape w/ boundary model output


class LSTMNetwork(nn.Module):
  """A simple unidirectional LSTM."""
  n_layers: int = 2  # or 4
  activation_fn: Callable = nn.tanh

  @functools.partial(
      nn.transforms.scan,
      variable_broadcast='params',
      in_axes=1,
      out_axes=1,
      split_rngs={'params': False})
  @nn.compact
  def __call__(self, carry, x):
    for _ in range(self.n_layers):
      carry, x = nn.OptimizedLSTMCell(activation_fn=self.activation_fn)(carry,
                                                                        x)
    return carry, x

  @staticmethod
  def initialize_carry(batch_dims, hidden_dim):
    # (N, C, H, W) = (N, 1, 1, W) where W = feature_dim and N is batch_size
    # Use fixed random key since default state init fn is just zeros.
    return nn.OptimizedLSTMCell.initialize_carry(
        jax.random.PRNGKey(0), batch_dims, hidden_dim)
