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

"""Shallow boundary module for first few time steps."""
import flax.linen as nn


class ShallowBoundaryModel(nn.Module):
  """Shallow boundary module."""
  out_dim: int = 1
  hidden_dim: int = 100
  model_num: int = 0

  @nn.compact
  def __call__(self, x):
    # need to flatten extra dimensions required by CNN and LSTM
    x = x.squeeze()
    x = nn.Dense(
        features=self.hidden_dim,
        use_bias=False,
        name=f"shallow_fc{1}_model" + str(self.model_num),
    )(
        x)
    x = nn.tanh(x)
    x = nn.Dense(
        features=self.out_dim,
        use_bias=True,
        name=f"shallow_fc{2}_model" + str(self.model_num))(
            x)
    return x.squeeze()  # squeeze for consistent shape w/ boundary model output
