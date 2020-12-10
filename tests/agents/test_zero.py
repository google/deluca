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
"""tests.agents.test_zero"""
import pytest

from deluca.agents import Zero


@pytest.mark.parametrize("shape", [None, 1, {}, (1, None), [2, ""]])
def test_bad_shapes(shape):
    """Test bad shapes"""
    with pytest.raises(ValueError):
        Zero(shape)


@pytest.mark.parametrize("shape", [[1, 2, 3], (4, 3, 2), (), []])
def test_zero(shape):
    """Test normal behavior"""
    agent = Zero(shape)

    assert agent.shape == tuple(shape)
