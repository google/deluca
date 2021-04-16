# Copyright 2021 Google LLC
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
"""deluca/tests/utils/test_experiment.py"""
import jax
import numpy as np
import pytest

from deluca.utils import Random
from deluca.utils.experiment import experiment

random = Random()


@pytest.mark.parametrize("shape", [(), (1,), (1, 2), (1, 2, 3)])
@pytest.mark.parametrize("num_args", [1, 2, 10])
def test_experiment(shape, num_args):
    """Test normal experiment behavior"""
    args = [
        (
            jax.random.uniform(random.generate_key(), shape=shape),
            jax.random.uniform(random.generate_key(), shape=shape),
        )
        for _ in range(num_args)
    ]

    @experiment("a,b", args)
    def dummy(a, b):
        """dummy"""
        return a + b

    results = dummy.run()
    print(results)
    for i in range(len(results)):
        np.testing.assert_array_almost_equal(results[i], np.sum(args[i], axis=0))


@pytest.mark.parametrize("times", [1, 2, 10])
def test_experiment_call(times):
    """Tests repeated decorator calls"""

    def dummy(a, b):
        """dummy"""
        return a + b

    for _ in range(times):
        dummy = experiment("a,b", [(1, 2)])(dummy)
        dummy._func(1, 2)
        assert isinstance(dummy, experiment)


def test_experiment_duplicated_argname():
    """Tests duplicated arguments"""
    with pytest.raises(ValueError):

        @experiment("a", [1])
        @experiment("a,b", [(1, 2)])
        def dummy(a, b):
            """dummy"""
            return a + b

        dummy._func(1, 2)
        dummy._validate()


def test_experiment_missing_argument():
    """Test missing arguments"""
    with pytest.raises(ValueError):

        @experiment("a", [1])
        def dummy(a, b):
            """dummy"""
            return a + b

        dummy._func(1, 2)
        dummy._validate()


def test_experiment_unused_arguments():
    """Testing unused arguments"""
    with pytest.raises(ValueError):

        @experiment("a,b,c", [(1, 2, 3)])
        def dummy(a, b):
            """dummy"""
            return a + b

        dummy._func(1, 2)
        dummy._validate()


def test_experiment_list_args():
    """Testing args as list"""

    @experiment(["a", "b"], [(1, 2)])
    def dummy(a, b):
        """dummmy"""
        return a + b

    dummy.run()


def test_experiment_list_atoms():
    """Testing atoms"""

    @experiment(["a"], [1])
    def dummy(a):
        """dummy"""
        return a

    assert 1 == dummy.run()[0]
