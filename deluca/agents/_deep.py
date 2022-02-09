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

"""deluca.agents._deep"""
from numbers import Real

import jax
import jax.numpy as jnp

from deluca.agents.core import Agent
from deluca.utils import Random


class Deep(Agent):
    """
    Generic deep controller that uses zero-order methods to train on an
    environment.
    """

    def __init__(
        self,
        env_state_size,
        action_space,
        learning_rate: Real = 0.001,
        gamma: Real = 0.99,
        max_episode_length: int = 500,
        seed: int = 0,
    ) -> None:
        """
        Description: initializes the Deep agent

        Args:
            env (Env): a deluca environment
            learning_rate (Real):
            gamma (Real):
            max_episode_length (int):
            seed (int):

        Returns:
            None
        """
        # Create gym and seed numpy
        self.env_state_size = int(env_state_size)
        self.action_space = action_space
        self.max_episode_length = max_episode_length
        self.lr = learning_rate
        self.gamma = gamma

        self.random = Random(seed)
        self.reset()

    def reset(self) -> None:
        """
        Description: reset agent

        Args:
            None

        Returns:
            None
        """
        # Init weight
        self.W = jax.random.uniform(
            self.random.generate_key(),
            shape=(self.env_state_size, len(self.action_space)),
            minval=0,
            maxval=1,
        )

        # Keep stats for final print of graph
        self.episode_rewards = []

        self.current_episode_length = 0
        self.current_episode_reward = 0
        self.episode_rewards = jnp.zeros(self.max_episode_length)
        self.episode_grads = jnp.zeros((self.max_episode_length, self.W.shape[0], self.W.shape[1]))

        # dummy values for attrs, needed to inform scan of traced shapes
        self.state = jnp.zeros((self.env_state_size,))
        self.action = self.action_space[0]
        ones = jnp.ones((len(self.action_space),))
        self.probs = ones * 1 / jnp.sum(ones)

    def policy(self, state: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        """
        Description: Policy that maps state to action parameterized by w

        Args:
            state (jnp.ndarray):
            w (jnp.ndarray):
        """
        z = jnp.dot(state, w)
        exp = jnp.exp(z)
        return exp / jnp.sum(exp)

    def softmax_grad(self, softmax: jnp.ndarray) -> jnp.ndarray:
        """
        Description: Vectorized softmax Jacobian

        Args:
            softmax (jnp.ndarray)
        """
        s = softmax.reshape(-1, 1)
        return jnp.diagflat(s) - jnp.dot(s, s.T)

    def __call__(self, state: jnp.ndarray):
        """
        Description: provide an action given a state

        Args:
            state (jnp.ndarray):

        Returns:
            jnp.ndarray: action to take
        """
        self.state = state
        self.probs = self.policy(state, self.W)
        self.action = jax.random.choice(
            self.random.generate_key(), a=self.action_space, p=self.probs
        )

        return self.action

    def feed(self, reward: Real) -> None:
        """
        Description: compute gradient and save with reward in memory for weight updates

        Args:
            reward (Real):

        Returns:
            None
        """
        dsoftmax = self.softmax_grad(self.probs)[self.action, :]
        dlog = dsoftmax / self.probs[self.action]
        grad = self.state.reshape(-1, 1) @ dlog.reshape(1, -1)

        self.episode_rewards = self.episode_rewards.at[self.current_episode_length].set(reward)
        self.episode_grads = self.episode_grads.at[self.current_episode_length].set(grad)
        self.current_episode_length += 1

    def update(self) -> None:
        """
        Description: update weights
        
        Args:
            None

        Returns:
            None
        """
        for i in range(self.current_episode_length):
            # Loop through everything that happend in the episode and update
            # towards the log policy gradient times **FUTURE** reward
            self.W += self.lr * self.episode_grads[i] + jnp.sum(
                jnp.array(
                    [
                        r * (self.gamma ** r)
                        for r in self.episode_rewards[i : self.current_episode_length]
                    ]
                )
            )

        # reset episode length
        self.current_episode_length = 0
