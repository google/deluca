from deluca.core import Agent
from deluca.agents._gpc import GPC
import matplotlib.pyplot as plt
import numpy as np

from numbers import Real
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad
from jax import jit

from deluca.core import Agent
from deluca.core import Env


def quad_loss(x: jnp.ndarray, u: jnp.ndarray) -> Real:
    """
    Quadratic loss.

    Args:
        x (jnp.ndarray):
        u (jnp.ndarray):

    Returns:
        Real
    """
    return jnp.sum(x.T @ x + u.T @ u)


class GPC(Agent):
    def __init__(
        self,
        Simulator: Env,
        d_state: int,
        d_action: int,
        start_time: int = 0,
        cost_fn: Callable[[jnp.ndarray, jnp.ndarray], Real] = None,
        H: int = 3,
        HH: int = 2,
        lr_scale: Real = 0.005,
        decay: bool = True,
    ) -> None:
        """
        Description: Initialize the dynamics of the model.

        Args:
            Simulator: a simulator for the environment
            start_time (int):
            cost_fn (Callable[[jnp.ndarray, jnp.ndarray], Real]):
            H (postive int): history of the controller
            HH (positive int): history of the system
            lr_scale (Real):
            lr_scale_decay (Real):
            decay (Real):
        """

        self.Simulator = Simulator

        cost_fn = cost_fn or quad_loss

        self.t = 0  # Time Counter (for decaying learning rate)

        self.H, self.HH = H, HH

        self.lr_scale, self.decay = lr_scale, decay

        self.bias = 0

        # Model Parameters
        # initial linear policy / perturbation contributions / bias
        # TODO: need to address problem of LQR with jax.lax.scan


        self.M = jnp.zeros((H, d_action, d_state))

        # Past H + HH noises ordered increasing in time
        self.noise_history = jnp.zeros((H + HH, d_state, 1))

        # past state and past action
        self.state, self.action = jnp.zeros((d_state, 1)), jnp.zeros((d_action, 1))

        def last_h_noises():
            """Get noise history"""
            return jax.lax.dynamic_slice_in_dim(self.noise_history, -H, H)

        self.last_h_noises = last_h_noises

        def policy_loss(M, w):
            """Surrogate cost function"""

            def action(state, h):
                """Action function"""
                return jnp.tensordot(
                    M, jax.lax.dynamic_slice_in_dim(w, h, H), axes=([0, 2], [0, 1])
                )

            def evolve(state, h):
                """Evolve function"""
                return self.Simulator( action(state,h) )  + w[h + H], None

            final_state, _ = jax.lax.scan(evolve, np.zeros((d_state, 1)), np.arange(H - 1))
            return cost_fn(final_state, action(final_state, HH - 1))

        self.policy_loss = policy_loss
        self.grad = jit(grad(policy_loss, (0, 1)))

    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Description: Return the action based on current state and internal parameters.

        Args:
            state (jnp.ndarray): current state

        Returns:
           jnp.ndarray: action to take
        """

        action = self.get_action(state)
        self.update(state, action)
        return action

    def update(self, state: jnp.ndarray, u: jnp.ndarray) -> None:
        """
        Description: update agent internal state.

        Args:
            state (jnp.ndarray):

        Returns:
            None
        """
        noise = state - self.Simulator(u)
        self.noise_history = self.noise_history.at[0].set(noise)
        self.noise_history = jnp.roll(self.noise_history, -1, axis=0)

        delta_M, delta_bias = self.grad(self.M, self.noise_history)

        lr = self.lr_scale
        lr *= (1 / (self.t + 1)) if self.decay else 1
        self.M -= lr * delta_M
        self.bias -= lr * delta_bias

        # update state
        self.state = state

        self.t += 1

    def get_action(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Description: get action from state.

        Args:
            state (jnp.ndarray):

        Returns:
            jnp.ndarray
        """
        return jnp.tensordot(self.M, self.last_h_noises(), axes=([0, 2], [0, 1]))



def plot_lds_with_a_given_agent(env, agent, name_of_agent, traj_length=1000, window_size=50, ax=None):
    obs = np.zeros(shape=(d_obs, 1))
    action = agent(obs)
    losses = np.zeros(traj_length)
    window_losses = np.zeros(traj_length - window_size)
    for i in range(traj_length):
        obs = env(action)
        action = agent(obs)
        losses[i] = np.linalg.norm(obs) + np.linalg.norm(action) # simple quadratic loss
        agent.update(obs, action)
        if i >= window_size - 1:
            window_losses[i - window_size] = np.mean(losses[i - window_size + 1: i + 1])
    ax.plot(window_losses)
    ax.set_xlabel('Time step (sliding window)')
    ax.set_ylabel('Loss')
    ax.set_title(f"Loss of {name_of_agent} on LDS")
env = lds.LDS()  # LDS with random initiations and dimensions as specified
d_action = 3
d_hidden = 10
d_obs = d_hidden
disturbance = SinusDisturbance()
disturbance.init(d_hidden, 0.5)
obs = env.init(d_action,d_hidden,d_obs, disturbance = disturbance)
fig, axes = plt.subplots(1, 2, figsize=(12, 8))
axes = axes.flatten()
agent = GPC(env, d_hidden, d_action)
agent.init()
plot_lds_with_a_given_agent(env, agent, "GPC", ax=axes[0])
plt.tight_layout()
plt.show()