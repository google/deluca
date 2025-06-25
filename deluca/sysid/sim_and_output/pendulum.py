"""Pendulum simulation and output functions.

This module provides JIT-compiled functions for simulating a pendulum system
and computing its outputs. The pendulum follows the standard equations of motion:
    θ̈ = (-3g/2l)sin(θ + π) + (3/ml²)u
where θ is the angle, g is gravity, l is length, m is mass, and u is the control input.
The control input is bounded by max_torque using a tanh activation.
"""

from jax import jit
import jax.numpy as jnp

@jit
def pendulum_sim(x: jnp.ndarray, u: jnp.ndarray, max_torque: float, m: float, l: float, g: float, dt: float) -> jnp.ndarray:
    theta, thdot = x
    action = max_torque * jnp.tanh(u[0])
    newth = theta + thdot * dt
    newthdot = thdot + ((3.0 * action) / (m * (l**2)) + (3.0 * g * jnp.sin(theta)) / (2.0 * l)) * dt
    return jnp.array([newth, newthdot])

@jit
def pendulum_output(x: jnp.ndarray) -> jnp.ndarray:
    return x