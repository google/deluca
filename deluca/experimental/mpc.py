"""Experiment script for MPC control on a Pendulum environment."""

import argparse
import functools
from typing import Callable, Any, Tuple
import importlib.util
import sys
sys.path.append("../../")
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import linen as nn

from deluca.experimental.enviornments.disturbances.zero import zero_disturbance
from deluca.experimental.enviornments.sim_and_output.lds import lds_sim, lds_output
from deluca.experimental.enviornments.cost_functions.quadratic_cost import quadratic_cost_evaluate
from deluca.experimental.enviornments.sim_and_output.pendulum import pendulum_sim, pendulum_output
from deluca.experimental.enviornments.cost_functions.pendulum_cost import pendulum_cost_evaluate
from deluca.experimental.enviornments.env import step
from deluca.experimental.agents.mpc import MPCModel
from deluca.experimental.enviornments.disturbances.sinusoidal import sinusoidal_disturbance
from deluca.experimental.enviornments.disturbances.gaussian import gaussian_disturbance
from deluca.experimental.enviornments.disturbances.zero import zero_disturbance

# Type hint for a PRNGKey
PRNGKey = Any

def create_random_system(
    key: jax.random.PRNGKey,
    d: int,
    n: int,
    min_eig: float,
    max_eig: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Create random system matrices A and B.
    
    Args:
        key: Random key
        d: State dimension
        n: Input dimension
        min_eig: Minimum eigenvalue of A
        max_eig: Maximum eigenvalue of A
        
    Returns:
        A: State matrix of shape (d, d)
        B: Input matrix of shape (d, n)
    """
    key1, key2 = jax.random.split(key)
    
    # Create diagonal A matrix with eigenvalues between min_eig and max_eig
    A = jnp.diag(jax.random.uniform(key1, (d,), minval=min_eig, maxval=max_eig))
    
    # Create random B matrix
    B = jax.random.normal(key2, (d, n))
    
    return A, B

def create_random_cost_matrices(
    key: jax.random.PRNGKey,
    d: int,
    n: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Create random positive semidefinite cost matrices Q and R.
    
    Args:
        key: Random key
        d: State dimension
        n: Input dimension
        
    Returns:
        Q: State cost matrix of shape (d, d)
        R: Input cost matrix of shape (n, n)
    """
    # Create random positive semidefinite matrices
    key1, key2 = jax.random.split(key)
    Q = jax.random.normal(key1, (d, d))
    R = jax.random.normal(key2, (n, n))
    
    # Make them positive semidefinite
    Q = Q.T @ Q
    R = R.T @ R
    
    return Q, R

def rollout(
    apply_fn: Callable,
    params: Any,
    sim: Callable,
    cost_fn: Callable,
    k: int,
    initial_state: jnp.ndarray
) -> float:
    """
    Simulates a k-step rollout and computes the total cost.
    """
    actions = apply_fn(params, initial_state)

    def rollout_step(state, action):
        next_state = sim(state, action)
        cost = cost_fn(next_state, action)
        return next_state, cost

    _, costs = jax.lax.scan(rollout_step, initial_state, actions)
    return jnp.sum(costs)

def run_trial(
    key: PRNGKey,
    model: nn.Module,
    optimizer: Any,
    sim: Callable,
    output_map: Callable,
    cost_fn: Callable,
    disturbance: Callable,
    d: int,
    n: int,
    k: int,
    num_steps: int,
    gradient_updates_per_step: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Run a single trial for the pendulum MPC experiment."""
    # Initialize model parameters and optimizer state
    init_key, loop_key = jax.random.split(key)
    params = model.init(init_key, jnp.zeros((d, 1)))
    opt_state = optimizer.init(params)

    # Start the pendulum with a slight angle to encourage learning
    initial_physical_state = jnp.zeros((d, 1)).at[0].set(0.15)

    def loss_fn(p, state):
        return rollout(
            apply_fn=model.apply,
            params=p,
            sim=sim,
            cost_fn=cost_fn,
            k=k,
            initial_state=state
        )

    grad_fn = jax.grad(loss_fn)

    def scan_body(carry, t):
        params, opt_state, physical_state, key = carry

        def update_step(i, val):
            p, o = val
            grads = grad_fn(p, physical_state)
            updates, o = optimizer.update(grads, o, p)
            p = optax.apply_updates(p, updates)
            return p, o

        params, opt_state = jax.lax.fori_loop(0, gradient_updates_per_step, update_step, (params, opt_state))

        # Get action from the updated model
        actions = model.apply(params, physical_state)
        action = actions[0]

        # Step the environment with zero disturbance
        key, step_key = jax.random.split(key)
        next_physical_state, output, _ = step(
            x=physical_state,
            u=action,
            sim=sim,
            output_map=output_map,
            dist_std=0.0,
            t_sim_step=t,
            disturbance=disturbance,
            key=step_key,
        )

        # Compute loss for reporting (cost of the single step taken)
        loss = cost_fn(output, action)

        new_carry = (params, opt_state, next_physical_state, key)
        return new_carry, (loss, next_physical_state)

    # Run simulation with scan
    initial_carry = (params, opt_state, initial_physical_state, loop_key)
    (_, _, _, _), (losses, states) = jax.lax.scan(
        scan_body, initial_carry, jnp.arange(num_steps)
    )

    return losses, states

def main():
    parser = argparse.ArgumentParser(description="Run MPC experiment")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config file"
    )
    args = parser.parse_args()

    # Load config file
    spec = importlib.util.spec_from_file_location("config", args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # Determine results directory
    base_results_dir = "mpc_results"
    if config.hidden_dims is None:
        model_dir_name = "linear_model"
    else:
        model_dir_name = "neural_net_model"
    results_dir = os.path.join(base_results_dir, model_dir_name)
    os.makedirs(results_dir, exist_ok=True)

    # Set random seed
    key = jax.random.PRNGKey(config.seed)

    # Create simulation and cost functions
    sim = functools.partial(pendulum_sim, max_torque=config.max_torque, m=config.m, l=config.l, g=config.g, dt=config.dt)
    # key, A_B_key = jax.random.split(key)
    # A, B = create_random_system(A_B_key, config.d, config.n, config.min_eig, config.max_eig)
    # key, Q_R_key = jax.random.split(key)
    # Q, R = create_random_cost_matrices(Q_R_key, config.d, config.n)
    # sim = functools.partial(lds_sim, A=A, B=B)
    # output_map = functools.partial(lds_output, C=jnp.eye(config.d))  # Identity output map
    # cost_fn = functools.partial(quadratic_cost_evaluate, Q=Q, R=R)
    # Create disturbance
    disturbance_params = config.disturbance_params[config.disturbance_type]
    if config.disturbance_type == 'sinusoidal':
        disturbance = lambda d, dist_std, t, key: sinusoidal_disturbance(d=d, noise_amplitude=disturbance_params['noise_amplitude'], noise_frequency=disturbance_params['noise_frequency'] * jnp.ones((config.d,1)), t=t, key=key)
    elif config.disturbance_type == 'gaussian':
        disturbance = lambda d, dist_std, t, key: gaussian_disturbance(d=d, dist_std=disturbance_params['std'], t=t, key=key)
    elif config.disturbance_type == 'zero':
        disturbance = lambda d, dist_std, t, key: zero_disturbance(d=d, dist_std=dist_std, t=t, key=key)
    else:
        raise ValueError(f"Unknown disturbance type: {config.disturbance_type}")
    output_map = pendulum_output
    cost_fn = pendulum_cost_evaluate

    # Create optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.R_M), optax.adam(config.learning_rate)
    )

    # --- Create MPC Model ---
    print("--- Creating MPC Model ---")
    mpc_model = MPCModel(
        d=config.d, n=config.n, k=config.k, hidden_dims=config.hidden_dims
    )

    # --- Run Trials ---
    print(f"--- Starting {config.num_trials} MPC Trials (vmapped) ---")
    vmapped_run_trial = jax.vmap(
        functools.partial(
            run_trial,
            model=mpc_model,
            optimizer=optimizer,
            sim=sim,
            output_map=output_map,
            cost_fn=cost_fn,
            disturbance=disturbance,
            d=config.d,
            n=config.n,
            k=config.k,
            num_steps=config.num_steps,
            gradient_updates_per_step=config.gradient_updates_per_step,
        ),
        in_axes=(0),
    )
    jitted_run_trial = jax.jit(vmapped_run_trial)
    print("Compiling and running MPC trials...")
    trial_keys = jax.random.split(key, config.num_trials)
    losses, states = jitted_run_trial(trial_keys)
    losses.block_until_ready()
    states.block_until_ready()
    print("--- MPC trials completed ---")

    # Create arrays subdirectory if it doesn't exist
    arrays_dir = os.path.join(results_dir, 'arrays')
    os.makedirs(arrays_dir, exist_ok=True)

    # Save losses and states data
    losses_path = os.path.join(arrays_dir, "mpc_losses.npy")
    states_path = os.path.join(arrays_dir, "mpc_states.npy")
    jnp.save(losses_path, losses)
    jnp.save(states_path, states)
    print(f"MPC losses data saved to {losses_path}")
    print(f"MPC states data saved to {states_path}")

    # --- Process and Plot Results ---
    mean_losses = jnp.mean(losses, axis=0)
    std_losses = jnp.std(losses, axis=0) / jnp.sqrt(config.num_trials)

    plt.figure(figsize=(10, 6))
    plt.plot(mean_losses, label="MPC Mean Loss")
    plt.fill_between(
        range(len(mean_losses)),
        mean_losses - std_losses,
        mean_losses + std_losses,
        alpha=0.2,
    )
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("MPC Loss")
    plt.legend()
    plt.grid(True)

    # Save the figure
    filename = "MPC_loss.png"
    filepath = os.path.join(results_dir, filename)
    plt.savefig(filepath)
    print(f"Plot saved to {filepath}")

    # --- Plot State Trajectories ---
    mean_states = jnp.mean(states, axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Theta plot
    ax1.plot(mean_states[:, 0, 0], label="Mean Theta", color="blue")
    ax1.set_ylabel('Theta (rad)')
    ax1.set_title("Pendulum State Over Time")
    ax1.legend()
    ax1.grid(True)

    # Theta_dot plot
    ax2.plot(mean_states[:, 1, 0], label="Mean Theta Dot", color="blue", linestyle='--')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Theta Dot (rad/s)')
    ax2.legend()
    ax2.grid(True)

    # Save the state plot
    state_filename = "pendulum_state.png"
    state_filepath = os.path.join(results_dir, state_filename)
    plt.savefig(state_filepath)
    print(f"State plot saved to {state_filepath}")


if __name__ == "__main__":
    main() 