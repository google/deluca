"""Experiment script for comparing GPC and SFC models."""

import argparse
import functools
from typing import Tuple, Any
import importlib.util
import sys
sys.path.append("../../")
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jax import debug as jax_debug

from deluca.experimental.agents.agent import policy_loss, update_agentstate, AgentState
from deluca.experimental.agents.gpc import get_gpc_features
from deluca.experimental.agents.sfc import get_sfc_features
from deluca.experimental.agents.model import FullyConnectedModel, SequentialModel, GridModel
from deluca.experimental.enviornments.disturbances.sinusoidal import sinusoidal_disturbance
from deluca.experimental.enviornments.disturbances.gaussian import gaussian_disturbance
from deluca.experimental.enviornments.disturbances.zero import zero_disturbance
from deluca.experimental.enviornments.sim_and_output.lds import lds_sim, lds_output
from deluca.experimental.enviornments.sim_and_output.brax import brax_sim, brax_output
from deluca.experimental.enviornments.sim_and_output.pendulum import pendulum_sim, pendulum_output
from deluca.experimental.enviornments.cost_functions.quadratic_cost import quadratic_cost_evaluate
from deluca.experimental.enviornments.env import step

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

def run_trial(
    key: jax.random.PRNGKey,
    config: Any,
    # A: jnp.ndarray,
    # B: jnp.ndarray,
    # Q: jnp.ndarray,
    # R: jnp.ndarray,
) -> jnp.ndarray:
    """Run a single trial."""
    # Create simulator and cost function
    if config.sim_type == 'lds':
        key, A_B_key = jax.random.split(key)
        A, B = create_random_system(A_B_key, config.d, config.n, config.min_eig, config.max_eig)
        key, Q_R_key = jax.random.split(key)
        Q, R = create_random_cost_matrices(Q_R_key, config.d, config.n)
        # if config.debug:
        #     print(f'A.shape={A.shape}, B.shape={B.shape}, Q.shape={Q.shape}, R.shape={R.shape}')

        sim = functools.partial(lds_sim, A=A, B=B)
        output_map = functools.partial(lds_output, C=jnp.eye(config.d))  # Identity output map
        cost_fn = functools.partial(quadratic_cost_evaluate, Q=Q, R=R)
    
    elif config.sim_type == 'pendulum':
        sim = functools.partial(pendulum_sim, max_torque=config.max_torque, m=config.mass, l=config.length, g=config.g, dt=config.dt)
        output_map = pendulum_output

        Q = jnp.array([[1.0, 0.0], [0.0, 0.0]])
        R = jnp.zeros((config.n, config.n))
        cost_fn = functools.partial(quadratic_cost_evaluate, Q=Q, R=R)
    
    # Create disturbance
    disturbance_params = config.disturbance_params[config.disturbance_type]
    if config.disturbance_type == 'sinusoidal':
        disturbance = lambda d, dist_std, t, key: sinusoidal_disturbance(d=d, noise_amplitude=disturbance_params['noise_amplitude'], noise_frequency=disturbance_params['noise_frequency'] * jnp.ones(config.d), t=t, key=key)
    elif config.disturbance_type == 'gaussian':
        disturbance = lambda d, dist_std, t, key: gaussian_disturbance(d=d, dist_std=disturbance_params['std'], t=t, key=key)
    elif config.disturbance_type == 'zero':
        disturbance = lambda d, dist_std, t, key: zero_disturbance(d=d, key=key)
    else:
        raise ValueError(f"Unknown disturbance type: {config.disturbance_type}")
    
    #Create feature extractor
    if config.algorithm_type == 'gpc':
        get_features = get_gpc_features(m=config.m, d=config.d)
        input_dim = config.m
    elif config.algorithm_type == 'sfc':
        get_features = get_sfc_features(m=config.m, d=config.d, h=config.h, gamma=config.gamma)
        input_dim = config.h
    else:
        raise ValueError(f"Unknown algorithm type: {config.algorithm_type}")

    # Create model
    if config.model_type == 'fully_connected':
        model = FullyConnectedModel(m=input_dim, d=config.d, k=config.k, n=config.n, hidden_dims=config.hidden_dims)
    elif config.model_type == 'sequential':
        model = SequentialModel(m=input_dim, d=config.d, k=config.k, n=config.n, hidden_dims=config.hidden_dims)
    elif config.model_type == 'grid':
        model = GridModel(m=input_dim, d=config.d, k=config.k, n=config.n, hidden_dims=config.hidden_dims)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    # Create optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.R_M),
        optax.adam(config.learning_rate)
    )
    
    # Initialize model parameters and agent state
    init_key, loop_key = jax.random.split(key)
    params = model.init(init_key, jnp.zeros((input_dim, config.d, 1)))
    opt_state = optimizer.init(params)
    
    initial_agentstate = AgentState(
        controller_t=0,
        state=jnp.zeros((config.d, 1)),
        dist_history=jnp.zeros((2*config.m, config.d, 1)),
        state_history=jnp.zeros((config.m, config.d, 1)),
        params=params,
        opt_state=opt_state
    )
    # if config.debug:
    #     initial_agentstate.show()
    initial_physical_state = jnp.zeros((config.d, 1))

    # Define loss function for grad and for reporting
    def loss_fn(p, dist_history, start_state):
        return policy_loss(
            apply_fn=model.apply,
            params=p,
            d=config.d,
            m=config.m,
            dist_history=dist_history,
            start_state=start_state,
            sim=sim,
            cost_fn=cost_fn,
            get_features=get_features,
            debug=config.debug,
        )
    
    grad_fn = jax.grad(loss_fn)

    def scan_body(carry, t):
        agentstate, physical_state, key = carry
        # if config.debug:
        #     jax_debug.print('--- Step {} ---', t)
        #     jax_debug.print('agentstate.controller_t: {}', agentstate.controller_t)
        #     jax_debug.print('physical_state: {}', physical_state)
        # Get action from model
        actions = model.apply(agentstate.params, get_features(config.m, agentstate.dist_history))
        action = actions[0]
        # if config.debug:
        #     jax_debug.print('actions shape: {}', actions.shape)
        #     jax_debug.print('action: {}', action)
        # Step the environment
        key1, key2 = jax.random.split(key)
        next_physical_state, output = step(
            x=physical_state,
            u=action,
            sim=sim,
            output_map=output_map,
            dist_std=1.0,
            t_sim_step=t,
            disturbance=disturbance,
            key=key1
        )
        # if config.debug:
        #     jax_debug.print('next_physical_state: {}, output: {}', next_physical_state, output)
        # Update agent state
        agentstate = update_agentstate(
            agentstate=agentstate,
            next_state=output,
            action=action,
            sim=sim,
            grad_fn=grad_fn,
            optimizer=optimizer,
            debug=config.debug
        )
        # Update physical state for next iteration
        physical_state = next_physical_state
        # Compute loss for reporting
        loss = loss_fn(agentstate.params, agentstate.dist_history, agentstate.state_history[0])
        
        new_carry = (agentstate, next_physical_state, key2)
        return new_carry, loss

    # Run simulation with scan
    initial_carry = (initial_agentstate, initial_physical_state, loop_key)
    (_, _, _), losses = jax.lax.scan(scan_body, initial_carry, jnp.arange(config.num_steps))
    
    return losses

def main():
    parser = argparse.ArgumentParser(description='Run GPC/SFC experiments')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    # Load config file
    spec = importlib.util.spec_from_file_location("config", args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    
    # Set random seed
    key = jax.random.PRNGKey(config.seed)
    
    print(f"--- Starting {config.num_trials} Trials (vmapped) ---")
    
    # Create keys for each trial
    trial_keys = jax.random.split(key, config.num_trials)

    # Vmap the run_trial function
    vmapped_run_trial = jax.vmap(
        functools.partial(run_trial, config=config),
        in_axes=(0),
    )
    
    # JIT compile the vmapped function for performance
    jitted_vmapped_run_trial = jax.jit(vmapped_run_trial)
    
    print("Compiling and running trials... (this may take a moment)")
    all_losses = jitted_vmapped_run_trial(trial_keys)
    all_losses.block_until_ready() # Wait for all computations to finish
    print("--- All trials completed ---")

    # Compute mean and standard error
    mean_losses = jnp.mean(all_losses, axis=0)
    std_losses = jnp.std(all_losses, axis=0) / jnp.sqrt(config.num_trials)

    # Create results subdirectories
    results_dir = 'results'
    plots_dir = os.path.join(results_dir, 'plots')
    arrays_dir = os.path.join(results_dir, 'arrays')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(arrays_dir, exist_ok=True)

    # Save mean_losses and std_losses as .npy files in arrays_dir
    base_name = f"{config.sim_type}_{config.model_type}_{config.algorithm_type}"
    mean_path = os.path.join(arrays_dir, f"{base_name}_mean_losses.npy")
    std_path = os.path.join(arrays_dir, f"{base_name}_std_losses.npy")
    jnp.save(mean_path, mean_losses)
    jnp.save(std_path, std_losses)
    print(f"Mean losses saved to {mean_path}")
    print(f"Std losses saved to {std_path}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(mean_losses, label='Mean Loss')
    plt.fill_between(
        range(len(mean_losses)),
        mean_losses - std_losses,
        mean_losses + std_losses,
        alpha=0.2
    )
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(f'{config.sim_type} {config.model_type} {config.algorithm_type} Performance')
    plt.legend()
    plt.grid(True)
    
    # Save the figure in plots_dir
    filename = f"{base_name}_performance.png"
    filepath = os.path.join(plots_dir, filename)
    plt.savefig(filepath)
    print(f"Plot saved to {filepath}")

if __name__ == '__main__':
    main() 