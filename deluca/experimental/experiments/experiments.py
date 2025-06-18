"""Experiment script for comparing GPC and SFC models."""

import argparse
import functools
from typing import Callable, Tuple, Any, Optional, Sequence
import importlib.util
import sys
sys.path.append("../../")
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import linen as nn

from deluca.experimental.agents.agent import policy_loss, update_agentstate, AgentState
from deluca.experimental.agents.gpc import GPCModel
from deluca.experimental.agents.sfc import SFCModel
from deluca.experimental.enviornments.disturbances.sinusoidal import sinusoidal_disturbance
from deluca.experimental.enviornments.disturbances.gaussian import gaussian_disturbance
from deluca.experimental.enviornments.disturbances.zero import zero_disturbance
from deluca.experimental.enviornments.sim_and_output.lds import lds_sim, lds_output
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
    A: jnp.ndarray,
    B: jnp.ndarray,
    Q: jnp.ndarray,
    R: jnp.ndarray,
    model: nn.Module,
    optimizer: Any,
    disturbance: Callable,
    d: int,
    m: int,
    num_steps: int,
) -> jnp.ndarray:
    """Run a single trial."""
    # Create simulator and cost function
    sim = functools.partial(lds_sim, A=A, B=B)
    output_map = functools.partial(lds_output, C=jnp.eye(d))  # Identity output map
    cost_fn = functools.partial(quadratic_cost_evaluate, Q=Q, R=R)

    # Initialize model parameters and agent state
    init_key, loop_key = jax.random.split(key)
    params = model.init(init_key, jnp.zeros((m, d, 1)))
    opt_state = optimizer.init(params)

    initial_agentstate = AgentState(
        controller_t=0,
        state=jnp.zeros((d, 1)),
        dist_history=jnp.zeros((m, d, 1)),
        params=params,
        opt_state=opt_state,
    )
    initial_physical_state = jnp.zeros((d, 1))

    # Define loss function for grad and for reporting
    def loss_fn(p, dist_history):
        return policy_loss(
            apply_fn=model.apply,
            params=p,
            d=d,
            m=m,
            dist_history=dist_history,
            sim=sim,
            cost_fn=cost_fn,
        )

    grad_fn = jax.grad(loss_fn)

    def scan_body(carry, t):
        agentstate, physical_state, key = carry

        # Get action from model
        actions = model.apply(agentstate.params, agentstate.dist_history)
        action = actions[0]

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
            key=key1,
        )

        # Update agent state
        agentstate = update_agentstate(
            agentstate=agentstate,
            next_state=output,
            action=action,
            sim=sim,
            grad_fn=grad_fn,
            optimizer=optimizer,
        )

        # Update physical state for next iteration
        physical_state = next_physical_state

        # Compute loss for reporting
        loss = loss_fn(agentstate.params, agentstate.dist_history)

        new_carry = (agentstate, next_physical_state, key)
        return new_carry, loss

    # Run simulation with scan
    initial_carry = (initial_agentstate, initial_physical_state, loop_key)
    (_, _, _), losses = jax.lax.scan(
        scan_body, initial_carry, jnp.arange(num_steps)
    )

    return losses

def main():
    parser = argparse.ArgumentParser(description="Run GPC/SFC experiments")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config file"
    )
    args = parser.parse_args()

    # Load config file
    spec = importlib.util.spec_from_file_location("config", args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # Set random seed
    key = jax.random.PRNGKey(config.seed)

    print("--- Creating System and Cost Matrices ---")
    # Create random system
    key, A_B_key = jax.random.split(key)
    A, B = create_random_system(
        A_B_key, config.d, config.n, config.min_eig, config.max_eig
    )

    # Create random cost matrices
    key, Q_R_key = jax.random.split(key)
    Q, R = create_random_cost_matrices(Q_R_key, config.d, config.n)

    # Create disturbance
    disturbance_params = config.disturbance_params[config.disturbance_type]
    if config.disturbance_type == "sinusoidal":
        disturbance = lambda d, dist_std, t, key: sinusoidal_disturbance(
            key=key, **disturbance_params
        )
    elif config.disturbance_type == "gaussian":
        disturbance = lambda d, dist_std, t, key: gaussian_disturbance(
            d=d, dist_std=disturbance_params["std"], t=t, key=key
        )
    elif config.disturbance_type == "zero":
        disturbance = lambda d, dist_std, t, key: zero_disturbance(d=d, key=key)
    else:
        raise ValueError(f"Unknown disturbance type: {config.disturbance_type}")

    # Create model
    if config.model_type == "gpc":
        model = GPCModel(
            d=config.d,
            n=config.n,
            m=config.m,
            k=config.k,
            hidden_dims=config.hidden_dims,
        )
    elif config.model_type == "sfc":
        model = SFCModel(
            d=config.d,
            n=config.n,
            m=config.m,
            h=config.h,
            k=config.k,
            gamma=config.gamma,
            hidden_dims=config.hidden_dims,
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

    # Create optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.R_M), optax.adam(config.learning_rate)
    )

    print(f"--- Starting {config.num_trials} Trials (vmapped) ---")

    # Create keys for each trial
    trial_keys = jax.random.split(key, config.num_trials)

    # Vmap the run_trial function
    vmapped_run_trial = jax.vmap(
        functools.partial(
            run_trial,
            A=A,
            B=B,
            Q=Q,
            R=R,
            model=model,
            optimizer=optimizer,
            disturbance=disturbance,
            d=config.d,
            m=config.m,
            num_steps=config.num_steps,
        ),
        in_axes=(0),
    )

    # JIT compile the vmapped function for performance
    jitted_vmapped_run_trial = jax.jit(vmapped_run_trial)

    print("Compiling and running trials... (this may take a moment)")
    all_losses = jitted_vmapped_run_trial(trial_keys)
    all_losses.block_until_ready()  # Wait for all computations to finish
    print("--- All trials completed ---")

    # Compute mean and standard error
    mean_losses = jnp.mean(all_losses, axis=0)
    std_losses = jnp.std(all_losses, axis=0) / jnp.sqrt(config.num_trials)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(mean_losses, label="Mean Loss")
    plt.fill_between(
        range(len(mean_losses)),
        mean_losses - std_losses,
        mean_losses + std_losses,
        alpha=0.2,
    )
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"{config.model_type.upper()} Performance")
    plt.legend()
    plt.grid(True)

    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Save the figure
    filename = f"{config.model_type.upper()}_performance.png"
    filepath = os.path.join(results_dir, filename)
    plt.savefig(filepath)
    print(f"Plot saved to {filepath}")

if __name__ == "__main__":
    main() 