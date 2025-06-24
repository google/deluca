"""Experiment script for comparing GPC and SFC models."""

import argparse
import functools
from typing import Callable, Tuple, Any, Optional, Sequence
import importlib.util
import sys
sys.path.append("../../../")
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import linen as nn

from deluca.experimental.agents.agent import policy_loss, update_agentstate, AgentState
from deluca.experimental.agents.gpc import get_gpc_features
from deluca.experimental.agents.sfc import get_sfc_features
from deluca.experimental.agents.model import FullyConnectedModel, SequentialModel, GridModel
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
    parser = argparse.ArgumentParser(description="Run GPC/SFC experiments on LDS")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config file"
    )
    args = parser.parse_args()

    # Load config file
    spec = importlib.util.spec_from_file_location("config", args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    
    # Setup results directory
    base_results_dir = "lds_results"
    results_dir = os.path.join(base_results_dir, "model_comparison")
    os.makedirs(results_dir, exist_ok=True)

    # Set random seed
    key = jax.random.PRNGKey(config.seed)

    print("--- Creating a Common Set of Environments and Costs ---")
    # Create a single set of system/cost matrices for all trials
    key, env_key = jax.random.split(key)
    A, B = create_random_system(
        env_key, config.d, config.n, config.min_eig, config.max_eig
    )
    key, cost_key = jax.random.split(key)
    Q, R = create_random_cost_matrices(cost_key, config.d, config.n)

    # Create a single set of trial keys for all experiments
    key, trial_key_master = jax.random.split(key)
    trial_keys = jax.random.split(trial_key_master, config.num_trials)
    
    # Create optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.R_M), optax.adam(config.learning_rate)
    )

    disturbance_types = config.disturbance_params.keys()

    for disturbance_type in disturbance_types:
        print(f"\n--- Running Experiment for Disturbance Type: {disturbance_type.upper()} ---")
        
        # Create disturbance function for this type
        disturbance_params = config.disturbance_params[disturbance_type]
        if disturbance_type == "sinusoidal":
            disturbance = lambda d, dist_std, t, key: sinusoidal_disturbance(
                d=d,
                noise_amplitude=disturbance_params["noise_amplitude"],
                noise_frequency=jnp.ones(d) * disturbance_params["noise_frequency"],
                t=t,
                key=key,
            )
        elif disturbance_type == "gaussian":
            disturbance = lambda d, dist_std, t, key: gaussian_disturbance(
                d=d, dist_std=disturbance_params["dist_std"], t=t, key=key
            )
        elif disturbance_type == "zero":
            disturbance = lambda d, dist_std, t, key: zero_disturbance(d=d, dist_std=dist_std, t=t, key=key)
        else:
            raise ValueError(f"Unknown disturbance type: {disturbance_type}")

        all_losses = {}

        # --- LINEAR MODELS ---
        print("\n-- Evaluating LINEAR Models --")
        # GPC (Linear)
        gpc_features = get_gpc_features(m=config.m, d=config.d, offset=0, dist_history=jnp.zeros((config.m, config.d, 1)))
        vmapped_gpc_linear_run = jax.vmap(functools.partial(run_trial, A=A, B=B, Q=Q, R=R, model=gpc_features, optimizer=optimizer, disturbance=disturbance, d=config.d, m=config.m, num_steps=config.num_steps), in_axes=(0))
        all_losses['gpc_linear'] = jax.jit(vmapped_gpc_linear_run)(trial_keys)
        print("GPC (Linear) trials complete.")

        # SFC (Linear)
        sfc_features = get_sfc_features(m=config.m, d=config.d, h=config.h, gamma=config.gamma)
        vmapped_sfc_linear_run = jax.vmap(functools.partial(run_trial, A=A, B=B, Q=Q, R=R, model=sfc_features, optimizer=optimizer, disturbance=disturbance, d=config.d, m=config.m, num_steps=config.num_steps), in_axes=(0))
        all_losses['sfc_linear'] = jax.jit(vmapped_sfc_linear_run)(trial_keys)
        print("SFC (Linear) trials complete.")

        # --- NEURAL NETWORK MODELS ---
        print("\n-- Evaluating NEURAL NET Models --")
        # GPC (NN)
        gpc_nn_model = GPCModel(d=config.d, n=config.n, m=config.m, k=config.k, hidden_dims=config.hidden_dims_nn)
        vmapped_gpc_nn_run = jax.vmap(functools.partial(run_trial, A=A, B=B, Q=Q, R=R, model=gpc_nn_model, optimizer=optimizer, disturbance=disturbance, d=config.d, m=config.m, num_steps=config.num_steps), in_axes=(0))
        all_losses['gpc_nn'] = jax.jit(vmapped_gpc_nn_run)(trial_keys)
        print("GPC (NN) trials complete.")

        # SFC (NN)
        sfc_nn_model = SFCModel(d=config.d, n=config.n, m=config.m, h=config.h, k=config.k, gamma=config.gamma, hidden_dims=config.hidden_dims_nn)
        vmapped_sfc_nn_run = jax.vmap(functools.partial(run_trial, A=A, B=B, Q=Q, R=R, model=sfc_nn_model, optimizer=optimizer, disturbance=disturbance, d=config.d, m=config.m, num_steps=config.num_steps), in_axes=(0))
        all_losses['sfc_nn'] = jax.jit(vmapped_sfc_nn_run)(trial_keys)
        print("SFC (NN) trials complete.")

        for name, losses in all_losses.items():
            losses.block_until_ready()
            print(f"All {name.upper()} trials finished computation.")

        # --- Process and Plot Results ---
        plt.figure(figsize=(12, 8))
        
        plot_configs = {
            'gpc_linear': {'label': 'GPC (Linear)', 'color': 'blue'},
            'sfc_linear': {'label': 'SFC (Linear)', 'color': 'green'},
            'gpc_nn': {'label': 'GPC (NN)', 'color': 'red'},
            'sfc_nn': {'label': 'SFC (NN)', 'color': 'purple'},
        }

        for name, losses in all_losses.items():
            mean_losses = jnp.mean(losses, axis=0)
            std_losses = jnp.std(losses, axis=0) / jnp.sqrt(config.num_trials)
            
            plt.plot(mean_losses, label=plot_configs[name]['label'], color=plot_configs[name]['color'])
            plt.fill_between(
                range(len(mean_losses)),
                mean_losses - std_losses,
                mean_losses + std_losses,
                alpha=0.2,
                color=plot_configs[name]['color']
            )

        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(f"Model Comparison on LDS ({disturbance_type.capitalize()} Disturbance)")
        plt.legend()
        plt.grid(True)
        
        # Save the figure
        filename = f"model_comparison_{disturbance_type}.png"
        filepath = os.path.join(results_dir, filename)
        plt.savefig(filepath)
        print(f"Comparison plot saved to {filepath}")


if __name__ == "__main__":
    main() 