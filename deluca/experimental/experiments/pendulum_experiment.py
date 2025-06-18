"""Experiment script for comparing GPC and SFC models on a Pendulum environment."""

import argparse
import functools
from typing import Callable, Any, Tuple
import importlib.util
import sys
sys.path.append("../../../")
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import linen as nn

from deluca.experimental.agents.agent import policy_loss, update_agentstate, AgentState
from deluca.experimental.agents.gpc import GPCModel
from deluca.experimental.agents.sfc import SFCModel
from deluca.experimental.enviornments.disturbances.sinusoidal import sinusoidal_disturbance
from deluca.experimental.enviornments.disturbances.gaussian import gaussian_disturbance
from deluca.experimental.enviornments.disturbances.zero import zero_disturbance
from deluca.experimental.enviornments.sim_and_output.pendulum import pendulum_sim, pendulum_output
from deluca.experimental.enviornments.cost_functions.pendulum_cost import pendulum_cost_evaluate
from deluca.experimental.enviornments.env import step

def run_trial(
    key: jax.random.PRNGKey,
    model: nn.Module,
    optimizer: Any,
    disturbance: Callable,
    sim: Callable,
    output_map: Callable,
    cost_fn: Callable,
    d: int,
    m: int,
    num_steps: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Run a single trial for the pendulum environment."""
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
    # Start the pendulum with a slight angle to encourage learning
    initial_physical_state = jnp.zeros((d, 1)).at[0].set(0.15)

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
            dist_std=1.0,  # Note: dist_std is not used by all disturbances
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
        loss = cost_fn(output, action)

        new_carry = (agentstate, next_physical_state, key2)
        return new_carry, (loss, physical_state)

    # Run simulation with scan
    initial_carry = (initial_agentstate, initial_physical_state, loop_key)
    (_, _, _), (losses, states) = jax.lax.scan(
        scan_body, initial_carry, jnp.arange(num_steps)
    )

    return losses, states

def main():
    parser = argparse.ArgumentParser(description="Run GPC/SFC experiments on Pendulum")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config file"
    )
    args = parser.parse_args()

    # Load config file
    spec = importlib.util.spec_from_file_location("config", args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # Determine results directory based on model type
    base_results_dir = "pendulum_results"
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
    output_map = pendulum_output
    cost_fn = pendulum_cost_evaluate

    # Create optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.R_M), optax.adam(config.learning_rate)
    )

    disturbance_types = config.disturbance_params.keys()

    for disturbance_type in disturbance_types:
        print(f"--- Running experiment for disturbance type: {disturbance_type} ---")
        # Create disturbance
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

        # Generate a common set of trial keys for this disturbance type
        key, trial_key_master = jax.random.split(key)
        trial_keys = jax.random.split(trial_key_master, config.num_trials)

        # --- Run GPC ---
        print("--- Creating GPC Model ---")
        gpc_model = GPCModel(
            d=config.d, n=config.n, m=config.m_gpc, k=config.k_gpc, hidden_dims=config.hidden_dims
        )

        print(f"--- Starting {config.num_trials} GPC Trials (vmapped) ---")
        vmapped_gpc_run_trial = jax.vmap(
            functools.partial(
                run_trial,
                model=gpc_model,
                optimizer=optimizer,
                disturbance=disturbance,
                sim=sim,
                output_map=output_map,
                cost_fn=cost_fn,
                d=config.d,
                m=config.m_gpc,
                num_steps=config.num_steps,
            ),
            in_axes=(0),
        )
        jitted_gpc_run_trial = jax.jit(vmapped_gpc_run_trial)
        print("Compiling and running GPC trials...")
        gpc_losses, gpc_states = jitted_gpc_run_trial(trial_keys)
        gpc_losses.block_until_ready()
        gpc_states.block_until_ready()
        print("--- GPC trials completed ---")

        # --- Run SFC ---
        print("--- Creating SFC Model ---")
        sfc_model = SFCModel(
            d=config.d, n=config.n, m=config.m_sfc, h=config.h_sfc, k=config.k_sfc, gamma=config.gamma, hidden_dims=config.hidden_dims
        )

        print(f"--- Starting {config.num_trials} SFC Trials (vmapped) ---")
        vmapped_sfc_run_trial = jax.vmap(
            functools.partial(
                run_trial,
                model=sfc_model,
                optimizer=optimizer,
                disturbance=disturbance,
                sim=sim,
                output_map=output_map,
                cost_fn=cost_fn,
                d=config.d,
                m=config.m_sfc,
                num_steps=config.num_steps,
            ),
            in_axes=(0),
        )
        jitted_sfc_run_trial = jax.jit(vmapped_sfc_run_trial)
        print("Compiling and running SFC trials...")
        sfc_losses, sfc_states = jitted_sfc_run_trial(trial_keys)
        sfc_losses.block_until_ready()
        sfc_states.block_until_ready()
        print("--- SFC trials completed ---")

        # --- Process and Plot Results ---
        gpc_mean_losses = jnp.mean(gpc_losses, axis=0)
        gpc_std_losses = jnp.std(gpc_losses, axis=0) / jnp.sqrt(config.num_trials)
        
        sfc_mean_losses = jnp.mean(sfc_losses, axis=0)
        sfc_std_losses = jnp.std(sfc_losses, axis=0) / jnp.sqrt(config.num_trials)

        plt.figure(figsize=(10, 6))
        
        # Plot GPC
        plt.plot(gpc_mean_losses, label="GPC Mean Loss")
        plt.fill_between(
            range(len(gpc_mean_losses)),
            gpc_mean_losses - gpc_std_losses,
            gpc_mean_losses + gpc_std_losses,
            alpha=0.2,
        )
        
        # Plot SFC
        plt.plot(sfc_mean_losses, label="SFC Mean Loss")
        plt.fill_between(
            range(len(sfc_mean_losses)),
            sfc_mean_losses - sfc_std_losses,
            sfc_mean_losses + sfc_std_losses,
            alpha=0.2,
        )

        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(f"GPC vs SFC on Pendulum ({disturbance_type.capitalize()} Disturbance)")
        plt.legend()
        plt.grid(True)

        # Save the figure
        filename = f"pendulum_GPC_vs_SFC_{disturbance_type}.png"
        filepath = os.path.join(results_dir, filename)
        plt.savefig(filepath)
        print(f"Plot saved to {filepath}")

        # --- Plot State Trajectories ---
        gpc_mean_states = jnp.mean(gpc_states, axis=0)
        sfc_mean_states = jnp.mean(sfc_states, axis=0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Theta plot
        ax1.plot(gpc_mean_states[:, 0, 0], label="GPC Mean Theta", color="blue")
        ax1.plot(sfc_mean_states[:, 0, 0], label="SFC Mean Theta", color="green")
        ax1.set_ylabel('Theta (rad)')
        ax1.set_title(f"Pendulum State Over Time ({disturbance_type.capitalize()} Disturbance)")
        ax1.legend()
        ax1.grid(True)

        # Theta_dot plot
        ax2.plot(gpc_mean_states[:, 1, 0], label="GPC Mean Theta Dot", color="blue", linestyle='--')
        ax2.plot(sfc_mean_states[:, 1, 0], label="SFC Mean Theta Dot", color="green", linestyle='--')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Theta Dot (rad/s)')
        ax2.legend()
        ax2.grid(True)

        # Save the state plot
        state_filename = f"pendulum_state_{disturbance_type}.png"
        state_filepath = os.path.join(results_dir, state_filename)
        plt.savefig(state_filepath)
        print(f"State plot saved to {state_filepath}")


if __name__ == "__main__":
    main()
