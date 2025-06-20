"""Experiment script for comparing GPC and SFC models on Brax environments."""

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
from brax import envs

from deluca.experimental.agents.agent import policy_loss, update_agentstate, AgentState
from deluca.experimental.agents.gpc import GPCModel
from deluca.experimental.agents.sfc import SFCModel
from deluca.experimental.enviornments.disturbances.sinusoidal import sinusoidal_disturbance
from deluca.experimental.enviornments.disturbances.gaussian import gaussian_disturbance
from deluca.experimental.enviornments.disturbances.zero import zero_disturbance

def brax_step(
    state: envs.State,
    u: jnp.ndarray,
    sim: Callable,
    output_map: Callable,
    disturbance: Callable,
    d: int,
    t_sim_step: int,
    key: jax.random.PRNGKey,
) -> Tuple[envs.State, jnp.ndarray]:
    """Custom step function for Brax environments to handle disturbances."""
    next_state = sim(state, u)
    w = disturbance(d, 1.0, t_sim_step, key).reshape(next_state.obs.shape)
    disturbed_obs = next_state.obs + w
    next_state_disturbed = next_state.replace(obs=disturbed_obs)
    output = output_map(next_state_disturbed)
    return next_state_disturbed, output

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
    initial_physical_state: envs.State,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Run a single trial for the Brax environment."""
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

    def loss_fn_for_grad(p, dist_history):
        actions = model.apply(p, dist_history)
        # Simplified loss for gradient calculation: just the cost of the first action
        # This is a simplification; a more complex version might unroll the simulation
        # but that is incompatible with the current brax sim state representation.
        # We use the state from the agent's perspective.
        # This is a heuristic for policy gradient.
        # The state for this simulation should be the *initial* agent state.
        sim_next_state, _ = brax_step(
            initial_physical_state.replace(obs=initial_agentstate.state.flatten()), actions[0].flatten(), sim, output_map, zero_disturbance, d, 0, loop_key
        )
        return cost_fn(sim_next_state, actions[0])

    grad_fn = jax.grad(loss_fn_for_grad)

    def scan_body(carry, t):
        agentstate, physical_state, key = carry

        actions = model.apply(agentstate.params, agentstate.dist_history)
        action = actions[0].flatten()

        key, step_key = jax.random.split(key)
        next_physical_state, output = brax_step(
            physical_state, action, sim, output_map, disturbance, d, t, step_key
        )

        # Since sim(state, action) is problematic for brax, we compute disturbance differently
        # We get the "clean" next state by stepping from the previous physical state without disturbance
        clean_next_state, _ = brax_step(physical_state, action, sim, output_map, zero_disturbance, d, t, step_key)
        inferred_disturbance = output - clean_next_state.obs.reshape(-1, 1)

        # Update disturbance history
        new_dist_history = jnp.roll(agentstate.dist_history, shift=-1, axis=0)
        new_dist_history = new_dist_history.at[-1].set(inferred_disturbance)

        # Update agent state (manual update to avoid sim call)
        grads = grad_fn(agentstate.params, new_dist_history)
        updates, new_opt_state = optimizer.update(grads, agentstate.opt_state, agentstate.params)
        new_params = optax.apply_updates(agentstate.params, updates)

        agentstate = AgentState(
            controller_t=agentstate.controller_t + 1,
            state=output,
            dist_history=new_dist_history,
            params=new_params,
            opt_state=new_opt_state,
        )
        
        loss = cost_fn(next_physical_state, action)
        
        new_carry = (agentstate, next_physical_state, key)
        return new_carry, (loss, next_physical_state.obs)

    initial_carry = (initial_agentstate, initial_physical_state, loop_key)
    (_, _, _), (losses, states) = jax.lax.scan(
        scan_body, initial_carry, jnp.arange(num_steps)
    )

    return losses, states

def main():
    parser = argparse.ArgumentParser(description="Run GPC/SFC experiments on Brax")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("config", args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    base_results_dir = "brax_results"
    model_dir_name = "nn_model" if config.hidden_dims else "linear_model"
    results_dir = os.path.join(base_results_dir, config.BRAX_ENV_NAME, model_dir_name)
    os.makedirs(results_dir, exist_ok=True)

    key = jax.random.PRNGKey(config.seed)
    
    brax_env = envs.get_environment(env_name=config.BRAX_ENV_NAME, backend='spring')
    d, n = brax_env.observation_size, brax_env.action_size

    key, reset_key = jax.random.split(key)
    initial_physical_state = jax.jit(brax_env.reset)(rng=reset_key)

    def brax_sim_wrapper(jit_step_fn):
        def sim(state, action):
            return jit_step_fn(state, action)
        return sim

    def brax_output_map(state):
        return state.obs.reshape(-1, 1)

    def brax_cost_fn(state, action):
        return -state.reward

    sim = brax_sim_wrapper(jax.jit(brax_env.step))
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.R_M), optax.adam(config.learning_rate)
    )

    disturbance_types = config.disturbance_params.keys()

    for disturbance_type in disturbance_types:
        print(f"--- Running experiment for disturbance type: {disturbance_type} ---")
        disturbance_params = config.disturbance_params[disturbance_type]
        if disturbance_type == "sinusoidal":
            disturbance = lambda d, dist_std, t, key: sinusoidal_disturbance(
                d=d, noise_amplitude=disturbance_params["noise_amplitude"],
                noise_frequency=jnp.ones(d) * disturbance_params["noise_frequency"], t=t, key=key)
        elif disturbance_type == "gaussian":
            disturbance = lambda d, dist_std, t, key: gaussian_disturbance(
                d=d, dist_std=disturbance_params["dist_std"], t=t, key=key)
        elif disturbance_type == "zero":
            disturbance = zero_disturbance
        else:
            raise ValueError(f"Unknown disturbance type: {disturbance_type}")

        key, trial_key_master = jax.random.split(key)
        trial_keys = jax.random.split(trial_key_master, config.num_trials)

        print("--- Creating GPC Model ---")
        gpc_model = GPCModel(d=d, n=n, m=config.m_gpc, k=config.k_gpc, hidden_dims=config.hidden_dims)

        print(f"--- Starting {config.num_trials} GPC Trials (vmapped) ---")
        vmapped_gpc_run = jax.vmap(
            functools.partial(run_trial, model=gpc_model, optimizer=optimizer, disturbance=disturbance, sim=sim, 
                              output_map=brax_output_map, cost_fn=brax_cost_fn, d=d, m=config.m_gpc, 
                              num_steps=config.num_steps, initial_physical_state=initial_physical_state),
            in_axes=(0))
        jitted_gpc_run = jax.jit(vmapped_gpc_run)
        gpc_losses, gpc_states = jitted_gpc_run(trial_keys)
        gpc_losses.block_until_ready()
        print("--- GPC trials completed ---")

        print("--- Creating SFC Model ---")
        sfc_model = SFCModel(d=d, n=n, m=config.m_sfc, h=config.h_sfc, k=config.k_sfc, gamma=config.gamma, hidden_dims=config.hidden_dims)

        print(f"--- Starting {config.num_trials} SFC Trials (vmapped) ---")
        vmapped_sfc_run = jax.vmap(
            functools.partial(run_trial, model=sfc_model, optimizer=optimizer, disturbance=disturbance, sim=sim,
                              output_map=brax_output_map, cost_fn=brax_cost_fn, d=d, m=config.m_sfc,
                              num_steps=config.num_steps, initial_physical_state=initial_physical_state),
            in_axes=(0))
        jitted_sfc_run = jax.jit(vmapped_sfc_run)
        sfc_losses, sfc_states = jitted_sfc_run(trial_keys)
        sfc_losses.block_until_ready()
        print("--- SFC trials completed ---")

        gpc_mean_losses = jnp.mean(gpc_losses, axis=0)
        gpc_std_losses = jnp.std(gpc_losses, axis=0) / jnp.sqrt(config.num_trials)
        sfc_mean_losses = jnp.mean(sfc_losses, axis=0)
        sfc_std_losses = jnp.std(sfc_losses, axis=0) / jnp.sqrt(config.num_trials)

        plt.figure(figsize=(10, 6))
        plt.plot(gpc_mean_losses, label="GPC Mean Loss")
        plt.fill_between(range(len(gpc_mean_losses)), gpc_mean_losses - gpc_std_losses, gpc_mean_losses + gpc_std_losses, alpha=0.2)
        plt.plot(sfc_mean_losses, label="SFC Mean Loss")
        plt.fill_between(range(len(sfc_mean_losses)), sfc_mean_losses - sfc_std_losses, sfc_mean_losses + sfc_std_losses, alpha=0.2)
        plt.xlabel("Step")
        plt.ylabel("Loss (Negative Reward)")
        plt.title(f"GPC vs SFC on Brax {config.BRAX_ENV_NAME} ({disturbance_type.capitalize()} Disturbance)")
        plt.legend()
        plt.grid(True)

        filename = f"brax_GPC_vs_SFC_{disturbance_type}.png"
        filepath = os.path.join(results_dir, filename)
        plt.savefig(filepath)
        print(f"Plot saved to {filepath}")

if __name__ == "__main__":
    main() 