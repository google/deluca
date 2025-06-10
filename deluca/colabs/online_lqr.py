import jax
from jax import numpy as jp
import jax.lax as lax
import jax.tree_util as jtu
from functools import partial
from IPython.display import HTML
import matplotlib.pyplot as plt

import brax
import flax # Required by brax.physics.base.State
from brax import envs
from brax.io import html

# # Enable JAX 64-bit mode
# jax.config.update("jax_enable_x64", True)

env_name = 'inverted_pendulum'
backend = 'generalized'  # @param ['generalized', 'positional', 'spring']

env = envs.get_environment(env_name=env_name,
                           backend=backend)
# Get a template state from a reset, and the JITted step function
template_brax_state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))
jit_env_step = jax.jit(env.step) # JIT env.step once
env_dt_for_plotting = env.dt # Store dt for plotting


def solve_dare_iterative(A, B, Q, R, num_iterations=50, tolerance=1e-7):
    """Solves the discrete-time algebraic Riccati equation (DARE) iteratively."""
    P = Q
    for _ in range(num_iterations):
        P_prev = P
        try:
            R_plus_BTPA_inv = jp.linalg.inv(R + B.T @ P @ B)
        except jp.linalg.LinAlgError:
            # Add regularization if R + B.T @ P @ B is singular
            # print("Warning: R + B.T @ P @ B is singular. Adding regularization.")
            reg = 1e-6 * jp.eye(R.shape[0])
            R_plus_BTPA_inv = jp.linalg.inv(R + B.T @ P @ B + reg)

        P = A.T @ P @ A - A.T @ P @ B @ R_plus_BTPA_inv @ B.T @ P @ A + Q
        if jp.max(jp.abs(P - P_prev)) < tolerance:
            break
    try:
        K_inv_term = jp.linalg.inv(R + B.T @ P @ B)
    except jp.linalg.LinAlgError:
        # print("Warning: R + B.T @ P @ B is singular for K. Adding regularization.")
        reg = 1e-6 * jp.eye(R.shape[0])
        K_inv_term = jp.linalg.inv(R + B.T @ P @ B + reg)
    K = K_inv_term @ B.T @ P @ A
    return K, P


def get_AB_matrices(current_env, template_brax_state_arg, q_current_vec, qd_current_vec, act_ref_vec, local_jit_env_step):
    """Linearizes the system dynamics dx = Ax + Bu around (q_current, qd_current, act_ref)."""
    q_size = current_env.sys.q_size()

    # @jax.jit # JITting this inner function can be slow with changing shapes if not handled carefully
    def f_dynamics_q_qd(q_vec, qd_vec, action_vec):
        # Create a pipeline_state with the given q_vec, qd_vec
        temp_pipeline_s = template_brax_state_arg.pipeline_state.replace(q=q_vec, qd=qd_vec)
        # Create a full Brax state using this pipeline_state and other fields from template_brax_state_arg
        input_state_for_env_step = template_brax_state_arg.replace(pipeline_state=temp_pipeline_s)
        # Call env.step
        next_brax_state_from_env_step = local_jit_env_step(input_state_for_env_step, action_vec)
        # Extract next q and qd
        return next_brax_state_from_env_step.pipeline_state.q, next_brax_state_from_env_step.pipeline_state.qd

    # @jax.jit
    def f_flat_dynamics(flat_state_vec, action_vec):
        q_vec = flat_state_vec[:q_size]
        qd_vec = flat_state_vec[q_size:]
        next_q, next_qd = f_dynamics_q_qd(q_vec, qd_vec, action_vec)
        return jp.concatenate([next_q, next_qd])

    x_current_flat = jp.concatenate([q_current_vec, qd_current_vec])
    
    A = jax.jacobian(f_flat_dynamics, argnums=0)(x_current_flat, act_ref_vec)
    B = jax.jacobian(f_flat_dynamics, argnums=1)(x_current_flat, act_ref_vec)
    
    return A, B

# LQR Setup (done once)
q_eq = jp.zeros(env.sys.q_size())  # Equilibrium position [cart_x, pole_angle]
qd_eq = jp.zeros(env.sys.qd_size()) # Equilibrium velocity [cart_v, pole_av]
act_eq = jp.zeros(env.action_size)  # Equilibrium action (reference action for linearization)

# Define LQR cost matrices Q and R
# State: [cart_pos, pole_angle, cart_vel, pole_ang_vel]
Q_cost = jp.diag(jp.array([1.0, 10.0, 0.1, 0.1]))  # Penalize pole angle more
R_cost = jp.diag(jp.array([0.01]))              # Penalize control effort

# Simulation parameters
NUM_SIM_STEPS = 1000

# Define initial conditions to test (batched)
# Each row in q_batch corresponds to a row in qd_batch
initial_conditions_q_batch = jp.array([
    [0.0, 0.05],      # IC 0: Small pos deviation, zero vel
    [0.0, 0.3],       # IC 1: Moderate pos deviation, zero vel
    [0.0, 0.7],       # IC 2: Larger pos deviation, zero vel (might fail)
    [0.2, 0.0],       # IC 3: Cart displaced, pole upright, zero vel
    [0.0, 0.0],       # IC 5: Upright, but pole has initial angular velocity
    [0.1, 0.05],      # IC 7: Cart and pole slightly displaced, pole has initial vel away from eq
])

initial_conditions_qd_batch = jp.array([
    [0.0, 0.0],       # IC 0: Zero vel
    [0.0, 0.0],       # IC 1: Zero vel
    [0.0, 0.0],       # IC 2: Zero vel
    [0.0, 0.0],       # IC 3: Zero vel
    [0.0, 0.5],       # IC 5: Pole angular velocity of 0.5 rad/s
    [0.0, -0.2],      # IC 7: Pole angular velocity of -0.2 rad/s (destabilizing initially)
])

num_simulations_to_run = initial_conditions_q_batch.shape[0]
assert initial_conditions_q_batch.shape[0] == initial_conditions_qd_batch.shape[0]

# --- JAX scan and vmap for ONLINE LQR simulation ---

def run_single_simulation_scan_online(initial_q_single, initial_qd_single, 
                                      Q_cost_mat, R_cost_mat,
                                      q_eq_global, qd_eq_global, 
                                      template_brax_state_global, local_jit_env_step_global, 
                                      num_steps_sim):
    """Runs a full ONLINE LQR simulation for a single initial condition using lax.scan."""
    initial_pipeline_state = template_brax_state_global.pipeline_state.replace(q=initial_q_single, qd=initial_qd_single)
    true_initial_brax_state = template_brax_state_global.replace(
        pipeline_state=initial_pipeline_state, 
        obs=jp.zeros_like(template_brax_state_global.obs), 
        reward=0.0, done=0.0
    )

    x_eq_flat_global = jp.concatenate([q_eq_global, qd_eq_global])

    def online_lqr_scan_body(carry_brax_state, _):
        """This function is the core of the online LQR, executed at each timestep."""
        # 1. Get current state from the simulator (the carry)
        current_q = carry_brax_state.pipeline_state.q
        current_qd = carry_brax_state.pipeline_state.qd
        
        # 2. Linearize the system dynamics around the current state and zero action
        A_current, B_current = get_AB_matrices(
            env, template_brax_state_global, current_q, current_qd, act_eq, local_jit_env_step_global
        )

        # 3. Solve the DARE for the current A and B to get the new gain K
        K_current, _ = solve_dare_iterative(A_current, B_current, Q_cost_mat, R_cost_mat)

        # 4. Compute the control action
        x_current_flat = jp.concatenate([current_q, current_qd])
        state_deviation = x_current_flat - x_eq_flat_global
        action = -K_current @ state_deviation
        
        # 5. Step the environment
        next_brax_state = local_jit_env_step_global(carry_brax_state, action)
        
        # Collect the state *before* the step, the action taken, and the gain used
        output_pipeline_state = carry_brax_state.pipeline_state
        return next_brax_state, (output_pipeline_state, action, K_current)

    # Use lax.scan to run the simulation loop efficiently
    _, (collected_pipeline_states, collected_actions, collected_gains) = lax.scan(
        online_lqr_scan_body, true_initial_brax_state, xs=None, length=num_steps_sim
    )
    return collected_pipeline_states, collected_actions, collected_gains

# Vmap the simulation runner to run all initial conditions in parallel
vmapped_simulation_runner_online = jax.vmap(
    run_single_simulation_scan_online, 
    in_axes=(0, 0, None, None, None, None, None, None)
)

print(f"Running {num_simulations_to_run} Online LQR simulations in parallel (vmap over scan)...")
batched_pipeline_states, batched_actions, batched_gains = vmapped_simulation_runner_online(
    initial_conditions_q_batch, 
    initial_conditions_qd_batch, 
    Q_cost, 
    R_cost,
    q_eq, 
    qd_eq, 
    template_brax_state, 
    jit_env_step,
    NUM_SIM_STEPS
)
print("All simulations finished.")

# --- Plotting and HTML saving loop ---
for sim_idx in range(num_simulations_to_run):
    print(f"Processing results for simulation {sim_idx+1}/{num_simulations_to_run} (Initial q: {initial_conditions_q_batch[sim_idx]})" )

    # Extract data for the current simulation
    current_sim_pipeline_states = jtu.tree_map(lambda x: x[sim_idx], batched_pipeline_states)
    current_sim_actions = batched_actions[sim_idx]
    current_sim_gains = batched_gains[sim_idx].squeeze(axis=1) # From (N, 1, 4) to (N, 4)

    # Extract states for plotting
    times_plot = jp.arange(NUM_SIM_STEPS) * env_dt_for_plotting
    cart_positions_plot = current_sim_pipeline_states.q[:, 0]
    pole_angles_plot = current_sim_pipeline_states.q[:, 1]
    cart_velocities_plot = current_sim_pipeline_states.qd[:, 0]
    pole_angular_velocities_plot = current_sim_pipeline_states.qd[:, 1]
    actions_plot_single_sim = current_sim_actions.flatten()

    # Save HTML rollout
    rollout_filename = f'rollout_online_lqr_ic_{sim_idx}.html'
    rollout_for_html = [jtu.tree_map(lambda x: x[t], current_sim_pipeline_states) for t in range(NUM_SIM_STEPS)]
    with open(rollout_filename, 'w') as f:
        f.write(html.render(env.sys.tree_replace({'opt.timestep': env_dt_for_plotting}), rollout_for_html))
    print(f"Online LQR simulation rollout saved to {rollout_filename}")

    # Generate and save Matplotlib plot
    plot_filename = f'online_lqr_state_evolution_ic_{sim_idx}.png'
    fig, axs = plt.subplots(6, 1, figsize=(10, 18), sharex=True)

    axs[0].plot(times_plot, cart_positions_plot, label='Cart Position (m)')
    axs[0].set_ylabel('Position (m)')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(times_plot, pole_angles_plot, label='Pole Angle (rad)')
    axs[1].set_ylabel('Angle (rad)')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(times_plot, cart_velocities_plot, label='Cart Velocity (m/s)')
    axs[2].set_ylabel('Velocity (m/s)')
    axs[2].legend()
    axs[2].grid(True)

    axs[3].plot(times_plot, pole_angular_velocities_plot, label='Pole Angular Velocity (rad/s)')
    axs[3].set_ylabel('Angular Velocity (rad/s)')
    axs[3].legend()
    axs[3].grid(True)

    axs[4].plot(times_plot, actions_plot_single_sim, label='Control Action', color='purple')
    axs[4].set_ylabel('Action')
    axs[4].legend()
    axs[4].grid(True)

    axs[5].plot(times_plot, current_sim_gains[:, 0], label='K[0] (pos)')
    axs[5].plot(times_plot, current_sim_gains[:, 1], label='K[1] (angle)')
    axs[5].plot(times_plot, current_sim_gains[:, 2], label='K[2] (vel)')
    axs[5].plot(times_plot, current_sim_gains[:, 3], label='K[3] (ang_vel)')
    axs[5].set_ylabel('LQR Gain Values')
    axs[5].set_xlabel('Time (s)')
    axs[5].legend()
    axs[5].grid(True)

    fig.suptitle(f'Online LQR Control IC {sim_idx}: q0={initial_conditions_q_batch[sim_idx]}, qd0={initial_conditions_qd_batch[sim_idx]}', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(plot_filename)
    plt.close(fig) # Close the figure to free memory
    print(f"Online LQR state evolution plot saved to {plot_filename}")

print("All processing complete.") 