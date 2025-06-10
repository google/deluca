import brax
from brax import envs
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt # For plotting

# Tell Brax to use the mjx backend (JAX-based)

def solve_discrete_are_iterative(A, B, Q, R, max_iters=1000, tol=1e-5):
    """Solves the discrete-time Algebraic Riccati Equation iteratively."""
    P = jnp.zeros_like(Q) 
    for i in range(max_iters):
        P_prev = P
        R_BT_P_B = R + B.T @ P_prev @ B
        R_BT_P_B_reg = R_BT_P_B + jnp.eye(R_BT_P_B.shape[0]) * 1e-8
        try:
            K_intermediate = jnp.linalg.solve(R_BT_P_B_reg, B.T @ P_prev @ A)
        except jnp.linalg.LinAlgError:
            # print("LinAlgError in solve_discrete_are_iterative, using pseudo-inverse.")
            K_intermediate = jnp.linalg.pinv(R_BT_P_B_reg) @ (B.T @ P_prev @ A)
        P = Q + A.T @ P_prev @ A - A.T @ P_prev @ B @ K_intermediate
        if jnp.linalg.norm(P - P_prev) < tol:
            return P
    return P

def calculate_lqr_gain(A, B, Q, R, P):
    """Calculates the LQR gain K."""
    R_BT_P_B = R + B.T @ P @ B
    R_BT_P_B_reg = R_BT_P_B + jnp.eye(R_BT_P_B.shape[0]) * 1e-8
    try:
        K = jnp.linalg.solve(R_BT_P_B_reg, B.T @ P @ A)
    except jnp.linalg.LinAlgError:
        # print("LinAlgError in calculate_lqr_gain, using pseudo-inverse.")
        K = jnp.linalg.pinv(R_BT_P_B_reg) @ (B.T @ P @ A)
    return K

def main():
    env_name = "inverted_pendulum"
    try:
        env = envs.get_environment(env_name=env_name, backend="mjx")
        print(f"Successfully loaded environment: {env_name}")
    except Exception as e: # Catching a broader exception to see if it's env related
        print(f"Error initializing Brax environment '{env_name}': {e}")
        # Attempt to list available environments if one is curious
        try:
            available_envs = envs.list_environments()
            print(f"Available Brax environments: {available_envs}")
        except Exception as list_e:
            print(f"Could not list Brax environments: {list_e}")
        return

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # Get system dimensions and observation size
    nq = env.sys.nq
    nv = env.sys.nv
    obs_size = env.observation_size
    print(f"Environment: {env_name}, nq={nq}, nv={nv}, observation_size={obs_size}")

    # For inverted_pendulum (cart-pole), nq=2, nv=2. State is [x_cart, theta_pole, x_dot_cart, theta_dot_pole]

    key = jax.random.PRNGKey(0)
    initial_state_for_linearization = jit_reset(key)
    
    # Equilibrium point for internal state [q_sys, qd_sys]
    q_eq = jnp.zeros(nq)  # e.g., [0.0 (cart_pos), 0.0 (pole_angle)]
    qd_eq = jnp.zeros(nv) # e.g., [0.0 (cart_vel), 0.0 (pole_vel)]
    x_internal_eq = jnp.concatenate([q_eq, qd_eq]) 
    u_eq = jnp.array([0.0]) # Single torque/force actuator

    # Template pipeline_state at equilibrium for linearization
    eq_pipeline_state = initial_state_for_linearization.pipeline_state.replace(
        q=q_eq, 
        qd=qd_eq  
    )

    # Define the dynamics function for linearization: (internal_state, action) -> next_internal_state
    def linearized_state_dynamics(x_internal, u, pipeline_state_template, env_instance):
        current_q = x_internal[:nq]
        current_qd = x_internal[nq:]
        current_pipeline_state = pipeline_state_template.replace(q=current_q, qd=current_qd)
        next_pipeline_state = env_instance.pipeline_step(current_pipeline_state, u)
        return jnp.concatenate([next_pipeline_state.q, next_pipeline_state.qd])

    # Get A and B matrices by Jacobian computation
    # Try jacfwd (forward-mode AD) as jacrev (reverse-mode AD, often default for jacobian) can have issues with some loops
    A = jax.jacfwd(linearized_state_dynamics, argnums=0)(x_internal_eq, u_eq, eq_pipeline_state, env)
    B = jax.jacfwd(linearized_state_dynamics, argnums=1)(x_internal_eq, u_eq, eq_pipeline_state, env)
    
    print(f"Linearized A:\n{A}")
    print(f"Linearized B:\n{B}")

    # Define Cost Matrices Q and R for internal state [cart_pos, pole_angle, cart_vel, pole_vel]
    Q_internal = jnp.diag(jnp.array([1.0, 10.0, 0.1, 1.0]))  # Penalize pole angle more
    R_internal = jnp.diag(jnp.array([0.1]))      

    # Solve ARE for P and calculate K
    P_internal = solve_discrete_are_iterative(A, B, Q_internal, R_internal)
    K_internal = calculate_lqr_gain(A, B, Q_internal, R_internal, P_internal)

    print(f"LQR Gain K (for internal state):\n{K_internal}")

    # Simulation Loop
    num_steps = 200
    # Initial state: cart at 0, pole slightly perturbed (0.1 rad), zero velocities
    q_init = jnp.array([0.0, 0.1]) 
    qd_init = jnp.array([0.0, 0.0])
    
    sim_state_0_template = jit_reset(jax.random.PRNGKey(1))
    sim_pipeline_state_0 = sim_state_0_template.pipeline_state.replace(q=q_init, qd=qd_init)
    
    # Construct obs_0 based on actual observation_size
    if obs_size == 5:
        # Standard Brax inverted_pendulum obs: [x_cart, sin(theta_pole), cos(theta_pole), x_dot_cart, theta_dot_pole]
        obs_0 = jnp.array([
            sim_pipeline_state_0.q[0],               # x_cart
            jnp.sin(sim_pipeline_state_0.q[1]),      # sin(theta_pole)
            jnp.cos(sim_pipeline_state_0.q[1]),      # cos(theta_pole)
            sim_pipeline_state_0.qd[0],              # x_dot_cart
            sim_pipeline_state_0.qd[1]               # theta_dot_pole
        ])
    elif obs_size == 4:
        # Common 4-element cartpole obs: [cart_pos, cart_vel, pole_angle_raw, pole_ang_vel_raw]
        # Assuming obs[0]=q[0], obs[1]=qd[0], obs[2]=q[1], obs[3]=qd[1]
        print("Constructing 4-element obs_0 as [q[0], qd[0], q[1], qd[1]]")
        obs_0 = jnp.array([
            sim_pipeline_state_0.q[0],    # cart_pos
            sim_pipeline_state_0.qd[0],   # cart_vel
            sim_pipeline_state_0.q[1],    # pole_angle_raw
            sim_pipeline_state_0.qd[1]    # pole_ang_vel_raw
        ])
    else:
        print(f"ERROR: Unsupported observation_size {obs_size}. Cannot construct obs_0 or configure plotting.")
        return # Exit if observation structure is unknown
    
    current_sim_state = sim_state_0_template.replace(
        pipeline_state=sim_pipeline_state_0,
        obs=obs_0,
        reward=jnp.array(0.0), # Ensure reward and done are initialized properly
        done=jnp.array(0.0)
    )

    history_obs = []
    history_actions = []
    history_reward = []

    print(f"Starting simulation with initial q={current_sim_state.pipeline_state.q}, qd={current_sim_state.pipeline_state.qd}")
    
    expected_obs_shape = (obs_size,) # Use actual obs_size from env
    
    if current_sim_state.obs.shape != expected_obs_shape:
        print(f"ERROR: Initial obs_0 shape is {current_sim_state.obs.shape}, expected {expected_obs_shape} (from env.observation_size)")
        raise ValueError("Initial observation shape is incorrect.")

    for step_num in range(num_steps):
        obs_to_append = current_sim_state.obs
        
        if obs_to_append.shape != expected_obs_shape:
            print(f"ERROR: Step {step_num}, obs shape is {obs_to_append.shape}, expected {expected_obs_shape}")
            print(f"Problematic obs content: {obs_to_append}")
            # You could print the full current_sim_state here for more debug info
            # print(f"Full state at error: {current_sim_state}")
            raise ValueError(f"Inconsistent observation shape detected at step {step_num}.")
        
        # Konditional debug print for obs shape and dtype
        if step_num < 3 or step_num > num_steps - 3: # Print for first/last 3 iterations
            print(f"Step {step_num}: obs shape={obs_to_append.shape}, dtype={obs_to_append.dtype}")

        history_obs.append(obs_to_append)
        
        current_q = current_sim_state.pipeline_state.q
        current_qd = current_sim_state.pipeline_state.qd
        current_x_internal = jnp.concatenate([current_q, current_qd])
        
        error_internal = current_x_internal - x_internal_eq
        action = -K_internal @ error_internal + u_eq 
        
        # Clipping action to control limits
        # ctrl_range is often [[min_val], [max_val]] for a single actuator
        if env.sys.actuator is not None and hasattr(env.sys.actuator, 'ctrl_range') and env.sys.actuator.ctrl_range is not None:
            min_ctrl, max_ctrl = env.sys.actuator.ctrl_range[0,0], env.sys.actuator.ctrl_range[1,0]
            action = jnp.clip(action, min_ctrl, max_ctrl)
        else:
            # Fallback if ctrl_range is not defined or is None, assume some typical limits like [-1, 1]
            action = jnp.clip(action, -1.0, 1.0)
            if step_num == 0: # Print warning only once
                 print("Warning: Actuator ctrl_range not found or None, clipping to [-1, 1].")

        history_actions.append(action)
        current_sim_state = jit_step(current_sim_state, action)
        history_reward.append(current_sim_state.reward)

        if current_sim_state.done and current_sim_state.done.ndim > 0: # Check if done is an array and take first element
            if current_sim_state.done[0]:
                print(f"Episode finished after {step_num+1} steps due to done signal.")
                # break # uncomment to stop early
        elif current_sim_state.done: # if done is scalar
             if current_sim_state.done:
                print(f"Episode finished after {step_num+1} steps due to done signal.")
                # break

    history_obs_np = jnp.array(history_obs)
    history_actions_np = jnp.array(history_actions)
    history_reward_np = jnp.array(history_reward)

    print(f"Final obs: {history_obs_np[-1]}")
    print(f"Total reward: {jnp.sum(history_reward_np)}")

    # Plotting
    # dt for time axis. env.dt might not exist directly. Use env.sys.opt.timestep
    dt = env.sys.opt.timestep if hasattr(env.sys, 'opt') and hasattr(env.sys.opt, 'timestep') else 0.02 # Default if not found
    time_axis = jnp.arange(len(history_obs_np)) * dt 

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Adjust plotting based on actual observation size and content
    # This part will need careful adjustment if obs_size is not 5 and content differs
    if obs_size == 5:
        # Assuming obs = [x_cart, sin(theta_pole), cos(theta_pole), x_dot_cart, theta_dot_pole]
        theta_from_obs = jnp.arctan2(history_obs_np[:, 1], history_obs_np[:, 2]) # sin is index 1, cos is index 2
        theta_dot_from_obs = history_obs_np[:, 4] # theta_dot is index 4
        axs[0].plot(time_axis, theta_from_obs, label='pole_angle (from sin/cos obs)')
        axs[1].plot(time_axis, theta_dot_from_obs, label='pole_angular_velocity (obs)')
    elif obs_size == 4:
        # GUESSING: obs = [cart_pos, cart_vel, pole_angle_raw, pole_ang_vel_raw]
        # This is a common 4-element cartpole observation.
        # If so, pole_angle_raw might be at index 2, pole_ang_vel_raw at index 3.
        print("Warning: obs_size is 4. Plotting assumes obs = [cart_pos, cart_vel, pole_angle_raw, pole_ang_vel_raw]")
        # Pole angle might be history_obs_np[:, 2], Pole angular velocity might be history_obs_np[:, 3]
        # For now, let's try to plot these if they exist. User might need to verify.
        if history_obs_np.shape[1] > 2:
            axs[0].plot(time_axis, history_obs_np[:, 2], label='obs[2] (suspected pole_angle)')
        if history_obs_np.shape[1] > 3:
            axs[1].plot(time_axis, history_obs_np[:, 3], label='obs[3] (suspected pole_ang_vel)')
    else:
        print(f"Warning: obs_size is {obs_size}. Plotting for specific components is not configured.")
        # Plot first two components as an example if they exist
        if history_obs_np.shape[1] > 0:
            axs[0].plot(time_axis, history_obs_np[:, 0], label='obs[0]')
        if history_obs_np.shape[1] > 1:
            axs[1].plot(time_axis, history_obs_np[:, 1], label='obs[1]')

    axs[0].set_ylabel('Angle/Obs Component')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_ylabel('AngVel/Obs Component')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(time_axis, history_actions_np, label='Action (torque/force)')
    axs[2].set_xlabel(f'Time (s), dt={dt:.3f}')
    axs[2].set_ylabel('Torque')
    axs[2].legend()
    axs[2].grid(True)

    fig.suptitle(f'LQR Control of Brax Environment ({env_name})')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_filename = "brax_lqr_pendulum_simulation.png"
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    # plt.show() 

if __name__ == '__main__':
    main()
