import argparse
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os

def plot_comparison(gpc_path: str, mpc_path: str, output_dir: str):
    """
    Loads GPC and MPC state trajectory data and plots a comparison.

    Args:
        gpc_path: Path to the GPC/SFC states .npy file.
        mpc_path: Path to the MPC states .npy file.
        output_dir: Directory to save the plot in.
    """
    # Load the data
    try:
        gpc_states = jnp.load(gpc_path)
        mpc_states = jnp.load(mpc_path)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure you have run both experiments.py and mpc.py first.")
        return

    # Compute the mean over the trials axis (axis 0)
    mean_gpc_states = jnp.mean(gpc_states, axis=0).squeeze()
    mean_mpc_states = jnp.mean(mpc_states, axis=0).squeeze()

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("GPC vs. MPC: Pendulum State Trajectories", fontsize=16)

    # Subplot 1: Theta (Angle)
    ax1.plot(mean_gpc_states[:, 0], label="GPC Theta")
    ax1.plot(mean_mpc_states[:, 0], label="MPC Theta", linestyle='--')
    ax1.set_ylabel("Theta (rad)")
    ax1.set_title("Pendulum Angle")
    ax1.legend()
    ax1.grid(True)

    # Subplot 2: Theta Dot (Angular Velocity)
    ax2.plot(mean_gpc_states[:, 1], label="GPC Theta Dot")
    ax2.plot(mean_mpc_states[:, 1], label="MPC Theta Dot", linestyle='--')
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Theta Dot (rad/s)")
    ax2.set_title("Pendulum Angular Velocity")
    ax2.legend()
    ax2.grid(True)

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    filename = "gpc_vs_mpc_state_comparison.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print(f"Comparison plot saved to {filepath}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot comparison of GPC and MPC state trajectories.")
    parser.add_argument(
        "--gpc_path",
        type=str,
        default="results/arrays/pendulum_fully_connected_gpc_states.npy",
        help="Path to the GPC/SFC states .npy file."
    )
    parser.add_argument(
        "--mpc_path",
        type=str,
        default="mpc_results/neural_net_model/arrays/mpc_states.npy",
        help="Path to the MPC states .npy file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/plots",
        help="Directory to save the comparison plot."
    )
    args = parser.parse_args()

    plot_comparison(args.gpc_path, args.mpc_path, args.output_dir)

if __name__ == "__main__":
    main() 