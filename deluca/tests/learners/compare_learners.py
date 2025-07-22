from brax.envs import PipelineEnv, inverted_double_pendulum
from deluca.agents import SimpleRandom
import jax
from deluca.agents._grc import GRC
from deluca.envs._brax import BraxEnv
from deluca.envs._lds import LDS, SinusDisturbance
from deluca.envs.brax._pendulum2d import Pendulum2D
from deluca.learners import LinearLearner, SpectralLearner, NNLearner
from deluca.core import Env
from deluca.utils.printing import Task
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors as mcolors, cm
import numpy as np
from matplotlib.lines import Line2D

def test_learners(learners, trajectories, m=30):
    data = []
    with Task("Testing Learners", len(learners)) as task:
        for learner, name in learners:
            task.update(increment=0, text=f"Testing {name}")
            loss, loss_by_coord, (pred, next_obs) = learner.test(trajectories)

            data.append((name, (pred, next_obs), loss_by_coord))
            task.update()

    # Derive trajectory length T from generated data
    T = trajectories[0].shape[1] if trajectories and isinstance(trajectories, tuple) else 0
    plot_learner_comparison(data, m, T)
    return data
        

def plot_learner_comparison(data, m=30, T: int | None = None):
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    for i, (name, (pred, next_obs), loss_by_coord) in enumerate(data):
        color_maps = [mpl.colormaps["Reds"], mpl.colormaps["Blues"], mpl.colormaps["Greens"], mpl.colormaps["Oranges"], mpl.colormaps["Purples"]]

        # ------------------------------------------------------------------
        # Determine data shape (flattened vs. 3-D) and construct time indices
        # ------------------------------------------------------------------
        if pred.ndim == 3:
            num_traj, T_local, obs_dim = pred.shape
            if T is None:
                T = T_local
            time_idx_flat = np.tile(np.arange(T_local), num_traj)
        elif pred.ndim == 2:
            total_pts, obs_dim = pred.shape
            if T is None:
                raise ValueError("Trajectory length T must be provided for flattened data.")

            group_len = T - m + 1  # points retained per trajectory after history trimming
            num_traj = total_pts // group_len

            # Build per-trajectory time indices and remove the initial m-1 steps to
            # match the Linear/Spectral learner output shape.
            per_traj_times = np.arange(T)[np.arange(T) >= (m - 1)]  # length = group_len
            time_idx_flat = np.tile(per_traj_times, num_traj)

            assert len(time_idx_flat) == total_pts, "Time index reconstruction mismatch"
        else:
            raise ValueError("Unsupported prediction array shape: {pred.shape}")

        # Ensure T is an int for downstream arithmetic
        assert T is not None, "T must be determined before plotting"
        T_int = int(T)

        # Normalizer maps time index -> [0, 1]
        norm = mcolors.Normalize(vmin=0, vmax=T_int - 1)
        # Adjust normalization to make earliest points more visible (not pure white)
        norm_adjusted = mcolors.Normalize(vmin=0.1, vmax=T_int - 1)

        # ------------------------------------------------------------------
        # Plotting per coordinate
        # ------------------------------------------------------------------
        # Colors for every point according to its timestep (shared for all coords)
        # We compute once to avoid redundant work.
        colours_all = cm.get_cmap("Greys")(norm_adjusted(time_idx_flat))  # placeholder; updated per coord

        # Plot each coordinate with its own colour-map.
        for coord in range(obs_dim):
            cmap = color_maps[coord % len(color_maps)]

            if pred.ndim == 3:
                true_vals = next_obs[:, :, coord].reshape(-1)
                pred_vals = pred[:, :, coord].reshape(-1)
                colours = cmap(norm_adjusted(time_idx_flat))
            else:  # 2D flattened
                true_vals = next_obs[:, coord]
                pred_vals = pred[:, coord]
                colours = cmap(norm_adjusted(time_idx_flat))

            mask_initial = time_idx_flat < m
            mask_later = ~mask_initial

            axs[i, 0].scatter(
                true_vals[mask_initial],
                pred_vals[mask_initial],
                c=colours[mask_initial],
                marker="x",
                s=20,
                linewidths=0.8,
                cmap=cmap,
                norm=norm,
                label=None,
            )

            axs[i, 0].scatter(
                true_vals[mask_later],
                pred_vals[mask_later],
                c=colours[mask_later],
                marker="o",
                s=20,
                edgecolors="none",
                cmap=cmap,
                norm=norm,
                label=None,
            )

        # Add y=x line for reference
        min_val = min(next_obs.min(), pred.min())
        max_val = max(next_obs.max(), pred.max())
        axs[i, 0].plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, linewidth=1, label="y=x")

        # ------------------------------------------------------------------
        # Add colourbar and marker legend (only once, on the first learner)
        # ------------------------------------------------------------------
        if i == 0:
            sm = cm.ScalarMappable(cmap=cm.get_cmap("Greys"), norm=norm_adjusted)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axs[i, 0])
            cbar.set_label("Timestep (early → late)")

            marker_handles = [
                Line2D([0], [0], marker="x", color="grey", linestyle="", markersize=6, label=f"First {m} steps"),
                Line2D([0], [0], marker="o", color="grey", linestyle="", markersize=6, label="Later steps"),
            ]
            # Combine y=x line with marker legend
            all_handles = [
                Line2D([0], [0], color="black", linestyle="--", alpha=0.5, linewidth=1, label="y=x"),
            ] + marker_handles
            axs[i, 0].legend(handles=all_handles, loc="upper left")
 
        axs[i, 0].plot(0.2, 1, "r--", alpha=0.75, zorder=0, label="y=x")
        axs[i, 0].set_title(f"{name} Pred vs True")
        axs[i, 0].set_xlabel("True")
        axs[i, 0].set_ylabel("Pred")
    
        axs[i, 1].bar(range(len(loss_by_coord)), loss_by_coord)
        axs[i, 1].set_title(f"{name} Loss by Coord")
        axs[i, 1].set_xlabel("Epoch")
        axs[i, 1].set_ylabel("Loss")
        axs[i, 1].set_xticks(range(len(loss_by_coord)))
        axs[i, 1].set_xticklabels(range(len(loss_by_coord)))
        


    plt.tight_layout()
    plt.legend()
    plt.show()
    

env = LDS()# BraxEnv.from_env(inverted_double_pendulum.InvertedDoublePendulum())
# env.unfreeze()

d_obs = 2
d_action = 3
d_hidden = 10
disturbance = SinusDisturbance()
disturbance.init(d_hidden, 0.5)
env.init(d_action,d_hidden, d_obs, disturbance = disturbance)

agent = GRC(env.A, env.B, env.C)

m = 30
eta = 1e-3

#nn = NNLearner(env, learning_rate=eta)
linear = LinearLearner(env, learning_rate=eta, history_length=m)
spectral = SpectralLearner(env, history_length=m, spectral_history_length=70, spectral_num_filters=m, rng=jax.random.key(2345), learning_rate=eta)


# Doesn't matter which learner we use to generate trajectories

with Task("Learner Comparison", 3) as task:
    trajectories = linear.generate_trajectories(100, 300, agent=agent) 
    task.update()
    
    #nn.learn(trajectories)
    linear.learn(trajectories)
    spectral.learn(trajectories)

    task.update()

    #test_learners([(nn, "NN"), (linear, "Linear"), (spectral, "Spectral")], m=m)
    test_trajectories = linear.generate_trajectories(20, 200, agent=agent)
    test_learners([(linear, "Linear"), (spectral, "Spectral")], test_trajectories, m=m)

