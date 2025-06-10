import brax
from brax import envs
from brax.training.agents.ppo import train as ppo_train
from brax.training.agents.ppo import networks as ppo_networks
import jax
import jax.numpy as jnp
import optax # For optimizer
import flax.linen as nn # For neural network definitions

def main():
    # Environment definition
    env_name = "inverted_pendulum"
    env = envs.get_environment(env_name)
    episode_length = 200 # typical for inverted pendulum

    # PPO parameters
    num_timesteps = 1_000_000
    learning_rate = 3e-4
    batch_size = 256
    num_envs = 128 # Number of parallel environments
    num_eval_envs = 32
    reward_scaling = 1.0
    entropy_cost = 1e-3
    unroll_length = 10 # The number of transitions per environment to collect before updating
    num_minibatches = 32
    num_updates_per_batch = 8
    discounting = 0.97
    gae_lambda = 0.95
    clipping_epsilon = 0.2
    seed = 0

    # Create PPO networks
    # This explicit creation might not be directly used by train if network_factory is correct,
    # but kept for clarity or potential later use.
    ppo_network = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=None, # No preprocessing for this simple env
        hidden_layer_sizes=(64, 64) # Reverting to hidden_layer_sizes for older brax version
    )

    # Optimizer (learning rate is passed to train function directly in this Brax version)
    # optimizer = optax.adam(learning_rate=learning_rate) # Not passed directly

    # Training function
    make_policy, params, metrics = ppo_train.train(
        environment=env,
        num_timesteps=num_timesteps,
        episode_length=episode_length,
        action_repeat=1,
        num_envs=num_envs,
        num_eval_envs=num_eval_envs,
        learning_rate=learning_rate,
        entropy_cost=entropy_cost,
        discounting=discounting,
        unroll_length=unroll_length,
        batch_size=batch_size,
        num_minibatches=num_minibatches,
        num_updates_per_batch=num_updates_per_batch,
        gae_lambda=gae_lambda,
        clipping_epsilon=clipping_epsilon,
        reward_scaling=reward_scaling,
        normalize_observations=True,
        network_factory=ppo_networks.make_ppo_networks, # Pass the function directly
        # optimizer=optimizer, # Removed optimizer argument
        seed=seed,
        progress_fn=lambda num_steps, metrics: print(f"Step: {num_steps}, Metrics: {metrics}"),
        # Make sure policy_params_fn is correctly passed if needed, or relies on defaults
    )

    print("Training finished.")
    # You can save the params here or run an evaluation loop
    # For example, to visualize:
    # from brax.io import html
    # jit_env_reset = jax.jit(env.reset)
    # jit_env_step = jax.jit(env.step)
    # jit_policy = jax.jit(make_policy(params, deterministic=True))

    # key = jax.random.PRNGKey(seed + 1)
    # state = jit_env_reset(key)
    # rollout = [state.pipeline_state]
    # for _ in range(episode_length):
    #     act_key, key = jax.random.split(key)
    #     action = jit_policy(state.obs, act_key)
    #     state = jit_env_step(state, action)
    #     rollout.append(state.pipeline_state)
    
    # with open("inverted_pendulum_trained.html", "w") as f:
    #     f.write(html.render(env.sys.replace(dt=env.dt), rollout))
    # print("Rendered trained policy to inverted_pendulum_trained.html")


if __name__ == '__main__':
    main() 