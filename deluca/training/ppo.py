# Copyright 2021 The Deluca Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PPO."""

import collections
import functools

from clu import metric_writers
import deluca.core
import jax
import jax.numpy as jnp
import optax

# pylint:disable=invalid-name

Rollout = collections.namedtuple("Rollout", [
    "agent_states", "actions", "log_probs", "values", "rewards", "done_masks",
    "advantages", "returns"
])


def gae_advantages(rewards: jnp.ndarray, terminal_masks: jnp.ndarray,
                   values: jnp.ndarray, discount: float, gae_param: float):
  """Use Generalized Advantage Estimation (GAE) to compute advantages.

  Args:
    rewards: array shaped (actor_steps, num_agents)
    terminal_masks: array shaped (actor_steps, num_agents), zeros for terminal
      and ones for non-terminal states
    values: array shaped (actor_steps, num_agents), values estimated by critic
    discount: discount factor gamma
    gae_param: GAE parameter lambda

  Returns:
    advantages: calculated advantages shaped (actor_steps, num_agents)
  """
  advantages = [rewards[-1] - values[-1]]
  gae = rewards[-1] - values[-1]
  for t in reversed(range(len(rewards) - 1)):
    delta = rewards[t] + discount * values[t +
                                           1] * terminal_masks[t] - values[t]
    gae = delta + discount * gae_param * terminal_masks[t] * gae
    advantages.append(gae)
  advantages = advantages[::-1]
  return jnp.array(advantages)


@functools.partial(
    jax.jit, static_argnums=(
        5,
        6,
    ))
def ac_rollout(env, env_start_state, env_init_obs, agent, agent_seed: int, H,
               reward_func, gamma, lambda_):
  """Comment."""
  agent_start_state = agent.init(seed=agent_seed.astype(int))

  def rollout_step(inp, counter):
    env_state, env_obs, agent_state = inp
    agent_next_state, action, log_probs, values = agent(agent_state, env_obs)
    env_next_state, env_next_obs = env(env_state, action)
    reward = reward_func(env_state, env_obs, action, env, counter)
    done_mask = 1.0
    return (env_next_state, env_next_obs,
            agent_next_state), (env_next_state, env_next_obs, agent_next_state,
                                action, log_probs, values, reward, done_mask)

  _, t = jax.lax.scan(rollout_step,
                      (env_start_state, env_init_obs, agent_start_state),
                      jnp.arange(H))
  advantages = gae_advantages(t[6], t[7], t[5], gamma, lambda_)
  returns = advantages + t[4]
  return Rollout(t[2], t[3], t[4], t[5], t[6], t[7], advantages, returns)


@functools.partial(
    jax.jit, static_argnums=(
        0,
        3,
        4,
    ))
def ac_parallel_rollouts(
    num,
    env,
    agent,
    H,
    cost_func,
    gamma,
    lambda_,
    agent_seeds=None,
):
  """Parallelize rollouts."""
  if agent_seeds == None:
    agent_seeds = jnp.arange(num)
  env_start_states, env_init_obs = env.reset()
  all_rollouts = jax.vmap(ac_rollout,
                          (None, None, None, None, 0, None, None, None, None))(
                              env, env_start_states, env_init_obs, agent,
                              agent_seeds, H, cost_func, gamma, lambda_)
  return all_rollouts


def ac_get_experience(agent, sim, actor_steps, num_agents, gamma, lambda_,
                      rewards_func):
  """Serialize the experiences."""
  rollouts = ac_parallel_rollouts(
      num_agents,
      sim,
      agent,
      actor_steps,
      rewards_func,
      gamma,
      lambda_,
      agent_seeds=None)
  experiences = jax.tree_map(
      lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:]), rollouts)
  return experiences


def ppo_loss_fn(agent, batch_data, clip_param: float, vf_coeff: float,
                entropy_coeff: float):
  """Evaluate the loss function.

  Compute loss as a sum of three components: the negative of the PPO clipped
  surrogate objective, the value function loss and the negative of the entropy
  bonus.

  Args:
    params: the parameters of the actor-critic model
    apply_fn: the actor-critic model"s apply function
    minibatch: Tuple of five elements forming one experience batch:
               states: shape (batch_size, 84, 84, 4)
               actions: shape (batch_size, 84, 84, 4)
               old_log_probs: shape (batch_size,)
               returns: shape (batch_size,)
               advantages: shape (batch_size,)
    clip_param: the PPO clipping parameter used to clamp ratios in loss function
    vf_coeff: weighs value function loss in total loss
    entropy_coeff: weighs entropy bonus in the total loss

  Returns:
    loss: the PPO loss, scalar quantity
  """
  agent_states, actions, old_log_probs, _, _, _, advantages, returns, = batch_data
  log_probs, values = agent.derive_prob_and_value(agent_states)
  probs = jnp.exp(log_probs)
  value_loss = jnp.mean(jnp.square(returns - values), axis=0)
  entropy = jnp.sum(-probs * log_probs, axis=1).mean()
  log_probs_act_taken = jax.vmap(agent.log_prob_action)(log_probs, actions)
  ratios = jnp.exp(log_probs_act_taken - old_log_probs)
  advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
  pg_loss = ratios * advantages
  clipped_loss = advantages * jax.lax.clamp(1. - clip_param, ratios,
                                            1. + clip_param)
  ppo_loss = -jnp.mean(jnp.minimum(pg_loss, clipped_loss), axis=0)
  return ppo_loss + vf_coeff * value_loss - entropy_coeff * entropy


def train_step(optim, optim_state, agent, experiences, batch_size: int,
               clip_param: float, vf_coeff: float, entropy_coeff: float):
  """Compilable train step.

  Runs an entire epoch of training (i.e. the loop over minibatches within
  an epoch is included here for performance reasons).
  """
  loss = 0.
  batch_inits = list(range(0, len(experiences.advantages), batch_size))
  # TODO(dsuo): convert to jax.lax.scan.
  for b in batch_inits:
    batched_exps = jax.tree_map(lambda x: x[b:b + batch_size], experiences)
    grad_fn = jax.value_and_grad(ppo_loss_fn)
    l, grads = grad_fn(agent, batched_exps, clip_param, vf_coeff, entropy_coeff)
    loss += l
    updates, optim_state = optim.update(grads, optim_state, agent)
    agent = optax.apply_updates(agent, updates)
  return loss, agent, optim_state


def train(env, agent, reward_fn, horizon, config, workdir=None):
  """Main training loop.

  config
    - num_episodes
    - episodes_per_eval
    - num_agents
    - batch_size: batches are in time steps
    - decaying_lr_and_clip_param
    - clip_param
    - gamma
    - lambda_
    - vf_coeff
    - entropy_coeff

  episodes
    parallel agents taking steps (to horizon)
  """
  print(config)
  if workdir is not None:
    writer = metric_writers.create_default_writer(
        logdir=workdir, just_logging=jax.process_index() != 0)
    writer.write_hparams(dict(config))

  # config
  episodes_per_eval = 10

  optim = optax.adam(learning_rate=config.learning_rate)
  optim_state = optim.init(agent)

  for episode in range(config.num_episodes):
    # Bookkeeping and testing.
    if (episode + 1) % episodes_per_eval == 0:
      state, obs = env.reset()
      # TODO(dsuo): testing currently takes random action
      rollout = ac_rollout(env, state, obs, agent, 0, horizon, reward_fn,
                           config.gamma, config.lambda_)
      score = rollout.rewards.mean()
      if workdir is not None:
        writer.write_scalars(episode, {"test_score": score})
      print(f"TESTING episode {episode}:")
      print(f"  - score: {score}")

    alpha = 1. - episode / config.num_episodes if config.decaying_lr_and_clip_param else 1.
    clip_param = config.clip_param * alpha

    # collect experience of all agents.
    experiences = ac_get_experience(agent, env, horizon, config.num_agents,
                                    config.gamma, config.lambda_, reward_fn)
    score = experiences.rewards.mean()

    for epoch in range(config.num_epochs):
      es = jax.tree_map(
          lambda x: jax.random.permutation(jax.random.PRNGKey(1), x),
          experiences)
      loss, agent, optim_state = train_step(optim, optim_state, agent, es,
                                            config.batch_size, clip_param,
                                            config.vf_coeff,
                                            config.entropy_coeff)
      print(f"TRAINING: episode {episode}, epoch {epoch}:")
      print(f"  - loss: {loss}")
      print(f"  - score: {score}")

    if workdir is not None:
      writer.write_scalars(episode, {"train_loss": loss, "train_score": score})
  return optim_state, agent
