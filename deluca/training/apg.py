# Copyright 2022 The Deluca Authors.
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
import time

# pylint:disable=invalid-name

Rollout = collections.namedtuple("Rollout", [
    "agent_states", "actions", "losses"
])

@functools.partial(
    jax.jit, static_argnums=(
        5,
        6,
    ))
def apg_rollout(env, env_start_state, env_init_obs, agent, agent_start_state, unroll_length, loss_func):
  """Comment."""
  def rollout_step(inp, counter):
    env_state, env_obs, agent_state = inp
    agent_next_state, action = agent(agent_state, env_obs)
    env_next_state, env_next_obs = env(env_state, action)
    loss = loss_func(env_state, env_obs, action, env, counter)
    return (env_next_state, env_next_obs,
            agent_next_state), (env_next_state, env_next_obs, agent_next_state,
                                action, loss)

  _, t = jax.lax.scan(rollout_step,
                      (env_start_state, env_init_obs, agent_start_state),
                      jnp.arange(unroll_length))
  return Rollout(t[2], t[3], t[4])


@functools.partial(
    jax.jit, static_argnums=(
        5,
        6,
    ))
def apg_parallel_rollouts(env, env_start_states, env_init_obs, agent,
                          agent_start_states, unroll_length, loss_func):
  """Parallelize rollouts."""
  # if agent_seeds == None:
  #   agent_seeds = jnp.arange(num)
  # env_start_states, env_init_obs = env.reset()
  all_rollouts = jax.vmap(apg_rollout,
                          (None, 0, 0, None, 0, None, None))(
                              env, env_start_states, env_init_obs, agent,
                              agent_start_states, unroll_length, loss_func)
  return all_rollouts


def apg_loss(agent, env, env_start_states, env_init_obs, agent_init_states,
             unroll_length, loss_func):
  """Serialize the experiences."""
  rollouts = apg_parallel_rollouts(env, env_start_states, env_init_obs, agent,
                                  agent_init_states, unroll_length, loss_func)
  return rollouts.losses.mean()


@functools.partial(
    jax.jit, static_argnums=(
        0,
        1,
        8,
        9,
    ))
def train_chunk(chunk_size, optim, optim_state, agent, env, env_start_states,
                env_init_obs, agent_init_states, unroll_length, loss_func):
  """Compilable train step.

  Runs an entire epoch of training (i.e. the loop over minibatches within
  an epoch is included here for performance reasons).
  """
  @jax.jit
  def single_step(carry, inp):
    agent, optim_state = carry
    grad_fn = jax.value_and_grad(apg_loss)
    l, grads = grad_fn(agent, env, env_start_states, env_init_obs, agent_init_states, unroll_length, loss_func)
    updates, optim_state = optim.update(grads, optim_state, agent)
    agent = optax.apply_updates(agent, updates)
    return (agent, optim_state), l

  (agent, optim_state), losses = jax.lax.scan(single_step, (agent, optim_state), None, length=chunk_size)
  return agent, optim_state, losses


def train(env, agent, loss_func, horizon, config, workdir=None):
  """Main training loop.

  config
    - num_episodes
    - episodes_per_eval
    - training_env_batch_size
    - eval_env_batch_size = 32

    - optimizer
    - learning_rate

    - seed = 1
  """
  print(config)
  if workdir is not None:
    writer = metric_writers.create_default_writer(
        logdir=workdir, just_logging=jax.process_index() != 0)
    writer.write_hparams(dict(config))

  key = jax.random.PRNGKey(config.seed)
  key_train_agent, key_eval_agent, key_train_env, key_eval_env, key_train, key = jax.random.split(key, 6)
  key_train_envs = jax.random.split(key_train_env, config.training_env_batch_size)
  key_train_agents = jax.random.split(key_train_agent, config.training_env_batch_size)
  key_eval_envs = jax.random.split(key_eval_env, config.eval_env_batch_size)
  key_eval_agents = jax.random.split(key_eval_agent, config.eval_env_batch_size)

  #TODO(danielsuo): The following vmap code does not work.
  train_env_start_states, train_env_init_obs = jax.vmap(env.init)(key_train_envs)
  eval_env_start_states, eval_env_init_obs = jax.vmap(env.init)(key_eval_envs)
  print(train_env_start_states)
  print(train_env_init_obs)

  # qtrain_init_list = list(map(env.init, key_train_envs))
  # qtrain_env_start_states = [a for (a,_) in qtrain_init_list]
  # qtrain_env_init_obs = [b for (_,b) in qtrain_init_list]
  # print(qtrain_env_start_states)
  # print(qtrain_env_init_obs)
  # eval_init_list = list(map(env.init, key_eval_envs))
  # eval_env_start_states = [a for (a,_) in eval_init_list]
  # eval_env_init_obs = [b for (_,b) in eval_init_list]

  train_agent_start_states = jax.vmap(agent.init)(key_train_agents)
  eval_agent_start_states = jax.vmap(agent.init)(key_eval_agents)

  if config.optimizer == "Adam":
    optim = optax.adam(learning_rate=config.learning_rate)
  else:
    # default is SGD
    optim = optax.sgd(learning_rate=config.learning_rate)
  optim_state = optim.init(agent)

  for episode in range(0, config.num_episodes, config.episodes_per_eval):
    # Eval Step
    tt = time.time()
    eval_rollouts = apg_parallel_rollouts(env, eval_env_start_states,
                                          eval_env_init_obs, agent,
                                          eval_agent_start_states, horizon,
                                          loss_func)
    test_score = eval_rollouts.losses.mean()
    print(f"TESTING episode {episode} - score:{test_score} - time:{time.time()-tt}")

    # Training Step
    tt = time.time()
    agent, optim_state, losses = train_chunk(config.episodes_per_eval, optim,
                                             optim_state, agent, env,
                                             train_env_start_states,
                                             train_env_init_obs,
                                             train_agent_start_states, horizon,
                                             loss_func)
    done_eps = episode + config.episodes_per_eval - 1
    print(f"TRAINING: episode {done_eps} - score:{losses[0]} - time {time.time() - tt}")

    if workdir is not None:
      for (i, loss) in enumerate(reversed(losses)):
        writer.write_scalars(episode+i, {"train_score": loss})
      writer.write_scalars(episode, {"test_score": test_score})
  return optim_state, agent
