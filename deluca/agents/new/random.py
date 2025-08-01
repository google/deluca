from deluca.agents.new.core import Agent

import jax
import jax.numpy as jnp


class SimpleRandom(Agent):
    def __call__(self, history, rng: jax.Array) -> jax.Array:
        return jax.random.normal(rng, (self.action_dim_out,1))
    
    def train_step(self, env, memory, rng: jax.Array):
        return jnp.zeros(1), memory
    
    def train(self, env, N, T, rng: jax.Array):
        return jnp.zeros(1)