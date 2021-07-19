---
layout: default
title: Home
---
## Reinforcement learning with derivatives

[deluca](https://github.com/google/deluca) is a [`jax`](https://github.com/google/jax)-based library that provides differentiable environments, control algorithms that take advantage of such environments, and benchmarking tools.

This software is currently in alpha and is changing rapidly. We have a paper describing the library available [here](https://arxiv.org/abs/2102.09968).

### Getting started
`deluca` is a Python library that can be installed via `pip install deluca`.

### Example notebooks
We maintain a number of Jupyter notebooks and will continue to add more:
- A Regret Minimization Approach to Iterative Learning Control ([paper](https://arxiv.org/abs/2102.13478), [code](https://github.com/google/deluca-igpc))
- Machine Learning for Mechanical Ventilation Control ([paper](https://arxiv.org/abs/2102.06779), [code](https://github.com/google/deluca-lung))

### Example without derivatives
```python
import jax

from deluca.envs import Reacher
from deluca.agents import Random

env = Reacher.create()
env_state = env.init()

agent = Random.create(func=lambda key: jax.random.uniform(key, (env.action_dim,)))
agent_state = agent.init()

action = jnp.array([0.0, 0.0])
for i in range(100):
    env_state, obs = env(env_state, action)
    agent_state, action = agent(agent_state, obs)
```

### Citation
If you find our work helpful, please consider citing the associated paper:

```
@article{gradu2020deluca,
  title={Deluca--A Differentiable Control Library: Environments, Methods, and Benchmarking},
  author={Gradu, Paula and Hallman, John and Suo, Daniel and Yu, Alex and Agarwal, Naman and Ghai, Udaya and Singh, Karan and Zhang, Cyril and Majumdar, Anirudha and Hazan, Elad},
  journal={Differentiable Computer Vision, Graphics, and Physics in Machine Learning (Neurips 2020 Workshop)},
  year={2020}
}
```
