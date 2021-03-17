---
layout: default
title: Home
---
## Reinforcement learning with derivatives

[deluca](https://github.com/google/deluca) is a library modeled after [OpenAI Gym](https://github.com/openai/gym) that provides differentiable environments, control algorithms that take advantage of such environments, and benchmarking tools.

This software is currently in alpha and is changing rapidly. We have a paper describing the library available [here](https://arxiv.org/abs/2102.09968).

### Example without derivatives
```python
from deluca.envs import DelayLung
from deluca.agents import PID

env = DelayLung()
agent = PID([3.0, 4.0, 0.0])

for _ in range(1000):
  error = env.observation["error"]
  control = agent(error)
  obs, reward, done, info = env.step(control)
  if done:
    break
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
