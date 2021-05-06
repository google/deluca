# `deluca`

Performant, differentiable reinforcement learning

## Notes
1. This is pre-alpha software and is undergoing a number of core changes. Updates to follow.
2. `deluca` is currently implemented as a minimal Python namespace package. It
   will contain environments and agents that *only* depend on NumPy, along with
   utilities for benchmarking, visualization, etc.
3. The remainder of environments are developed in separate plugins (also Python
   namespace packages)
   - [`deluca-jax`](https://github.com/MinRegret/deluca-jax): differentiable environments and relevant agents implemented
     using `jax`
   - [`deluca-lung`](https://pypi.org/project/deluca-lung/): differentiable lung simulators and relevant agents
     implemented in `PyTorch`
4. Documentation forthcoming!

[![pypi](https://badgen.net/pypi/v/deluca)](https://pypi.org/project/deluca/)
[![pyversions](https://raw.githubusercontent.com/MinRegret/deluca/dev/.github/badges/python_versions.svg)](https://pypi.org/project/deluca)
[![security: bandit](https://raw.githubusercontent.com/MinRegret/deluca/dev/.github/badges/bandit.svg)](https://github.com/PyCQA/bandit)
[![Code style: black](https://raw.githubusercontent.com/MinRegret/deluca/dev/.github/badges/black.svg)](https://github.com/psf/black)
[![License: Apache 2.0](https://raw.githubusercontent.com/MinRegret/deluca/dev/.github/badges/apache.svg)](https://github.com/MinRegret/deluca/blob/dev/LICENSE)

[![build](https://github.com/MinRegret/deluca/workflows/build/badge.svg)](https://github.com/MinRegret/deluca/actions)
[![coverage](https://badgen.net/codecov/c/github/MinRegret/deluca)](https://codecov.io/github/MinRegret/deluca)
[![Documentation Status](https://readthedocs.org/projects/deluca/badge/?version=latest)](https://deluca.readthedocs.io/en/latest/?badge=latest)
[![doc_coverage](https://raw.githubusercontent.com/MinRegret/deluca/dev/.github/badges/docstring_coverage.svg)](https://github.com/MinRegret/deluca)

![deluca](https://raw.githubusercontent.com/MinRegret/deluca/dev/assets/img/deluca.svg?token=AAURLVRRLKHPK4VELPKH6X27RW5LC)
