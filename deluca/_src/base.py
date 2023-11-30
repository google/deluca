# Copyright 2023 The Deluca Authors.
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

"""Base interfaces and datatypes.

NOTES
- Agents are not, generally speaking, interchangeable and neither are
  environments.
- Agents and environments likely need to know how to interpret each others'
  inputs and outputs.
- Any notion of an API is a loose one (as evidenced by the number of `*args` and
  `**kwargs` in `Protocol` signatures); this API serves a more minor ergonomic
  role and provides some consistency.
- There may be subsets of environments (e.g., lungs) or agents that can have a
  more specific API. This should probably be encouraged.

Questions
- Should we be able to further update agent/actionler state when getting the
  action of the agent based on its state?
- Should `Environment`s action what is observable or leave that to `Agent`s to
  responsibly access only what they should be allowed to?
- Should `AgentActionFn`s take `AgentState` and `EnvironmentState` or be more
  permissive?
- Should we have a `state_dim` and `action_dim` property for `EnvironmentState`?
- Is there any point to having a registry of agents or environments if they're
  just `NamedTuple`s?
"""

from typing import Any, NamedTuple, Protocol


AgentState = Any
AgentRegistry = {}
Action = Any
EnvironmentState = Any
EnvironmentRegistry = {}


class CostFn(Protocol):
  """A callable type for the `cost` function of a task."""

  def __call__(
      self, state: EnvironmentState, action: Action, *args, **kwargs
  ) -> float:
    """The `cost` function.

    Args:
      state: An `EnvironmentState` object.
      action: An `Action` object.
      *args: Arbitrary positional arguments.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      The cost of a task.
    """


class AgentInitFn(Protocol):
  """A callable type for the `init` step of an `Agent`.

  The `init` step takes `*args` and `**kwargs` to construct an arbitrary initial
  `state` for the Agent. This may hold statistics of the past updates or any
  other non-static information.
  """

  def __call__(self, *args, **kwargs) -> AgentState:
    """The `init` function.

    Args:
      *args: Arbitrary positional arguments.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      The initial state of an Agent.
    """


class AgentActionFn(Protocol):
  """A callable type for the `action` step of an `Agent`.

  The `action` step takes an `AgentState`, an `EnvironmentState`, and `*args` /
  `**kwargs` and produces a possibly updated `AgentState` and a `Action`.
  """

  def __call__(
      self, state: AgentState, obs: EnvironmentState, *args, **kwargs
  ) -> tuple[AgentState, Action]:
    """The `action` function.

    Args:
      state: An `AgentState` object to update.
      obs: An `EnvironmentState` object.
      *args: Arbitrary positional arguments.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      An updated `AgentState` for an `Agent` and a `Action`.
    """


class AgentUpdateFn(Protocol):
  """A callable type for the `update` step of an `Agent`.

  The `update` step takes an `AgentState` and `*args` / `**kwargs`, typically
  from the `Environment` and produces a new `AgentState`.
  """

  def __call__(self, state: AgentState, *args, **kwargs) -> AgentState:
    """The `update` function.

    Args:
      state: An `AgentState` object to update.
      *args: Arbitrary positional arguments.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      An updated `AgentState` for an `Agent`.
    """


class Agent(NamedTuple):
  """A set of pure functions implementing an agent's behavior."""

  init: AgentInitFn
  action: AgentActionFn
  update: AgentUpdateFn

  def __init_subclass__(cls, *args, **kwargs) -> None:
    """Adds a new `Agent` subclass to the registry.

    Args:
      *args: Arbitrary positional arguments.
      **kwargs: Arbitrary keyword arguments.
    """
    name = cls.__name__
    if name in AgentRegistry:
      raise ValueError(f'Agent {name} already exists.')
    AgentRegistry[name] = cls


class EnvironmentInitFn(Protocol):
  """A callable type for the `init` step of an `Environment`.

  The `init` step takes `*args` and `**kwargs` to construct an arbitrary initial
  `state` for the Environment. This may hold statistics of the past updates or
  any
  other non-static information.
  """

  def __call__(self, *args, **kwargs) -> EnvironmentState:
    """The `init` function.

    Args:
      *args: Arbitrary positional arguments.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      The initial state of an Environment.
    """


class EnvironmentUpdateFn(Protocol):
  """A callable type for the `update` step of an `Environment`.

  The `update` step takes an `EnvironmentState` and `*args` / `**kwargs`,
  typically
  from the `Environment` and produces a new `EnvironmentState`.
  """

  def __call__(
      self, state: EnvironmentState, action: Any, *args, **kwargs
  ) -> EnvironmentState:
    """The `update` function.

    Args:
      state: An `EnvironmentState` object to update.
      action: An `Action` object.
      *args: Arbitrary positional arguments.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      An updated `EnvironmentState` for an `Environment`.
    """


class Environment(NamedTuple):
  """A set of pure functions implementing an agent's behavior."""

  init: EnvironmentInitFn
  update: EnvironmentUpdateFn

  def __init_subclass__(cls, *args, **kwargs) -> None:
    """Adds a new `Environment` subclass to the registry.

    Args:
      *args: Arbitrary positional arguments.
      **kwargs: Arbitrary keyword arguments.
    """
    name = cls.__name__
    if name in EnvironmentRegistry:
      raise ValueError(f'Environment {name} already exists.')
    EnvironmentRegistry[name] = cls