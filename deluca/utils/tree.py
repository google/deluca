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

"""Jax tree utilities"""
import jax


def tree_transpose_leaves(pytrees):
  """Take list of pytrees and returns list of jnp.arrays

    Example:

      import jax
      import jax.numpy as jnp

      import deluca.core


      class Blah(deluca.Env):
        a: int = 1
        b: float = 2.0
        c: float = 3.

      blahs = [Blah.create(c=9), Blah.create(c=4)]
      blahs_t = tree_transpose_leaves(blahs)

      def func(args1, args2):
        blah1 = Blah(*args1)
        blah2 = Blah(*args2)
        return (blah1.a + blah1.b + blah1.c), (blah2.a * blah2.b * blah2.c)

      jax.vmap(func)((*blahs_t,), (*blahs_t,))
    """
  tree_def = jax.tree_util.tree_structure(pytrees[0])

  pytree = jax.tree_util.tree_transpose(
      outer_treedef=jax.tree_structure([0 for p in pytrees]),
      inner_treedef=jax.tree_structure(pytrees[0]),
      pytree_to_transpose=pytrees)

  leaves = tree_def.flatten_up_to(pytree)

  leaves = [jnp.array(leaf) for leaf in leaves]
  pytree = jax.tree_util.tree_unflatten(tree_def, leaves)

  return leaves


def tree_untranspose_leaves(treedef, leaves):
  """Take pytree of jnp.arrays and returns a list of pytrees"""
  return [
      treedef.unflatten([leaf[i]
                         for leaf in leaves])
      for i in range(len(leaves[0]))
  ]
