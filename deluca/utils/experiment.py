# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""deluca/utils/experiment.py

Experiment decorator for running experiments in parallel
"""
import inspect
from collections.abc import Iterable
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import multiprocess.context as ctx
import numpy as np
import pathos

ctx._force_start_method("spawn")


def runner(args: Tuple[Callable, Dict[Any, Any]]) -> Any:
    """Runner function for process pool
    
    Args:
        args: A pair where the first element is the function to run and
        the second element is a dictionary of kwargs

    Returns:
        Any: the result of the function in the first element
    """
    return args[0](**args[1])


def product(*args: Any) -> List[Any]:
    """Recursive function to yield the cartesian product of a list of arguments

    Notes:
        * We don't use itertools.product because it will recursively examine its
        arguments and we want a tuple to stay a tuple in the resulting product
    """
    if args:
        for a in args[0]:
            for prod in product(*args[1:]) if args[1:] else ((),):
                yield (a,) + prod


class experiment:
    """Class decorator to run experiments"""

    def __init__(self, argnames: Union[List[str], str], arglists: List[Any]) -> None:
        """Initializes the experiment with argnames and arglists

        Args:
            argnames: List of argument names or comma-delimited string of names (as in pytest)
            arglists: A list of one or more tuples of values for each name in argnames
        """
        if isinstance(argnames, list):
            self._argnames = argnames
        else:
            self._argnames = [argname.strip() for argname in argnames.split(",")]

        # Validate that all arglists have the correct number of arguments
        # for arglist in arglists:
        # NOTE: Removed this check because it realizes generators passed in as args
        # arglist_length = 1 if "__len__" not in dir(arglist) else len(arglist)

        self._arglists = arglists
        self._spec = [(self._argnames, self._arglists)]

    def __call__(self, funcOrExp):
        """Decorator magic

        Notes:
            * Takes either a Callable or experiment object so we can chain decorators
        """

        # If we see an experiment, pass on the function to execute
        if hasattr(funcOrExp, "_func"):
            self._func = funcOrExp._func
            self._spec.extend(funcOrExp._spec)

        # If we see a callable, we know we're wrapping a naked function
        elif isinstance(funcOrExp, Callable):
            self._func = funcOrExp

        return self

    def _validate(self):
        """Validate an experiment's argnames and arglists against _func's arguments
        
        Notes:
            * Each decorated call to @experiment must specify arguments disjoint with
            every other call to @experiment (i.e., can't have a call "a,b,c" and "b")
            * The union of all arguments specified must be a subset of _func's arguments
            and a superset of _func's arguments without defaults
            * If _func has *args, we ignore
            * If _func has **kwargs, it will eat up any unused arguments
        """
        # Check for duplicate argument specifications
        argnames = set()
        for spec in self._spec:
            for argname in spec[0]:
                if argname in argnames:
                    raise ValueError("Found duplicate argname {} in {}".format(argname, spec))
                argnames.add(argname)

        func_fullargspec = inspect.getfullargspec(self._func)
        func_args = func_fullargspec.args
        func_args_without_defaults = (
            func_args
            if func_fullargspec.defaults is None
            else func_args[: -len(func_fullargspec.defaults)]
        )

        # NOTE: We don't use set operations because we want more descriptive error messages

        # Find any missing arguments
        missing_args = []
        for arg in func_args_without_defaults:
            if arg not in argnames:
                missing_args.append(arg)

        if len(missing_args) > 0:
            raise ValueError(
                "Arguments without defaults not found. Required: {}. Supplied: {}".format(
                    missing_args, argnames
                )
            )

        # Find any extra arguments
        extra_args = []
        if func_fullargspec.varkw is None:
            for arg in argnames:
                if arg not in func_args:
                    extra_args.append(arg)

        if len(extra_args) > 0:
            raise ValueError("Found unused arguments: {}".format(extra_args))

    def _generate_arglists(self):
        """Create a generator to pass function and arguments to a multiprocessing pool"""

        argnames = [item for sublist in [spec[0] for spec in self._spec] for item in sublist]
        arglists = product(*[spec[1] for spec in self._spec])
        for arglist in arglists:
            flattened = []

            # Flatten a list of lists and atoms
            for arg in arglist:
                if isinstance(arg, Iterable) and not isinstance(arg, str):
                    flattened.extend(arg)
                else:
                    flattened.append(arg)
            yield (self._func, {key: val for key, val in zip(argnames, flattened)})

    def run(self, processes=1, chunksize=1, tqdm=None):
        """Execute the experiment"""
        self._validate()

        single = lambda: map(runner, self._generate_arglists())  # noqa: E731
        num_tasks = np.product([len(spec[1]) for spec in self._spec])

        # TODO: Figure out why parallel hangs in pytest
        parallel = lambda: pathos.pools.ProcessPool(nodes=processes).imap(  # noqa: E731
            runner, self._generate_arglists(), chunksize=chunksize
        )
        process = single if processes == 1 else parallel
        return list(process() if tqdm is None else tqdm(process(), total=num_tasks))
