# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Intercepts Jax and Flax objects."""

import functools
import sys
import threading
import types
from typing import Any, Callable, Mapping, TypeAlias

from qwix._src import aux_data

Function: TypeAlias = Callable[..., Any]

# A dict of { (thread_id, interceptor_id): enabled }. The interception only
# applies when (thread_id, interceptor_id) is in the dict and enabled is True.
#
# The purposes are:
#   * (thread_id) avoids the interception leaking to other threads.
#   * (interceptor_id) avoids the same interceptor being applied twice in the
#     same thread.
#   * (enabled) allows the interceptor to be disabled for recursive calls.
_intercepted_threads: dict[tuple[int, int], bool] = {}


def wrap_func_intercepted(
    func: Function,
    get_intercept_map: Callable[[], Mapping[str, Function]],
    *,
    input_transform: Callable[[Any, Any], tuple[Any, Any]] = lambda *x: x,
    output_transform: Callable[[Any], Any] = lambda x: x,
    should_intercept: Callable[[], bool] = lambda: True,
) -> Function:
  """Wrap a function in a scope where functions in intercept_map are intercepted.

  The interception is both thread-local and non-recursive, which means
    * the interception of one thread won't affect another thread.
    * calling an intercepted function inside another intercepted function will
      not trigger the double interception.
    * the original functions will be restored when the replaced functions are
      called.

  Args:
    func: The function to wrap.
    get_intercept_map: A function that returns a mapping from function names to
      functions, e.g. {"jax.lax.dot_general": quantized_dot_general}.
    input_transform: A function to transform the input (args and kwargs) of the
      function.
    output_transform: A function to transform the output of the function.
    should_intercept: A predicate to decide whether the interception should be
      applied at all.

  Returns:
    A wrapped function.
  """

  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    this_thread = threading.get_ident()
    interceptor_id = id(get_intercept_map)

    # Skip if already intercepted.
    if (this_thread, interceptor_id) in _intercepted_threads:
      return func(*args, **kwargs)

    if not should_intercept():
      return func(*args, **kwargs)

    intercept_map = dict(get_intercept_map())
    # Ensure that the names in the intercept_map are valid and get the original
    # functions.
    origin_map = _install({name: None for name in intercept_map})

    # NOTE: the current implementation patches the code object of a function to
    # rather than the function itself. This allows us to patch all the aliases
    # of a function, e.g. jax.nn.gelu and flax.linen.gelu.
    #
    # These can be patched by replacing their code object:
    #   * global functions, e.g. jax.lax.sin
    #   * unbound methods, e.g. jax.numpy.ufunc.__call__. These don't have to be
    #     patched through code objects, but it's hard to distinguish them from
    #     global functions. These also includes static methods.
    # These cannot:
    #   * callable objects, e.g. a PjitFunction object jnp.sin.
    #   * bound methods, e.g. jax.lax.sin_p.bind. They have __code__ attribute,
    #     but cannot be set. These also includes class methods.
    #   * functions with freevars.
    #
    # It's also possible to patch a PjitFunction object by replacing its
    # _fun.__code__ attributes and wrapping everything in a jax.disable_jit()
    # scope. However, it's not widely needed and not supported yet.
    #
    # In case of patching the code object,
    #      origin_map: "jax.lax.sin.__code__" => the original code object
    #      origin_fns: "jax.lax.sin.__code__" => a copy of the original function
    #   intercept_map: "jax.lax.sin.__code__" => the intercepted function
    # and the intercepted function will be converted to a code object through
    # _fn_to_code when being prepared.

    # Modify the intercept_map to try to patch the code objects of functions.
    origin_fns = {}  # stores the original functions in case of patching code.
    for name, fn in list(origin_map.items()):
      if isinstance(fn, types.FunctionType) and not fn.__code__.co_freevars:
        origin_map.pop(name)
        origin_map[name + ".__code__"] = fn.__code__
        origin_fns[name + ".__code__"] = _copy_fn(fn)
        intercept_map[name + ".__code__"] = intercept_map.pop(name)

    # Wrap the intercepting functions so that unrelated threads and recursive
    # calls get redirected to the original functions. Also convert the functions
    # to code objects if needed.
    def prepare(fn, name):
      @functools.wraps(fn)
      def wrapper(*args, **kwargs):
        this_thread = threading.get_ident()
        enabled = _intercepted_threads.get((this_thread, interceptor_id), False)
        if not enabled:
          if name.endswith(".__code__"):
            return origin_fns[name](*args, **kwargs)
          return origin_map[name](*args, **kwargs)

        # Temporarily disable the interception for recursive calls.
        _intercepted_threads[(this_thread, interceptor_id)] = False
        try:
          return fn(*args, **kwargs)
        finally:
          _intercepted_threads[(this_thread, interceptor_id)] = True

      if name.endswith(".__code__"):
        # We're patching the code object of a function.
        return _fn_to_code(wrapper)
      return wrapper

    intercept_map = {
        name: prepare(fn, name) for name, fn in intercept_map.items()
    }

    # Apply the input transform.
    args, kwargs = input_transform(args, kwargs)

    _intercepted_threads[(this_thread, interceptor_id)] = True
    _install(intercept_map)
    try:
      output = func(*args, **kwargs)
    finally:
      _install(origin_map)
      _intercepted_threads.pop((this_thread, interceptor_id))

    # Apply the output transform.
    return output_transform(output)

  return wrapper


def _install(func_map: Mapping[str, Any]) -> dict[str, Any]:
  """Install the functions and return the original function map.

  If the given func_map contains None values, the installation will be skipped.
  This is useful for collecting the original functions.

  Args:
    func_map: A mapping from function names to functions.

  Returns:
    The original function map.
  """
  orig_func_map = {}
  for name, func in func_map.items():
    name_parts = name.split(".")
    if name_parts[0] in sys.modules:
      obj = sys.modules[name_parts[0]]
    else:
      raise ValueError(f"Cannot find module: {name_parts[0]}")
    for attr in name_parts[1:-1]:
      obj = getattr(obj, attr)
    attr = name_parts[-1]
    orig_func_map[name] = getattr(obj, attr)
    if func is not None:
      setattr(obj, attr, func)
  return orig_func_map


def _fn_to_code(fn: Function) -> types.CodeType:
  """Returns a code object that takes 0 freevars."""
  # To replace the code object of a global function, we need to create a new
  # code object that is not a closure, but still be able to remember the
  # original function. The trick is to associate the original function with the
  # code object itself with the aux_data module. When the code object is
  # executed, the code object itself can be accessed through
  # inspect.currentframe().f_code.

  def wrapper(*args, **kwargs):
    # Because the wrapper object can replace any code in other modules, so it
    # has to import the modules here.
    import inspect  # pylint: disable=g-import-not-at-top,redefined-outer-name,reimported
    from qwix._src import aux_data  # pylint: disable=g-import-not-at-top,redefined-outer-name,reimported

    fn = aux_data.get(inspect.currentframe().f_code, "fn")  # pytype: disable=attribute-error
    return fn(*args, **kwargs)

  code = wrapper.__code__.replace()  # this creates a new code object
  aux_data.set(code, "fn", fn)
  return code


def _copy_fn(fn: types.FunctionType) -> types.FunctionType:
  """Constructs a new function object with the same attributes as the given one."""
  fn_copy = types.FunctionType(fn.__code__, fn.__globals__)
  for field in (
      "__name__",
      "__qualname__",
      "__annotations__",
      "__defaults__",
      "__kwdefaults__",
      "__module__",
      "__doc__",
      "__dict__",
  ):
    if hasattr(fn, field):
      setattr(fn_copy, field, getattr(fn, field))
  return fn_copy


def disable_interceptions(fn):
  """Return the function with interceptions disabled when called."""

  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    this_thread = threading.get_ident()
    enabled_interceptors = []
    for key, enabled in _intercepted_threads.copy().items():
      if key[0] == this_thread and enabled:
        enabled_interceptors.append(key)
        _intercepted_threads[key] = False
    try:
      return fn(*args, **kwargs)
    finally:
      for key in enabled_interceptors:
        _intercepted_threads[key] = True

  return wrapper
