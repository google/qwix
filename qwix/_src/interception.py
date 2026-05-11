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
"""Intercepts Python functions by patching."""

import collections.abc
import contextlib
import dataclasses
import functools
import sys
import threading
import types
from typing import Any, Callable, TypeAlias

import jax
from qwix._src import aux_data

Function: TypeAlias = Callable[..., Any]

# The key used to intercept jax._src.core.Primitive.bind.
PRIMITIVE_BIND_KEY = "jax._src.core.Primitive.bind"


@dataclasses.dataclass(frozen=True)
class Interceptor(collections.abc.Mapping[str, Function]):
  """A container for interception mappings with a stable identity.

  Provides an explicit `id` to guarantee a stable hashcode during JIT traces,
  even if the enclosed mapping functions are dynamically wrapped or modified.

  Attributes:
    mapping: A dictionary mapping the target function name (e.g. module path or
      op name) to its intercepted function implementation.
    id: A stable, unique integer identifier (typically derived from the
      quantization provider instance) that guarantees consistent JIT cache keys.
  """

  mapping: dict[str, Function]
  id: int

  def __getitem__(self, key: str) -> Function:
    return self.mapping[key]

  def __iter__(self):
    return iter(self.mapping)

  def __len__(self) -> int:
    return len(self.mapping)

  def __hash__(self) -> int:
    return self.id


def _preprocess_interceptor(
    interceptor: Interceptor, disable_jit: bool
) -> Interceptor:
  """Preprocesses the interceptor by rewriting keys.

  This function prepares the interceptor keys for the `_InterceptionManager` in
  order to do a little bit more than just monkey-patching the attributes of the
  objects. The keys are rewritten based on the type of the target function and
  the `disable_jit` flag:

  - **Functions (No Freevars):** Rewritten to use `.__code__` to target
    bytecode patching instead of module attribute patching (helps with aliases).
  - **JAX `PjitFunction`:**
    - When `disable_jit=True`: Rewritten to use `._fun` (and eventually
      `._fun.__code__`) to bypass JAX's C++ dispatch and target the inner
      Python bytecode.
    - When `disable_jit=False`: Keys remain unchanged to use standard module
      attribute patching, ensuring the JIT compilation and caching layers are
      properly preserved.
  - **Everything Else:** Keys remain as-is, which tells the manager to use
    standard module-level attribute monkey-patching.

  Args:
    interceptor: An Interceptor object containing the mapping and ID.
    disable_jit: Whether JIT is disabled. Affects how `PjitFunction`s are
      handled.

  Returns:
    A new Interceptor object with preprocessed keys.
  """
  interceptor_mapping = dict(interceptor)
  # Preprocess the interceptor for JAX-specific and alias-aware rewrites.
  for name in list(interceptor_mapping):
    # Resolve path to the actual object (e.g., PjitFunction or FunctionType).
    target_path = name
    function_to_modify = get_attribute(target_path)

    # 1. Rewrite `PjitFunction` to its inner Python function `_fun`
    #    only when JIT is disabled; otherwise, use standard attribute patching.
    if disable_jit and isinstance(
        function_to_modify,
        jax._src.lib._jax.PjitFunction,  # pylint: disable=protected-access
    ):
      new_path = target_path + "._fun"
      interceptor_mapping[new_path] = interceptor_mapping.pop(target_path)
      target_path = new_path
      function_to_modify = function_to_modify._fun  # pylint: disable=protected-access  # pyrefly: ignore[missing-attribute]

    # 2. Rewrite `Function` to its code object for bytecode patching.
    if (
        isinstance(function_to_modify, types.FunctionType)
        and not function_to_modify.__code__.co_freevars
    ):
      interceptor_mapping[target_path + ".__code__"] = interceptor_mapping.pop(
          target_path
      )

  return Interceptor(mapping=interceptor_mapping, id=interceptor.id)


def wrap_func_intercepted(
    func: Function,
    get_interceptor: Callable[[], Interceptor],
    *,
    disable_jit: bool,
    input_transform: Callable[[Any, Any], tuple[Any, Any]] = lambda *x: x,
    output_transform: Callable[[Any], Any] = lambda x: x,
    should_intercept: Callable[[], bool] = lambda: True,
) -> Function:
  """Wraps a function to execute within an active interception scope.

  This returns a wrapped version of `func`. When called, it activates the
  interceptors for the duration of the call. The scope is thread-local (isolated
  per thread) and non-recursive (safe to call intercepted functions inside
  handlers without infinite loops).

  Args:
    func: The function to wrap.
    get_interceptor: A function/factory that returns an `Interceptor` object.
      Using a factory function instead of a static instance defers any dynamic
      attribute resolution or circular import checks until interception time,
      while also allowing subclasses to cleanly append custom hooks.
    disable_jit: Whether to disable JIT when calling the wrapped function.
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
    interceptor = _preprocess_interceptor(get_interceptor(), disable_jit)
    # If the interceptor is already active for the current thread or
    # should_intercept() returns False, return the original function.
    if interception_manager.is_active(interceptor) or not should_intercept():
      return func(*args, **kwargs)

    # Apply the input transform.
    args, kwargs = input_transform(args, kwargs)

    # Scope the function execution within active interception.
    interception_manager.activate_interceptor(interceptor)
    context_manager = (
        jax.disable_jit()
        if (not jax.config.jax_disable_jit and disable_jit)  # pyrefly: ignore[missing-attribute]
        else contextlib.nullcontext()
    )
    try:
      with context_manager:
        output = func(*args, **kwargs)
    finally:
      interception_manager.deactivate_interceptor(interceptor)

    # Apply the output transform.
    return output_transform(output)

  return wrapper


class _InterceptionManager:
  """Manages the lifecycle of function interceptions.

  This class implements
  * Thread-local interception: the installation of an interceptor in a thread
    won't affect other threads.
  * Multi-thread support: it's possible to install the same interceptor from
    multiple threads.
  * Non-recursive interception: inside an interceptor function, the same
    interceptor is disabled so that we won't have recursive interception.
  * Nested interception: it's possible to install different interceptors in a
    nested way.

  Since patching a Python module is a global state mutation, this class has to
  be a process-wide singleton and be protected by a lock.

  When an interceptor is installed, this class will

  1. Patch the Python attributes (via `setattr`). This will affect the entire
     process.
  2. Activate the interceptor for the current thread.

  When an intercepted function is called, this manager determines which active
  thread context should handle the call, providing thread isolation and
  recursion protection.
  """

  def __init__(self):
    # Accessing following variables must be protected by this lock.
    self._lock = threading.Lock()

    # A mapping from intercepted function names to the original functions. This
    # is used to call the original functions when inside the interceptor, and
    # to restore the original functions when the interception is removed.
    self._original_fns: dict[str, types.FunctionType] = {}

    # A list of interceptors, in installation order. We don't expect too many
    # interceptors, so it's fine to use a list for all the interceptors, and
    # it's fine to iterate over all of them in _on_intercepted_called.
    self._interceptors: list[Interceptor] = []

    # A dict of { (thread_id, interceptor_id): enabled } indicating whether the
    # current thread should apply the interception. The interception only
    # applies when the key (thread_id, interceptor_id) is in the dict and the
    # value enabled is True.
    self._intercepted_threads: dict[tuple[int, int], bool] = {}

  def is_active(self, interceptor: Interceptor) -> bool:
    """Returns whether the interceptor is active for the current thread."""
    this_thread = threading.get_ident()
    with self._lock:
      return (this_thread, interceptor.id) in self._intercepted_threads

  def activate_interceptor(self, interceptor: Interceptor):
    """Activates the interceptor for the current thread.

    If this interceptor is not yet installed (from other threads), this method
    also triggers the global patching of the relevant Python modules.

    Args:
      interceptor: An Interceptor mapping object containing replacement
        functions and its stable ID.
    """
    interceptor_id = interceptor.id
    this_thread = threading.get_ident()
    with self._lock:
      if (this_thread, interceptor_id) in self._intercepted_threads:
        raise ValueError(
            f"{interceptor_id} already activated in {this_thread}."
        )
      self._intercepted_threads[(this_thread, interceptor_id)] = True
      # Check if the interceptor is already installed by other threads.
      if any(interceptor_id == i.id for i in self._interceptors):
        return
      self._interceptors.append(interceptor)
      # Register the interception for all the intercepted names.
      registered = []
      try:
        for name in interceptor:
          self._maybe_apply_interception(name)
          registered.append(name)
      except ValueError as e:
        # Uninstall to ensure data consistency if an registration fails.
        del self._intercepted_threads[(this_thread, interceptor_id)]
        self._interceptors.pop()
        for name in registered:
          self._maybe_remove_interception(name)
        raise e

  def deactivate_interceptor(self, interceptor: Interceptor):
    """Deactivates the interceptor for the current thread."""
    interceptor_id = interceptor.id
    this_thread = threading.get_ident()
    with self._lock:
      # The current thread must already be intercepted.
      if (this_thread, interceptor_id) not in self._intercepted_threads:
        raise ValueError(f"{interceptor_id} not activated for {this_thread}.")
      if not self._intercepted_threads[(this_thread, interceptor_id)]:
        raise ValueError(f"{interceptor_id} is disabled for {this_thread}.")
      del self._intercepted_threads[(this_thread, interceptor_id)]
      # Check if any other threads are still using this interceptor.
      if any(interceptor_id == iid for _, iid in self._intercepted_threads):
        return
      # Remove the interceptor. This is inefficient but we don't expect too
      # many interceptors.
      interceptor_index = next(
          i
          for i, interceptor in enumerate(self._interceptors)
          if interceptor.id == interceptor_id
      )
      interceptor = self._interceptors.pop(interceptor_index)
      for name in interceptor:
        self._maybe_remove_interception(name)

  def _maybe_apply_interception(self, name: str):
    """Tries to patch a specific Python attribute.

    This method resolves the dot-separated path and applies the override logic
    (either bytecode-level or attribute-level) to the global environment.

    * Do nothing if the interception is already applied.
    * Raise ValueError if we accidentally apply interceptions for different
      aliases of the same function.

    Calling this function must be protected by self._lock.

    Args:
      name: The name of the function to intercept.
    """
    if name in self._original_fns:
      return
    obj, attr = _resolve_path(name)
    # It's unclear but we cannot return a functools.partial object here,
    # otherwise the test_intercept_class_method will fail.
    replacing_fn = lambda *args, **kwargs: self._on_intercepted_called(
        name, args, kwargs
    )
    if attr == "__code__":  # special handling for code objects.
      # Check if we accidentally register different aliases for the same object.
      if aux_data.get(obj.__code__, "fn", None) is not None:  # pytype: disable=attribute-error
        raise ValueError(f"Intercept aliases for the same object: {name}.")
      # _copy_fn is needed because obj will be modified below.
      self._original_fns[name] = _copy_fn(obj)
      setattr(obj, attr, _fn_to_code(replacing_fn))
    else:
      original_fn = getattr(obj, attr)
      # Check if we accidentally register different aliases for the same object.
      if aux_data.get(original_fn, "intercepted", False):
        raise ValueError(f"Intercept aliases for the same object: {name}")
      aux_data.set(replacing_fn, "intercepted", True)
      self._original_fns[name] = original_fn
      setattr(obj, attr, replacing_fn)

  def _maybe_remove_interception(self, name: str):
    """Tries to remove the patch for one name.

    * Raise KeyError if the interception is not applied.
    * Do nothing if there is any other interceptor still needing the same
      interception.

    Calling this function must be protected by self._lock.

    Args:
      name: The name of the function to un-intercept.
    """
    if any(name in interceptor for interceptor in self._interceptors):
      return
    obj, attr = _resolve_path(name)
    if attr == "__code__":
      # Special handling for code objects.
      setattr(obj, attr, self._original_fns.pop(name).__code__)
    else:
      setattr(obj, attr, self._original_fns.pop(name))

  def _on_intercepted_called(self, name: str, args, kwargs):
    """Called when an intercepted function is called."""
    # Locate the interceptor to disable and the handler to call.
    this_thread = threading.get_ident()
    interceptor_to_use = None
    with self._lock:
      # We apply the earliest interceptor first. This creates a behavior that
      # a later-installed interceptor will be called inside an earlier-installed
      # interceptor.
      for interceptor in self._interceptors:
        interceptor_id = interceptor.id
        if (
            self._intercepted_threads.get((this_thread, interceptor_id), False)
            and name in interceptor
        ):
          # Disable this interceptor for the current thread to avoid recursion.
          self._intercepted_threads[(this_thread, interceptor_id)] = False
          interceptor_to_use = interceptor_id, interceptor
          break

    if interceptor_to_use is None:
      return self._original_fns[name](*args, **kwargs)
    try:
      return interceptor_to_use[1][name](*args, **kwargs)
    finally:
      with self._lock:
        self._intercepted_threads[(this_thread, interceptor_to_use[0])] = True

  def disable_interception(self) -> list[int]:
    """Disables all interceptions for the current thread and returns the list of disabled interceptors."""
    this_thread = threading.get_ident()
    disabled_interceptor_ids = []
    with self._lock:
      for (tid, iid), enabled in self._intercepted_threads.items():
        if tid == this_thread and enabled:
          self._intercepted_threads[(tid, iid)] = False
          disabled_interceptor_ids.append(iid)
    return disabled_interceptor_ids

  def enable_interception(self, interceptor_ids: list[int]):
    """Enables the given interceptions for the current thread."""
    this_thread = threading.get_ident()
    with self._lock:
      for iid in interceptor_ids:
        if self._intercepted_threads[(this_thread, iid)]:
          raise ValueError(f"{iid} is already enabled for {this_thread}.")
        self._intercepted_threads[(this_thread, iid)] = True


interception_manager = _InterceptionManager()


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
    disabled_interceptor_ids = interception_manager.disable_interception()
    try:
      return fn(*args, **kwargs)
    finally:
      interception_manager.enable_interception(disabled_interceptor_ids)

  return wrapper


def get_attribute(name: str) -> Any:
  """Returns the attribute from the name."""
  obj, attr = _resolve_path(name)
  return getattr(obj, attr)


def has_attribute(name: str) -> bool:
  """Returns if the module exists and has the attribute."""
  try:
    get_attribute(name)
    return True
  except AttributeError:
    return False


def _resolve_path(name: str) -> tuple[Any, str]:
  """Resolves a dot-separated name into (parent_object, attribute_name)."""
  name_parts = name.split(".")
  if name_parts[0] not in sys.modules:
    raise AttributeError(f"Cannot find module: {name_parts[0]}")
  obj = sys.modules[name_parts[0]]
  for attr in name_parts[1:-1]:
    obj = getattr(obj, attr)
  return obj, name_parts[-1]
