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


class _ActiveInterceptor:
  """Tracks an active interceptor and its enabled state in a thread."""

  __slots__ = ("interceptor", "enabled")

  def __init__(self, interceptor: Interceptor, enabled: bool = True):
    self.interceptor = interceptor
    self.enabled = enabled


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
  be a process-wide singleton and be protected by a lock for administrative
  operations (such as installing or removing global module/bytecode patches).

  When an interceptor is installed, this class will

  1. Patch the Python attributes (via `setattr`). This will affect the entire
     process.
  2. Activate the interceptor for the current thread.

  When an intercepted function is called, this manager determines which active
  thread context should handle the call, providing thread isolation and
  recursion protection.
  """

  def __init__(self):
    # Administrative operations (installing/removing patches) are protected by
    # this lock.
    self._lock = threading.Lock()

    # A mapping from intercepted function names to the original functions. This
    # is used to call the original functions when inside the interceptor, and
    # to restore the original functions when the interception is removed.
    self._original_fns: dict[str, types.FunctionType] = {}

    # Globally installed interceptors: interceptor_id -> Interceptor.
    self._global_interceptors: dict[int, Interceptor] = {}

    # Reference count of active threads per interceptor:
    # interceptor_id -> count.
    self._interceptor_refcounts: dict[int, int] = {}

    # Thread-local storage for per-thread active/disabled interceptor state.
    self._local = threading.local()

  def _get_thread_interceptors(self) -> dict[int, _ActiveInterceptor]:
    """Returns the thread-local dictionary of active interceptors."""
    interceptors = getattr(self._local, "interceptors", None)
    if interceptors is None:
      interceptors = {}
      self._local.interceptors = interceptors
    return interceptors

  def is_active(self, interceptor: Interceptor) -> bool:
    """Returns whether the interceptor is active for the current thread."""
    return interceptor.id in self._get_thread_interceptors()

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
    thread_interceptors = self._get_thread_interceptors()
    if interceptor_id in thread_interceptors:
      raise ValueError(f"{interceptor_id} already activated in {this_thread}.")
    thread_interceptors[interceptor_id] = _ActiveInterceptor(
        interceptor=interceptor, enabled=True
    )
    with self._lock:
      if interceptor_id in self._global_interceptors:
        self._interceptor_refcounts[interceptor_id] += 1
        return
      # Register the interception for all the intercepted names.
      registered = []
      try:
        self._global_interceptors[interceptor_id] = interceptor
        self._interceptor_refcounts[interceptor_id] = 1
        for name in interceptor:
          self._maybe_apply_interception(name)
          registered.append(name)
      except ValueError as e:
        # Uninstall to ensure data consistency if a registration fails.
        del thread_interceptors[interceptor_id]
        self._global_interceptors.pop(interceptor_id, None)
        self._interceptor_refcounts.pop(interceptor_id, None)
        for name in registered:
          self._maybe_remove_interception(name)
        raise e

  def deactivate_interceptor(self, interceptor: Interceptor):
    """Deactivates the interceptor for the current thread."""
    interceptor_id = interceptor.id
    this_thread = threading.get_ident()
    thread_interceptors = self._get_thread_interceptors()
    # The current thread must already be intercepted.
    if interceptor_id not in thread_interceptors:
      raise ValueError(f"{interceptor_id} not activated for {this_thread}.")
    if not thread_interceptors[interceptor_id].enabled:
      raise ValueError(f"{interceptor_id} is disabled for {this_thread}.")
    del thread_interceptors[interceptor_id]
    with self._lock:
      self._interceptor_refcounts[interceptor_id] -= 1
      if self._interceptor_refcounts[interceptor_id] == 0:
        del self._interceptor_refcounts[interceptor_id]
        removed_interceptor = self._global_interceptors.pop(interceptor_id)
        for name in removed_interceptor:
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
      if aux_data.get(obj.__code__, "fn", None) is not None:  # pyrefly: ignore[missing-attribute]
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
    if any(
        name in interceptor
        for interceptor in self._global_interceptors.values()
    ):
      return
    obj, attr = _resolve_path(name)
    if attr == "__code__":
      # Special handling for code objects.
      setattr(obj, attr, self._original_fns.pop(name).__code__)
    else:
      setattr(obj, attr, self._original_fns.pop(name))

  def _on_intercepted_called(self, name: str, args, kwargs):
    """Called when an intercepted function is called."""
    # Locate the interceptor to disable and the handler to call using
    # thread-local storage.
    target_entry = None
    thread_interceptors = self._get_thread_interceptors()
    # We apply the earliest interceptor first. This creates a behavior that
    # a later-installed interceptor will be called inside an earlier-installed
    # interceptor.
    for entry in thread_interceptors.values():
      if entry.enabled and name in entry.interceptor:
        target_entry = entry
        break

    if target_entry is None:
      return self._original_fns[name](*args, **kwargs)

    # Disable this interceptor for the current thread to avoid recursion.
    target_entry.enabled = False
    try:
      return target_entry.interceptor[name](*args, **kwargs)
    finally:
      target_entry.enabled = True

  def disable_interception(self) -> list[int]:
    """Disables all interceptions for the current thread and returns the list of disabled interceptors."""
    disabled_interceptor_ids = []
    for entry in self._get_thread_interceptors().values():
      if entry.enabled:
        entry.enabled = False
        disabled_interceptor_ids.append(entry.interceptor.id)
    return disabled_interceptor_ids

  def enable_interception(self, interceptor_ids: list[int]):
    """Enables the given interceptions for the current thread."""
    this_thread = threading.get_ident()
    thread_interceptors = self._get_thread_interceptors()
    for iid in interceptor_ids:
      entry = thread_interceptors.get(iid)
      if entry is None or entry.enabled:
        raise ValueError(f"{iid} is already enabled for {this_thread}.")
      entry.enabled = True


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

    fn = aux_data.get(inspect.currentframe().f_code, "fn")  # pyrefly: ignore[missing-attribute]
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
