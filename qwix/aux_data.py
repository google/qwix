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
"""Associate auxiliary data with (almost) any object.

This supports objects that don't allow dynamic attributes, for example,
jax.Array or DynamicJaxprTracer.

Note that aux_data are associated to the exact object and will be cleared after
e.g. copy.copy, jax.tree.map, jnp.astype, etc.
"""

from typing import Any
import weakref


_aux_data: dict[int, tuple[weakref.ref, dict[str, Any]]] = {}
_MISSING = object()


def set(obj: Any, key: str, value: Any):  # pylint: disable=redefined-builtin
  obj_id = id(obj)
  data = _aux_data.get(obj_id, None)
  if data is None or data[0]() is None:
    # _aux_data might be None during shutdown.
    ref = weakref.ref(obj, lambda _: _aux_data and _aux_data.pop(obj_id, None))
    _aux_data[obj_id] = data = (ref, {})
  data[1][key] = value


def get(obj: Any, key: str, default: Any = _MISSING) -> Any:
  data = _aux_data.get(id(obj), None)
  if data is None or data[0]() is None or key not in data[1]:
    if default is _MISSING:
      raise KeyError(f"No auxiliary data {key!r} found for {obj}.")
    return default
  return data[1][key]


def clear(obj: Any):
  _aux_data.pop(id(obj), None)
