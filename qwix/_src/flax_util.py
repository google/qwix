# Copyright 2025 Google LLC
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
"""Utility functions to work with Flax linen and nnx modules."""

from collections.abc import Callable, Collection, Sequence
import dataclasses
import functools
import inspect
from typing import Any

import flax
from flax import linen as nn
from flax import nnx
import jax


@nnx.register_variable_name('quant_stats', overwrite=True)
class QuantStat(nnx.Variable):
  """NNX variable that stores quantization statistics."""


def should_update_quant_stats() -> bool:
  """Returns whether the current module should update the quantization statistics."""
  module = get_current_module()
  match module:
    case nn.Module():
      if module.is_initializing():
        return False  # Don't update quant_stats during initialization.
      return module.is_mutable_collection('quant_stats')
    case nnx.Module():
      return not module.disable_quant_stats_update


def get_current_module() -> nn.Module | nnx.Module:
  """Returns the current module, or raises an error if current module is not known.

  Args: None.

  Returns:
    The current module, it can be either a linen module or an nnx module.
  Raises:
    ValueError: If the current module is not known.
  """
  # Linen modules can be found in nn.module._context.module_stack.
  if nn.module._context.module_stack and nn.module._context.module_stack[-1]:  # pylint: disable=protected-access
    return nn.module._context.module_stack[-1]  # pylint: disable=protected-access

  # Retrieves the nnx modules from the caller stack.
  current_frame = inspect.currentframe()
  while current_frame is not None:
    if 'self' in current_frame.f_locals and isinstance(
        current_frame.f_locals['self'], nnx.Module
    ):
      return current_frame.f_locals['self']
    current_frame = current_frame.f_back

  raise ValueError('Current module is not known.')


def get_current_module_path() -> tuple[str, ...]:
  """Returns the path of the current module, or raises an error if current module is not known.

  Args: None.

  Returns:
    A tuple representing a unique path to the current module from the root
    module.
  """
  module = get_current_module()
  match module:
    case nn.Module():
      return module.path
    case nnx.Module():
      # Paths of nnx modules are set when they are quantized.
      return module.qwix_path


def get_or_create_variable(
    collection: str, name: str, init_fn: Callable[[], Any]
) -> nn.Variable | nnx.Variable:
  """Gets or creates a quant_stat in the current module.

  Args:
    collection: The variable collection name.
    name: The variable name.
    init_fn: The function that will be called to compute the initial value of
      the variable.

  Returns:
    A variable wrapped in nn.Variable or nnx.Variable.
  """
  module = get_current_module()
  if isinstance(module, nn.Module):
    # Instead of module.variable, use module.scope.variable which allows us to
    # create variables in non-compact modules.
    assert module.scope is not None
    variable = module.scope.variable(collection, name, init_fn)
    _check_shape(variable.value, init_fn)
    return variable
  elif isinstance(module, nnx.Module):
    var_type = nnx.variable_type_from_name(collection)
    if hasattr(module, name):
      variable = getattr(module, name)
      assert type(variable) is var_type, f'{type(variable)} != {var_type}'  # pylint: disable=unidiomatic-typecheck
      _check_shape(variable.value, init_fn)
    else:
      variable = var_type(init_fn())
      setattr(module, name, variable)
    return variable


def get_and_delete_variable(collection: str, name: str) -> Any | None:
  """Gets and deletes a quant_stat in the current module.

  This is mainly for NNX and doesn't make much sense for linen.

  Args:
    collection: The variable collection name, only used for linen modules.
    name: The variable name.

  Returns:
    The variable, or None if it doesn't exist.
  """
  module = get_current_module()
  if isinstance(module, nn.Module):
    assert module.scope is not None
    if module.has_variable(collection, name):
      return module.scope._mutable_collection(collection).pop(name)  # pylint: disable=protected-access
    return None
  elif isinstance(module, nnx.Module):
    quant_stat = getattr(module, name, None)
    if quant_stat is not None:
      delattr(module, name)
    return quant_stat


def get_or_create_param(
    name: str,
    init_fn: Callable[..., Any],
    *,
    nnx_param_type: type[nnx.Param] = nnx.Param,
    need_rng: bool = False,
) -> Any:
  """Gets or creates a param in the current module.

  Args:
    name: The param name.
    init_fn: The function that will be called to compute the initial value of
      this variable, which may take a jax.Array as the RNG key.
    nnx_param_type: The nnx param type of the param, if it's not a nnx.Param.
    need_rng: If True, the init_fn will be called with a rng key as the first
      argument.

  Returns:
    The unboxed param.
  """
  module = get_current_module()
  if isinstance(module, nn.Module):
    # Instead of module.param, use module.scope.param which allows us to
    # create params in non-compact modules.
    assert module.scope is not None
    return module.scope.param(
        name, init_fn if need_rng else lambda _: init_fn()
    )
  elif isinstance(module, nnx.Module):
    if hasattr(module, name):
      param = getattr(module, name)
      if need_rng:
        init_fn = functools.partial(init_fn, jax.random.key(0))
      _check_shape(param.value, init_fn)
    else:
      if need_rng:
        if not isinstance(module.qwix_rngs, nnx.Rngs):
          raise ValueError(
              'Cannot find rngs in the current module. Please set rngs via'
              ' model.set_attributes(qwix_rngs=nnx.Rngs(...)).'
          )
        init_value = init_fn(module.qwix_rngs.params())
      else:
        init_value = init_fn()
      param = jax.tree.map(nnx_param_type, init_value)
      setattr(module, name, param)
    return unbox(param)


def find_param(x: Any, ptq_array_type=None) -> str | None:
  """Finds the param name of a given array in the current module.

  This function is useful when
    * We want to know if an array is a weight to decide how to quantize it.
    * We want to locate and replace the weights in PTQ mode.
    * We want to create lora weights together with the base weights.

  This function is designed so that array and array-like objects are treated
  equally.

  Args:
    x: jax.Array or array-like object (such as ptq.WithAux) that has the "shape"
      attribute.
    ptq_array_type: when looking for a param in the current module, also
      consider this type in addition to jax.Array. This is only used in QLoRA
      mode and should be set to ptq.WithAux. We don't import ptq here to avoid a
      circular dependency.

  Returns:
    The name of the param that contains the given array, or None if not found.

  Raises:
    ValueError: If multiple params are found.
  """
  # First, identify all the candidates. We support the following cases:
  # * The type of the param is promoted, e.g., astype.
  # * The sharding constraints are added, i.e. with_sharding_constraint.
  # * The param is reshaped (only through the approach 1 below).
  module = get_current_module()
  candidates: dict[str, Any] = {}
  if isinstance(module, nn.Module):
    array_types = jax.Array | ptq_array_type if ptq_array_type else jax.Array
    assert module.scope is not None
    for name, value in module.scope._collection('params').items():  # pylint: disable=protected-access
      value = nn.unbox(value)
      if isinstance(value, array_types):
        candidates[name] = value
  elif isinstance(module, nnx.Module):
    array_types = nnx.Param | ptq_array_type if ptq_array_type else nnx.Param
    for name, node in module.__dict__.items():
      if isinstance(node, array_types):
        candidates[name] = node.value
  else:
    raise ValueError('Current module is not known.')

  candidates_by_id = {id(c): n for n, c in candidates.items()}

  # Approach 0: check if we could find the exact x in the params.
  if id(x) in candidates_by_id:
    return candidates_by_id[id(x)]

  # Approach 1: if x is a JitTracer, we can actually trace it back to the param.
  if isinstance(x, jax.core.Tracer) and hasattr(x, 'parent'):
    while True:
      if id(x) in candidates_by_id:
        return candidates_by_id[id(x)]
      if x.parent and len(x.parent.in_tracers) == 1:
        # Allow any unary primitives to be applied to the param, including
        # reshape, with_sharding_constraint, astype, etc.
        x = x.parent.in_tracers[0]
      elif id(const := x.get_const()) in candidates_by_id:
        # Even if x is a tracer, the candidate might be a concrete value.
        return candidates_by_id[id(const)]
      else:
        return None

  # Approach 2: use heuristics on the shape which may not be accurate.
  candidates = {n: c for n, c in candidates.items() if c.shape == x.shape}
  if len(candidates) > 2:
    raise ValueError(f'Multiple candidate params found: {candidates.keys()}')
  if len(candidates) == 1:
    return list(candidates.keys())[0]

  return None


def unbox(maybe_boxed: Any) -> Any:
  """Similar to nn.unbox but also works for nnx.Variable/VariableState/VariableMetadata."""

  def fn(x):
    if isinstance(x, nnx.Variable):
      return x.value
    elif isinstance(x, nnx.VariableState):
      return x.value
    elif isinstance(x, nnx.VariableMetadata):
      return x.raw_value
    elif isinstance(x, nn.meta.AxisMetadata):
      return x.unbox()
    else:
      return x

  return jax.tree.map(
      fn,
      maybe_boxed,
      is_leaf=lambda x: isinstance(x, nn.meta.AxisMetadata | nnx.Variable),
  )


def update_sharding(
    spec: Sequence[Any],
    *,
    shape: Sequence[int] | None = None,
    split: Collection[int] | None = None,
    merge: Collection[int] | None = None,
    transpose: Sequence[int | None] | None = None,
) -> tuple[Any, ...]:
  """Derives the partition spec from an existing spec.

  Args:
    spec: The existing partition spec.
    shape: Optional. The shape of the value. If provided, the partition spec
      will be updated to remove sharding for dimensions of size 1.
    split: Splits the given spec into (metadata, None).
    merge: Merge the given spec (metadata, None) into metadata. This is the
      inverse of split and takes the same arguments.
    transpose: Transposes the spec.

  Returns:
    The updated partition spec.
  """
  assert bool(split) + bool(merge) + bool(transpose) <= 1
  if split:
    spec = [(a, None) if i in split else (a,) for i, a in enumerate(spec)]
    spec = sum(spec, ())  # flatten the list of tuples.
  elif merge:
    for i in merge:
      spec = spec[: i + 1] + spec[i + 2 :]  # pytype: disable=unsupported-operands
  elif transpose:
    spec = tuple(spec[i] if i is not None else None for i in transpose)

  if shape:
    assert len(shape) == len(spec), f'{shape=} {spec=}'
    # For scales: remove sharding for dimensions of size 1.
    spec = tuple(None if d == 1 else a for a, d in zip(spec, shape))

  return spec


def update_boxed(
    boxed: nn.meta.AxisMetadata | nnx.Variable | jax.Array,
    *,
    value: jax.Array | None = None,
    split: Collection[int] | None = None,
    merge: Collection[int] | None = None,
    transpose: Sequence[int | None] | None = None,
) -> nn.meta.AxisMetadata | nnx.Variable | jax.Array:
  """Update a boxed value and its metadata, i.e., nn.meta.AxisMetadata or nnx.Variable.

  The most common use case of metadata is to store the logical axis names for
  sharding.

  This method can split, merge, or transpose the axes metadata. Only one of
  them can be used at a time.

  Args:
    boxed: The metadata to update, or a jax.Array if no metadata is available.
    value: The value used to replace the boxed value.
    split: Splits the given axes metadata into (metadata, None).
    merge: Merge the given axes (metadata, None) into metadata. This is the
      inverse of split and takes the same arguments.
    transpose: Transposes the axes metadata.

  Returns:
    The updated metadata, which may be the same as the input.
  """
  if isinstance(boxed, nn.meta.AxisMetadata):
    if value is not None:
      boxed = boxed.replace_boxed(value)
    shape = boxed.unbox().shape
    for possible_field in ('names', 'mesh_axes', 'axes_types'):
      axes = getattr(boxed, possible_field, None)
      if isinstance(axes, (list, tuple)):
        axes = update_sharding(
            axes, shape=shape, split=split, merge=merge, transpose=transpose
        )
        boxed = dataclasses.replace(boxed, **{possible_field: axes})
  elif isinstance(boxed, nnx.Variable):
    if value is not None:
      boxed = boxed.replace(value)
    shape = boxed.value.shape
    axes = boxed.get_metadata().get('sharding_names', None)
    if isinstance(axes, (list, tuple)):
      axes = update_sharding(
          axes, shape=shape, split=split, merge=merge, transpose=transpose
      )
      boxed.set_metadata(sharding_names=axes)
  elif isinstance(boxed, jax.Array):  # not boxed.
    if value is not None:
      boxed = value
  else:
    raise ValueError(f'Unsupported type: {type(boxed)}')

  return boxed


def _check_shape(value: Any, init_fn: Callable[[], Any]):
  """Checks the shape of the value to match the shape of init_fn().

  This is similar to the check in nn.Module.param, which is not performed for
  linen variables or in nnx.

  Args:
    value: The value to check.
    init_fn: The function that will be called to compute the initial value of
      this variable.

  Raises:
    ValueError: If the shape of the value does not match the shape of init_fn().
  """
  # Value may be a FrozenDict.
  abs_value = jax.eval_shape(lambda: flax.core.unfreeze(value))
  abs_init_value = jax.eval_shape(lambda: unbox(init_fn()))
  if abs_value != abs_init_value:
    raise ValueError(f'{abs_value} != {abs_init_value}')


def make_rng(rng_stream: str) -> jax.Array:
  """Generate a random key from the given rng_stream in the current module."""
  module = get_current_module()
  if isinstance(module, nn.Module):
    return module.make_rng(rng_stream)
  elif isinstance(module, nnx.Module):
    if not isinstance(module.qwix_rngs, nnx.Rngs):
      raise ValueError(
          'Cannot find rngs in the current module. Please set rngs via'
          ' model.set_attributes(qwix_rngs=nnx.Rngs(...)).'
      )
    return module.qwix_rngs[rng_stream]()
  else:
    raise ValueError('Current module is not known.')
