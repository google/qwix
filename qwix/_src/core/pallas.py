# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Support feeding QArray into Pallas kernels."""

import copy
import dataclasses
import functools
from typing import Any, Callable

import jax
from jax.experimental import pallas as pl
import numpy as np
from qwix._src.core import qarray


def pallas_call(
    kernel: Callable[..., None],
    out_shape: Any,
    *,
    grid_spec: pl.GridSpec | None = None,
    grid=(),
    in_specs=pl.no_block_spec,
    out_specs=pl.no_block_spec,
    scratch_shapes=(),
    **kwargs,
) -> Callable[..., Any]:
  """A lifted version of jax.pallas_call that takes QArray as arguments.

  The specs for the QArray arguments should be the same as the original jax
  arrays, and this function will take care of converting them properly.

  Args:
    kernel: same as in jax.pallas_call.
    out_shape: same as in jax.pallas_call.
    grid_spec: same as in jax.pallas_call.
    grid: same as in jax.pallas_call.
    in_specs: same as in jax.pallas_call. The specs for QArrays should stay the
      same as the original jax arrays.
    out_specs: same as in jax.pallas_call.
    scratch_shapes: same as in jax.pallas_call.
    **kwargs: extra arguments to pass to jax.pallas_call.

  Returns:
    A function that can be called on a number of positional array arguments to
    invoke the Pallas kernel.
  """
  if grid_spec is None:
    grid_spec = pl.GridSpec(grid, in_specs, out_specs, scratch_shapes)

  def wrapper(*args):
    num_scalar_prefetch = getattr(grid_spec, "num_scalar_prefetch", 0)

    in_specs = update_block_specs_for_qarray(
        grid_spec.in_specs, args[num_scalar_prefetch:]
    )
    in_specs, new_args, restore_fn = transform_block_specs_for_tpu(
        in_specs, args[num_scalar_prefetch:]
    )
    args = args[:num_scalar_prefetch] + new_args

    # pl.GridSpec doesn't support dataclasses.replace.
    new_grid_spec = copy.copy(grid_spec)
    new_grid_spec.in_specs = in_specs

    return pl.pallas_call(
        lambda *kernel_args: kernel(
            *kernel_args[:num_scalar_prefetch],
            *restore_fn(kernel_args[num_scalar_prefetch : len(args)]),
            *kernel_args[len(args) :],
        ),
        out_shape,
        grid_spec=new_grid_spec,
        **kwargs,
    )(*args)

  return wrapper


def update_block_specs_for_qarray(block_specs: Any, args: Any) -> Any:
  """Update block specs for QArray arguments."""

  def _update_block_spec(spec: pl.BlockSpec, arg):
    if not isinstance(arg, qarray.QArray):
      return spec

    # Calculate block size of the scale array for each axis.
    scale_block_shape = []
    for v, bv, s in zip(arg.qvalue.shape, spec.block_shape, arg.scale.shape):
      if bv is None:
        scale_block_shape.append(None)
      elif s == 1:
        scale_block_shape.append(1)
      else:
        assert v % bv == 0 and s % (v // bv) == 0, f"{v=} {bv=} {s=}"
        scale_block_shape.append(s // (v // bv))
    scale_block_shape = tuple(scale_block_shape)

    scale_index_map = lambda *a: tuple(
        i if s > 1 else 0 for i, s in zip(spec.index_map(*a), arg.scale.shape)
    )
    scale_block_spec = dataclasses.replace(
        spec, block_shape=scale_block_shape, index_map=scale_index_map
    )
    assert arg.zero_point is None, "Zero point is not supported yet."

    return dataclasses.replace(arg, qvalue=spec, scale=scale_block_spec)

  return jax.tree.map(_update_block_spec, block_specs, args)


def transform_block_specs_for_tpu(
    block_specs: Any, args: Any
) -> tuple[Any, Any, Callable[..., Any]]:
  """Transform block specs and arguments so that they can be used on TPU.

  This workarounds the Pallas TPU requirement that the block shapes must be
  divisible by 8x128. A "restore" function is returned to restore the actual
  unpadded block shapes inside the kernel.

  Args:
    block_specs: a pytree of pl.BlockSpec.
    args: the arguments to the pallas_call, which have the same structure as
      block_specs.

  Returns:
    A tuple of (new_block_specs, new_args, restore_fn), where new_block_specs
    and new_args are the transformed pytrees, and restore_fn is a function that
    needs to be called inside the kernel on the kernel arguments to restore the
    actual block shapes.
  """
  flatten_block_specs, treedef = jax.tree.flatten(block_specs)
  flatten_args = treedef.flatten_up_to(args)

  # Information needed to restore the original block shapes inside the kernel.
  # The keys are the indices of the arrays in the flattened pytree. A key may
  # only appear in one of the dictionaries.
  reverse_transposes = {}
  reverse_reshapes = {}

  for i, (spec, arg) in enumerate(zip(flatten_block_specs, flatten_args)):
    # Check if the block shape is already optimal.
    if _is_optimal_for_tpu(spec.block_shape, arg.shape):
      continue

    # Solution 1: try to transpose the array to put the longest axis at the end.
    transpose = np.argsort([1 if s is None else s for s in spec.block_shape])
    block_shape_t = _reorder(spec.block_shape, transpose)
    if _can_fit_tpu_requirements(block_shape_t, _reorder(arg.shape, transpose)):
      flatten_args[i] = arg.transpose(transpose)
      index_map_t = functools.partial(
          lambda spec, transpose, *a: _reorder(spec.index_map(*a), transpose),
          spec,
          transpose,
      )
      flatten_block_specs[i] = dataclasses.replace(
          spec, block_shape=block_shape_t, index_map=index_map_t
      )
      reverse_transposes[i] = np.argsort(
          [t for t in transpose if spec.block_shape[t] is not None]
      )
      continue

    # Solution 2: reshape the array into (*num_blocks, 1, prod(block_shape))).
    # This satisfies the TPU requirement because the last two dimensions are
    # equal to the respective dimension of the overall array.
    dims = range(arg.ndim)
    arg_t = qarray.split_axis(arg, {i: spec.block_shape[i] or 1 for i in dims})
    arg_t = arg_t.transpose([2 * i for i in dims] + [2 * i + 1 for i in dims])
    arg_t = arg_t.reshape([arg_t.shape[i] for i in dims] + [1, -1])
    flatten_args[i] = arg_t
    block_shape_t = tuple([1 for _ in dims] + [1, arg_t.shape[-1]])
    index_map_t = functools.partial(
        lambda spec, *a: spec.index_map(*a) + (0, 0), spec
    )
    flatten_block_specs[i] = dataclasses.replace(
        spec, block_shape=block_shape_t, index_map=index_map_t
    )
    reverse_reshapes[i] = [bs for bs in spec.block_shape if bs is not None]

  def restore(kernel_args):
    kernel_args = treedef.flatten_up_to(kernel_args)
    for i, kernel_arg in enumerate(kernel_args):
      if i in reverse_transposes:
        # Use pallas-friendly transpose.
        kernel_args[i] = qarray.transpose_array(
            kernel_arg[...], reverse_transposes[i]
        )
      elif i in reverse_reshapes:
        kernel_args[i] = kernel_arg[...].reshape(reverse_reshapes[i])
    return treedef.unflatten(kernel_args)

  return (
      treedef.unflatten(flatten_block_specs),
      treedef.unflatten(flatten_args),
      restore,
  )


def _reorder(
    sequence: tuple[Any, ...], order: tuple[int, ...]
) -> tuple[Any, ...]:
  """Reorder/transpose a sequence of elements."""
  return tuple(sequence[i] for i in order)


def _can_fit_tpu_requirements(
    block_shape: tuple[int | None, ...], arg_shape: tuple[int, ...]
) -> bool:
  """Check if the block shape can fit the TPU requirements."""
  block_shape = tuple(1 if s is None else s for s in block_shape)
  return (block_shape[-1] % 128 == 0 or block_shape[-1] == arg_shape[-1]) and (
      block_shape[-2] % 8 == 0 or block_shape[-2] == arg_shape[-2]
  )


def _is_optimal_for_tpu(
    block_shape: tuple[int | None, ...], arg_shape: tuple[int, ...]
) -> bool:
  """Check if the block shape is already optimal for TPU."""
  block_shape = tuple(1 if s is None else s for s in block_shape)
  if not _can_fit_tpu_requirements(block_shape, arg_shape):
    # Not optimal if cannot fit the TPU requirements.
    return False
  if block_shape[-1] % 128 == 0 and block_shape[-2] % 8 == 0:
    # Optimal if no padding is needed.
    return True
  # Optimal if the block shape is already sorted.
  return sorted(block_shape) == list(block_shape)
