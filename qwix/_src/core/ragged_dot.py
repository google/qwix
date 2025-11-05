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

"""Quantized jax.lax.ragged_dot and jax.lax.ragged_dot_general."""

from collections.abc import Collection, Sequence
import jax
from jax import numpy as jnp
from qwix._src.core import numerics
from qwix._src.core import qarray


# RaggedDotDimensionNumbers that specify the simple case (i.e., qwix.ragged_dot)
_BASIC_RAGGED_DOT_DIMENSION_NUMBERS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(((1,), (1,)), ((), ())),
    lhs_ragged_dimensions=[0],
    rhs_group_dimensions=[0],
)


def _apply_group_channelwise_scale(
    rhs_scale: jax.Array,
    lhs_shape: tuple[int, ...],
    group_sizes: jax.Array,
    dimension_numbers: jax.lax.RaggedDotDimensionNumbers,
    precision: jax.lax.PrecisionLike,
    group_offset: jax.Array | None,
) -> jax.Array:
  """Expands the group dimension of rhs_scale using a gather-like op."""
  (lhs_ca, _), _ = dimension_numbers.dot_dimension_numbers

  # Create a `jnp.ones` tensor with the same rank and layout as lhs_val,
  # but with contracting dimensions set to size 1.
  ones_shape = list(lhs_shape)
  for contracting_axis in lhs_ca:
    ones_shape[contracting_axis] = 1
  lhs_ones = jnp.ones(tuple(ones_shape), rhs_scale.dtype)

  return jax.lax.ragged_dot_general(
      lhs_ones,
      rhs_scale,
      group_sizes,
      dimension_numbers,
      precision=precision,
      group_offset=group_offset,
  )


def _apply_tiling(
    contracting_axes: Sequence[int],
    batch_axes: Sequence[int],
    tiled_axes: Collection[int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
  """Apply tiling to dimension numbers.

  Each tiled contracting axis is split into two axes, the first being the new
  batch axis, and the second being the new contracting axis.

  Args:
    contracting_axes: The original contracting axes.
    batch_axes: The original batch axes.
    tiled_axes: The tiled axes. Must be a subset of contracting_axes.

  Returns:
    A tuple of (new_ca, new_ba, sum_axes).
  """
  new_ca = [a + sum(t <= a for t in tiled_axes) for a in contracting_axes]
  new_ba = [a + sum(t < a for t in tiled_axes) for a in batch_axes]
  # We choose to insert the tile_count axes to the end of the batch axes.
  # Alternatively, we could insert them to the beginning or to the middle,
  # as long as lhs and rhs use the same order.
  new_ba += [
      a + sum(t < a for t in tiled_axes)
      for a in contracting_axes
      if a in tiled_axes
  ]
  sum_axes = range(len(batch_axes), len(new_ba))
  return tuple(new_ca), tuple(new_ba), tuple(sum_axes)


def _ragged_get_scale_transpose(
    dimension_numbers: jax.lax.RaggedDotDimensionNumbers,
    ndims: tuple[int, int],
) -> tuple[list[int | None], list[int | None]]:
  """Calculates the transpose permutation for lhs_scale and rhs_scale."""
  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers.dot_dimension_numbers
  lhs_ragged_dims = dimension_numbers.lhs_ragged_dimensions
  rhs_group_dims = dimension_numbers.rhs_group_dimensions

  lhs_remaining_dims = sorted(
      set(range(ndims[0])) - set(lhs_ca) - set(lhs_ba) - set(lhs_ragged_dims)
  )
  rhs_remaining_dims = sorted(
      set(range(ndims[1])) - set(rhs_ca) - set(rhs_ba) - set(rhs_group_dims)
  )

  lhs_scale_transpose = (
      list(lhs_ba)
      + list(lhs_ragged_dims)
      + list(lhs_remaining_dims)
      + [None] * len(rhs_remaining_dims)
  )
  rhs_scale_transpose = (
      list(rhs_ba)
      + [None] * (len(lhs_ragged_dims) + len(lhs_remaining_dims))
      + list(rhs_remaining_dims)
  )

  return lhs_scale_transpose, rhs_scale_transpose


def _fast_ragged_dot_general(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    group_sizes: jax.Array,
    dimension_numbers: jax.lax.RaggedDotDimensionNumbers,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    group_offset: jax.Array | None = None,
):
  """Quantized ragged_dot_general with a fast path."""
  if isinstance(lhs, qarray.QArray):
    lhs_val = lhs.qvalue
    lhs_scale = lhs.scale
    lhs_tiled_axes = qarray.get_tiled_axes(lhs)
  else:
    lhs_val = lhs
    lhs_scale = None
    lhs_tiled_axes = {}
  if isinstance(rhs, qarray.QArray):
    rhs_val = rhs.qvalue
    rhs_scale = rhs.scale
    rhs_tiled_axes = qarray.get_tiled_axes(rhs)
  else:
    rhs_val = rhs
    rhs_scale = None
    rhs_tiled_axes = {}

  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers.dot_dimension_numbers

  # Figure out the tiled axes to use for the dot_general. For greater
  # flexibility, we allow a non-tiled axis to be contracted with a tiled axis.
  # However, if both axes are tiled, their tile sizes must be the same.
  lhs_tiled_ca = {}
  rhs_tiled_ca = {}
  for l, r in zip(lhs_ca, rhs_ca):
    lhs_tile_size = lhs_tiled_axes.get(l)
    rhs_tile_size = rhs_tiled_axes.get(r)
    if lhs_tile_size and rhs_tile_size and lhs_tile_size != rhs_tile_size:
      raise ValueError(
          'Contracting axes must be tiled with the same tile size.'
          f' {lhs_tiled_axes=} {rhs_tiled_axes=} {dimension_numbers=}'
      )
    if lhs_tile_size or rhs_tile_size:
      lhs_tiled_ca[l] = lhs_tile_size or rhs_tile_size
      rhs_tiled_ca[r] = lhs_tile_size or rhs_tile_size

  # Split lhs/rhs_value for tiled axes.
  lhs_val = qarray.split_axis(lhs_val, lhs_tiled_ca)
  rhs_val = qarray.split_axis(rhs_val, rhs_tiled_ca)

  lhs_ca, lhs_ba, sum_axes = _apply_tiling(lhs_ca, lhs_ba, lhs_tiled_ca)
  rhs_ca, rhs_ba, _ = _apply_tiling(rhs_ca, rhs_ba, rhs_tiled_ca)
  dot_dimension_numbers = (lhs_ca, rhs_ca), (lhs_ba, rhs_ba)
  dimension_numbers = jax.lax.RaggedDotDimensionNumbers(
      dot_dimension_numbers=dot_dimension_numbers,
      lhs_ragged_dimensions=dimension_numbers.lhs_ragged_dimensions,
      rhs_group_dimensions=dimension_numbers.rhs_group_dimensions,
  )

  preferred_element_type, result_type = qarray.get_accumulator_and_result_type(
      lhs, rhs, preferred_element_type=preferred_element_type
  )

  out = jax.lax.ragged_dot_general(
      lhs_val,
      rhs_val,
      group_sizes,
      dimension_numbers,
      precision=precision,
      preferred_element_type=preferred_element_type,
      group_offset=group_offset,
  )

  lhs_scale_transpose, rhs_scale_transpose = _ragged_get_scale_transpose(
      dimension_numbers, (len(lhs_val.shape), len(rhs_val.shape))
  )
  if lhs_scale is not None:
    lhs_scale = qarray.split_axis(lhs_scale, {a: 1 for a in lhs_tiled_ca})
    lhs_scale = qarray.transpose_array(lhs_scale, lhs_scale_transpose)
    out = qarray.call_with_generic_broadcast(jnp.multiply, out, lhs_scale)
  if rhs_scale is not None:
    rhs_scale = qarray.split_axis(rhs_scale, {a: 1 for a in rhs_tiled_ca})
    # Check if the scale has a group dimension that needs special handling.
    if (
        dimension_numbers.rhs_group_dimensions
        and rhs_scale.shape[dimension_numbers.rhs_group_dimensions[0]] > 1
    ):
      rhs_scale = _apply_group_channelwise_scale(
          rhs_scale,
          lhs_val.shape,
          group_sizes,
          dimension_numbers,
          precision,
          group_offset,
      )
    else:
      rhs_scale = qarray.transpose_array(rhs_scale, rhs_scale_transpose)
    out = qarray.call_with_generic_broadcast(jnp.multiply, out, rhs_scale)

  if sum_axes:
    # [tile_count1, tile_count2, ..., M, N] -> [M, N]
    out = jnp.sum(out, axis=sum_axes)

  return out.astype(result_type)


def _slow_ragged_dot_general(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    group_sizes: jax.Array,
    dimension_numbers: jax.lax.RaggedDotDimensionNumbers,
    **kwargs,
):
  """A ragged_dot_general which dequantizes first."""
  lhs = qarray.dequantize(lhs) if isinstance(lhs, qarray.QArray) else lhs
  rhs = qarray.dequantize(rhs) if isinstance(rhs, qarray.QArray) else rhs
  return jax.lax.ragged_dot_general(
      lhs, rhs, group_sizes, dimension_numbers, **kwargs
  )


def ragged_dot_general(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    group_sizes: jax.Array,
    dimension_numbers: jax.lax.RaggedDotDimensionNumbers,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    group_offset: jax.Array | None = None,
) -> jax.Array:
  """Quantized jax.lax.ragged_dot_general."""
  use_fast_path = True
  for operand in (lhs, rhs):
    if isinstance(operand, qarray.QArray):
      if operand.zero_point is not None:
        use_fast_path = False
        break
    else:
      if numerics.should_quantize(operand.dtype):
        # Always dequantize on inputs if any of the operands is in bf16/fp32,
        # because XLA is able to fuse the dequantize and the matmul. The slow
        # path is usually not slower than the fast path, since both use fp
        # matmul, and will be significantly faster when subchannel or zero_point
        # is used.
        use_fast_path = False
        break

  if use_fast_path:
    return _fast_ragged_dot_general(
        lhs,
        rhs,
        group_sizes,
        dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
    )
  else:
    return _slow_ragged_dot_general(
        lhs,
        rhs,
        group_sizes,
        dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
    )


def ragged_dot(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    group_sizes: jax.Array,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    group_offset: jax.Array | None = None,
) -> jax.Array:
  """Quantized jax.lax.ragged_dot."""
  return ragged_dot_general(
      lhs,
      rhs,
      group_sizes,
      dimension_numbers=_BASIC_RAGGED_DOT_DIMENSION_NUMBERS,
      precision=precision,
      preferred_element_type=preferred_element_type,
      group_offset=group_offset,
  )
