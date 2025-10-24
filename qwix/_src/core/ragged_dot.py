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

# RaggedDotDimensionNumbers that specify the tiled case.
_TILED_RAGGED_DOT_DIMENSION_NUMBERS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(((2,), (2,)), ((1,), (1,))),
    lhs_ragged_dimensions=[0],
    rhs_group_dimensions=[0],
)


def _apply_group_channelwise_scale(
    rhs_scale: jax.Array,
    lhs: qarray.MaybeQArray,
    group_sizes: jax.Array,
    dimension_numbers: jax.lax.RaggedDotDimensionNumbers,
    precision: jax.lax.PrecisionLike,
    group_offset: jax.Array | None,
) -> jax.Array:
  """Expands the group dimension of rhs_scale using a gather-like op."""
  (_, _), (lhs_ba, _) = dimension_numbers.dot_dimension_numbers
  ones_shape = []
  # Add the ragged dimension.
  ones_shape.append(lhs.shape[dimension_numbers.lhs_ragged_dimensions[0]])
  # Add the tile_count dimension if present.
  for dim in lhs_ba:
    ones_shape.append(lhs.shape[dim])
  # Add the contracting dimension.
  ones_shape.append(1)

  return jax.lax.ragged_dot_general(
      jnp.ones(tuple(ones_shape), rhs_scale.dtype),
      rhs_scale,
      group_sizes,
      dimension_numbers,
      precision=precision,
      group_offset=group_offset,
  )


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


def _tile_operand(
    operand: qarray.MaybeQArray,
    tiled_axes_spec: dict[int, int],
    ca: int,
) -> qarray.MaybeQArray:
  """Tile the operand for ragged_dot."""
  # Case 1. Not Quantized, return split jax.Array.
  if not isinstance(operand, qarray.QArray):
    # LHS: [M, K] -> [M, tile_count, tile_size]
    # RHS: [G, K, N] -> [G, tile_count, tile_size, N]
    return qarray.split_axis(operand, tiled_axes_spec)

  # Case 2. Quantized, return split QArray.
  assert operand.zero_point is None, 'zero_point not supported for ragged_dot'
  # LHS: [M, K] -> [M, tile_count, tile_size]
  # RHS: [G, K, N] -> [G, tile_count, tile_size, N]
  new_qvalue = qarray.split_axis(operand.qvalue, tiled_axes_spec)
  # tiled LHS: [M, tile_count] -> [M, tile_count, 1]
  # tiled RHS: [G, tile_count, N] -> [G, tile_count, 1, N]
  # non-tiled LHS: [M, 1] -> [M, 1, 1]
  # non-tiled RHS: [G, 1, N] -> [G, 1, 1, N]
  new_scale = qarray.split_axis(operand.scale, {ca: 1})
  return qarray.QArray(new_qvalue, new_scale, None, operand.qtype)


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
  is_tiled_path = False
  if dimension_numbers == _BASIC_RAGGED_DOT_DIMENSION_NUMBERS:
    # Tiling is only implemented for the basic ragged_dot case.
    ca = 1
    lhs_tiled_axes = (
        qarray.get_tiled_axes(lhs) if isinstance(lhs, qarray.QArray) else {}
    )
    rhs_tiled_axes = (
        qarray.get_tiled_axes(rhs) if isinstance(rhs, qarray.QArray) else {}
    )

    if ca in lhs_tiled_axes or ca in rhs_tiled_axes:
      is_tiled_path = True
      lhs_tile_size = lhs_tiled_axes.get(ca)
      rhs_tile_size = rhs_tiled_axes.get(ca)
      if lhs_tile_size and rhs_tile_size and lhs_tile_size != rhs_tile_size:
        raise ValueError(
            'Contracting axes for ragged_dot must be tiled with the same size.'
        )
      tile_size = lhs_tile_size or rhs_tile_size
      tiled_axes_spec = {ca: tile_size}

      lhs = _tile_operand(lhs, tiled_axes_spec, ca)
      rhs = _tile_operand(rhs, tiled_axes_spec, ca)
      dimension_numbers = _TILED_RAGGED_DOT_DIMENSION_NUMBERS

  lhs_val = lhs.qvalue if isinstance(lhs, qarray.QArray) else lhs
  rhs_val = rhs.qvalue if isinstance(rhs, qarray.QArray) else rhs
  lhs_scale = lhs.scale if isinstance(lhs, qarray.QArray) else None
  rhs_scale = rhs.scale if isinstance(rhs, qarray.QArray) else None

  lhs_scale_transpose, rhs_scale_transpose = _ragged_get_scale_transpose(
      dimension_numbers, (len(lhs.shape), len(rhs.shape))
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

  if lhs_scale is not None:
    lhs_scale = qarray.transpose_array(lhs_scale, lhs_scale_transpose)
    out = qarray.call_with_generic_broadcast(jnp.multiply, out, lhs_scale)
  if rhs_scale is not None:
    # Check if the scale has a group dimension that needs special handling.
    if (
        dimension_numbers.rhs_group_dimensions
        and rhs_scale.shape[dimension_numbers.rhs_group_dimensions[0]] > 1
    ):
      rhs_scale = _apply_group_channelwise_scale(
          rhs_scale,
          lhs,
          group_sizes,
          dimension_numbers,
          precision,
          group_offset,
      )
    else:
      rhs_scale = qarray.transpose_array(rhs_scale, rhs_scale_transpose)
    out = qarray.call_with_generic_broadcast(jnp.multiply, out, rhs_scale)

  if is_tiled_path:
    # [tile_count, M, N] -> [M, N]
    out = jnp.sum(out, axis=0)

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
