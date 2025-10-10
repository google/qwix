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
      ones_lhs = jnp.ones(
          (lhs.shape[dimension_numbers.lhs_ragged_dimensions[0]], 1),
          rhs_scale.dtype,
      )
      rhs_scale = jax.lax.ragged_dot(
          ones_lhs,
          rhs_scale,
          group_sizes,
          precision=precision,
          group_offset=group_offset,
      )
    else:
      rhs_scale = qarray.transpose_array(rhs_scale, rhs_scale_transpose)
    out = qarray.call_with_generic_broadcast(jnp.multiply, out, rhs_scale)

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
  (lhs_ca, rhs_ca), _ = dimension_numbers.dot_dimension_numbers
  for operand, ca_dims in zip((lhs, rhs), (lhs_ca, rhs_ca)):
    if isinstance(operand, qarray.QArray):
      if operand.zero_point is not None or any(
          operand.scale.shape[ca] > 1 for ca in ca_dims
      ):
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
