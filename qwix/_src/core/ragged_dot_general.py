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

"""Quantized jax.lax.ragged_dot_general."""

import jax
from jax import numpy as jnp
from qwix._src.core import numerics
from qwix._src.core import qarray


def _slow_ragged_dot_general(
    lhs,
    rhs,
    group_sizes,
    dimension_numbers,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    group_offset: jax.Array | None = None,
):
  """A ragged_dot_general which dequantizes first."""
  lhs = qarray.dequantize(lhs) if isinstance(lhs, qarray.QArray) else lhs
  rhs = qarray.dequantize(rhs) if isinstance(rhs, qarray.QArray) else rhs
  return jax.lax.ragged_dot_general(
      lhs,
      rhs,
      group_sizes,
      dimension_numbers,
      precision=precision,
      preferred_element_type=preferred_element_type,
      group_offset=group_offset,
  )


def _fast_ragged_dot_general(
    lhs,
    rhs,
    group_sizes,
    dimension_numbers,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    group_offset: jax.Array | None = None,
):
  """Quantized ragged_dot_general with a fast path."""
  lhs_val = lhs.qvalue if isinstance(lhs, qarray.QArray) else lhs
  rhs_val = rhs.qvalue if isinstance(rhs, qarray.QArray) else rhs
  lhs_scale = lhs.scale if isinstance(lhs, qarray.QArray) else None
  rhs_scale = rhs.scale if isinstance(rhs, qarray.QArray) else None

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

  if lhs_scale is not None:  # [M, 1]
    out = qarray.call_with_generic_broadcast(jnp.multiply, out, lhs_scale)
  if rhs_scale is not None:  # [1, N]
    rhs_scale = qarray.transpose_array(rhs_scale, (0, None, 1))
    out = qarray.call_with_generic_broadcast(jnp.multiply, out, rhs_scale)
  return out.astype(result_type)


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
    else:  # is jax.Array
      if numerics.should_quantize(operand.dtype):
        # Always dequantize on inputs if any of the operands is in bf16/fp32,
        # because XLA is able to fuse the dequantize and the matmul. The slow
        # path is usually not slower than the fast path, since both use fp
        # matmul, and will be significantly faster when subchannel or zero_point
        # is used.
        use_fast_path = False
        break
      # For raw arrays in lower precision, e.g. fp8, int4, bool, using fast path
      # may be beneficial.
      continue

    qarray.validate_qarray(operand)

    # qtypes like nf4 cannot be dequantized on output.
    if not numerics.can_dequant_on_output(operand.qtype):
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
