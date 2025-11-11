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
"""Quantized jax.lax.conv_general_dilated."""

from collections.abc import Sequence
from typing import Any
import jax
from jax import numpy as jnp
from qwix._src.core import numerics
from qwix._src.core import qarray


def get_how_to_quantize(
    *,
    dimension_numbers: jax.lax.ConvDimensionNumbers,
    for_lhs: bool,
    **kwargs: Any,
) -> qarray.HowToQuantize:
  """Gets how to quantize from conv's dimension_numbers.

  Use channelwise for batch dimension and out feature dimension.

  Args:
    dimension_numbers: The conv's dimension_numbers.
    for_lhs: Whether to quantize lhs or rhs.
    **kwargs: Additional keyword arguments to HowToQuantize.

  Returns:
    How to quantize lhs or rhs.
  """
  if for_lhs:
    channelwise_axes = [dimension_numbers.lhs_spec[0]]
  else:
    channelwise_axes = [dimension_numbers.rhs_spec[0]]
  return qarray.HowToQuantize(
      channelwise_axes=channelwise_axes,
      tiled_axes={},
      **kwargs,
  )


def get_transpose(
    dimension_numbers: jax.lax.ConvDimensionNumbers, for_lhs: bool
) -> list[int | None]:
  """Returns the transpose list for the given dimension_numbers."""
  transpose = [None] * len(dimension_numbers.out_spec)
  if for_lhs:
    # Only batch dimension can be channelwise thus transposed.
    transpose[dimension_numbers.out_spec[0]] = dimension_numbers.lhs_spec[0]
  else:
    # Only out feature dimension can be channelwise thus transposed.
    transpose[dimension_numbers.out_spec[1]] = dimension_numbers.rhs_spec[0]
  return transpose


def _slow_conv_general_dilated(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    window_strides: Sequence[int],
    padding: str | Sequence[tuple[int, int]],
    lhs_dilation: Sequence[int] | None = None,
    rhs_dilation: Sequence[int] | None = None,
    dimension_numbers: jax.lax.ConvGeneralDilatedDimensionNumbers = None,
    feature_group_count: int = 1,
    batch_group_count: int = 1,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    out_sharding=None,
) -> jax.Array:
  """Dequantizes first then computes in floating-point types."""
  if isinstance(lhs, qarray.QArray):
    lhs = qarray.dequantize(lhs)
  if isinstance(rhs, qarray.QArray):
    rhs = qarray.dequantize(rhs)
  return jax.lax.conv_general_dilated(
      lhs,
      rhs,
      window_strides,
      padding,
      lhs_dilation,
      rhs_dilation,
      dimension_numbers,
      feature_group_count,
      batch_group_count,
      precision,
      preferred_element_type,
      out_sharding,
  )


def _fast_conv_general_dilated(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    window_strides: Sequence[int],
    padding: str | Sequence[tuple[int, int]],
    lhs_dilation: Sequence[int] | None = None,
    rhs_dilation: Sequence[int] | None = None,
    dimension_numbers: jax.lax.ConvGeneralDilatedDimensionNumbers = None,
    feature_group_count: int = 1,
    batch_group_count: int = 1,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    out_sharding=None,
) -> jax.Array:
  """Quantized jax.lax.conv_general_dilated. Both sides must be QArrays."""
  dimension_numbers = jax.lax.conv_dimension_numbers(
      lhs.shape, rhs.shape, dimension_numbers
  )
  preferred_element_type, result_type = qarray.get_accumulator_and_result_type(
      lhs, rhs, preferred_element_type=preferred_element_type
  )

  if isinstance(lhs, qarray.QArray):
    lhs_value = lhs.qvalue
    lhs_scale = lhs.scale
    lhs_zero_point = lhs.zero_point
    if qarray.get_tiled_axes(lhs):
      raise ValueError('subchannel is not supported for conv_general_dilated.')
  else:
    lhs_value = lhs
    lhs_scale = None
    lhs_zero_point = None

  if isinstance(rhs, qarray.QArray):
    rhs_value = rhs.qvalue
    rhs_scale = rhs.scale
    rhs_zero_point = rhs.zero_point
    if qarray.get_tiled_axes(rhs):
      raise ValueError('subchannel is not supported for conv_general_dilated.')
  else:
    rhs_value = rhs
    rhs_scale = None
    rhs_zero_point = None

  if rhs_zero_point is not None:
    raise ValueError('Asymmetric quantization for rhs is not supported.')

  res = jax.lax.conv_general_dilated(
      lhs_value,
      rhs_value,
      window_strides=window_strides,
      padding=padding,
      lhs_dilation=lhs_dilation,
      rhs_dilation=rhs_dilation,
      dimension_numbers=dimension_numbers,
      feature_group_count=feature_group_count,
      batch_group_count=batch_group_count,
      preferred_element_type=preferred_element_type,
      out_sharding=out_sharding,
  )
  if lhs_zero_point is not None:
    # TODO(zhuyunx): This value can be constant folded in SRQ scenarios.
    res -= jax.lax.conv_general_dilated(
        jnp.broadcast_to(lhs_zero_point, lhs_value.shape),
        rhs_value,
        window_strides=window_strides,
        padding=padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count,
        preferred_element_type=preferred_element_type,
    )

  if lhs_scale is not None:
    transpose = get_transpose(dimension_numbers, for_lhs=True)
    lhs_scale = qarray.transpose_array(lhs_scale, transpose)
    res = qarray.call_with_generic_broadcast(jnp.multiply, res, lhs_scale)

  if rhs_scale is not None:
    transpose = get_transpose(dimension_numbers, for_lhs=False)
    rhs_scale = qarray.transpose_array(rhs_scale, transpose)
    res = qarray.call_with_generic_broadcast(jnp.multiply, res, rhs_scale)

  return res.astype(result_type)


def conv_general_dilated(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    window_strides: Sequence[int],
    padding: str | Sequence[tuple[int, int]],
    lhs_dilation: Sequence[int] | None = None,
    rhs_dilation: Sequence[int] | None = None,
    dimension_numbers: jax.lax.ConvGeneralDilatedDimensionNumbers = None,
    feature_group_count: int = 1,
    batch_group_count: int = 1,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    out_sharding=None,
) -> jax.Array:
  """Dispatches to fast or slow conv_general_dilated depending on the inputs."""
  use_fast_path = True

  for x in (lhs, rhs):
    if isinstance(x, jax.Array) and numerics.should_quantize(x.dtype):
      use_fast_path = False
      break

  if use_fast_path:
    return _fast_conv_general_dilated(
        lhs,
        rhs,
        window_strides,
        padding,
        lhs_dilation,
        rhs_dilation,
        dimension_numbers,
        feature_group_count,
        batch_group_count,
        preferred_element_type,
        out_sharding,
    )
  else:
    return _slow_conv_general_dilated(
        lhs,
        rhs,
        window_strides,
        padding,
        lhs_dilation,
        rhs_dilation,
        dimension_numbers,
        feature_group_count,
        batch_group_count,
        precision,
        preferred_element_type,
        out_sharding,
    )
