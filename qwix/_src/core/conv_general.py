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

import jax
from jax import numpy as jnp
from qwix._src.core import qarray


def get_how_to_quantize(
    *,
    dimension_numbers: jax.lax.ConvDimensionNumbers,
    for_lhs: bool,
    qtype: jax.typing.DTypeLike,
    calibration_method: str,
) -> qarray.HowToQuantize:
  """Gets how to quantize from conv's dimension_numbers.

  Use channelwise for batch dimension and out feature dimension.

  Args:
    dimension_numbers: The conv's dimension_numbers.
    for_lhs: Whether to quantize lhs or rhs.
    qtype: The logical type of the quantized value.
    calibration_method: The calibration method to use.

  Returns:
    How to quantize lhs or rhs.
  """
  if for_lhs:
    channelwise_axes = [dimension_numbers.lhs_spec[0]]
  else:
    channelwise_axes = [dimension_numbers.rhs_spec[0]]
    if calibration_method == 'minmax':
      raise ValueError('Asymmetric quantization for rhs is not supported.')
  return qarray.HowToQuantize(
      qtype=qtype,
      channelwise_axes=channelwise_axes,
      tiled_axes={},
      calibration_method=calibration_method,
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
  )


def _fast_conv_general_dilated(
    lhs: qarray.QArray,
    rhs: qarray.QArray,
    window_strides: Sequence[int],
    padding: str | Sequence[tuple[int, int]],
    lhs_dilation: Sequence[int] | None = None,
    rhs_dilation: Sequence[int] | None = None,
    dimension_numbers: jax.lax.ConvGeneralDilatedDimensionNumbers = None,
    feature_group_count: int = 1,
    batch_group_count: int = 1,
) -> jax.Array:
  """Quantized jax.lax.conv_general_dilated. Both sides must be QArrays."""
  if qarray.get_tiled_axes(lhs) or qarray.get_tiled_axes(rhs):
    raise ValueError('subchannel is not supported for conv_general_dilated.')

  lhs_value = lhs.qvalue
  lhs_scale = lhs.scale
  lhs_zero_point = lhs.zero_point
  rhs_value = rhs.qvalue
  rhs_scale = rhs.scale
  rhs_zero_point = rhs.zero_point

  if rhs_zero_point is not None:
    raise ValueError('Asymmetric quantization for rhs is not supported.')

  dimension_numbers = jax.lax.conv_dimension_numbers(
      lhs_value.shape, rhs_value.shape, dimension_numbers
  )
  transpose = get_transpose(dimension_numbers, for_lhs=True)
  lhs_scale = qarray.transpose_array(lhs_scale, transpose)
  transpose = get_transpose(dimension_numbers, for_lhs=False)
  rhs_scale = qarray.transpose_array(rhs_scale, transpose)

  if all(x.dtype.name.startswith('int') for x in (lhs_value, rhs_value)):
    acc_type = jnp.int32
  else:
    acc_type = rhs_scale.dtype

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
      preferred_element_type=acc_type,
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
        preferred_element_type=acc_type,
    )

  res *= lhs_scale
  res *= rhs_scale
  return res


def conv_general_dilated(
    lhs: qarray.QArray,
    rhs: qarray.QArray,
    window_strides: Sequence[int],
    padding: str | Sequence[tuple[int, int]],
    lhs_dilation: Sequence[int] | None = None,
    rhs_dilation: Sequence[int] | None = None,
    dimension_numbers: jax.lax.ConvGeneralDilatedDimensionNumbers = None,
    feature_group_count: int = 1,
    batch_group_count: int = 1,
) -> jax.Array:
  """Dispatches to fast or slow conv_general_dilated depending on the inputs."""
  if isinstance(lhs, qarray.QArray) and isinstance(rhs, qarray.QArray):
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
    )
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
  )
