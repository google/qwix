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
"""Quantized jax.lax.dot_general with subchannel support."""

from collections.abc import Collection

import jax
from jax import numpy as jnp
from qwix.core import numerics
from qwix.core import qarray


def get_how_to_quantize(
    *,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    ndims: tuple[int, int],
    for_lhs: bool,
    qtype: jax.typing.DTypeLike,
    tile_size: int | float | None,
    calibration_method: str,
    batch_axes: Collection[int],
) -> qarray.HowToQuantize:
  """Get how to quantize from dimension_numbers and remaining_dims.

  By default, use channelwise for all non-contraction axes, and subchannel
  for contraction axes if a tile_size is given.

  Args:
    dimension_numbers: The dimension numbers passed to dot_general.
    ndims: The number of dimensions for lhs and rhs.
    for_lhs: Whether to quantize lhs or rhs.
    qtype: The logical type of the quantized value.
    tile_size: The tile size for subchannel quantization.
    calibration_method: The calibration method to use.
    batch_axes: Batch axes used for calibration.

  Returns:
    How to quantize.
  """
  if for_lhs:
    ndim = ndims[0]
    contracting_axes = dimension_numbers[0][0]
  else:
    ndim = ndims[1]
    contracting_axes = dimension_numbers[0][1]

  channelwise_axes = sorted(
      set(range(ndim)) - set(contracting_axes) - set(batch_axes)
  )
  tiled_axes = {}
  if tile_size:
    tiled_axes = {axis: tile_size for axis in contracting_axes}

  return qarray.HowToQuantize(
      qtype=qtype,
      channelwise_axes=channelwise_axes,
      tiled_axes=tiled_axes,
      batch_axes=batch_axes,
      calibration_method=calibration_method,
  )


def _get_scale_transpose(
    dimension_numbers: jax.lax.DotDimensionNumbers,
    ndims: tuple[int, int],
    for_lhs: bool,
    is_tiled: bool,
) -> list[int | None]:
  """Returns the transpose list for lhs_scale or rhs_scale."""
  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
  lhs_ra = sorted(set(range(ndims[0])) - set(lhs_ca) - set(lhs_ba))
  rhs_ra = sorted(set(range(ndims[1])) - set(rhs_ca) - set(rhs_ba))
  if for_lhs:
    # out is ba + lhs_ra + rhs_ra.
    transpose = list(lhs_ba) + list(lhs_ra) + [None] * len(rhs_ra)
    if is_tiled:
      # When subchannel is enabled, the original scale has one less dimension
      # than ndims here, which is the new contracting axis. Adjust axes after
      # the contracting axis by one.
      transpose = [a - 1 if a and a > lhs_ca[0] else a for a in transpose]
  else:
    transpose = list(rhs_ba) + [None] * len(lhs_ra) + list(rhs_ra)
    if is_tiled:
      transpose = [a - 1 if a and a > rhs_ca[0] else a for a in transpose]
  return transpose


def _apply_subchannel(
    dimension_numbers: jax.lax.DotDimensionNumbers,
) -> jax.lax.DotDimensionNumbers:
  """Apply subchannel to dimension_numbers."""
  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
  if len(lhs_ca) != 1:
    raise ValueError('Only a single tiled axis is supported for now.')
  # 1. Adjust axes because a new axis is added.
  lhs_ba = [a if a < lhs_ca[0] else a + 1 for a in lhs_ba]
  rhs_ba = [a if a < rhs_ca[0] else a + 1 for a in rhs_ba]
  # 2. Add the original ca as new ba.
  lhs_ba += lhs_ca
  rhs_ba += rhs_ca
  # 3. Make the new ca.
  lhs_ca = [lhs_ca[0] + 1]
  rhs_ca = [rhs_ca[0] + 1]
  return (lhs_ca, rhs_ca), (lhs_ba, rhs_ba)


def _fast_dot_general(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    **kwargs,
) -> jax.Array:
  """Dot general in optimized path by computing in quantized types first then dequantize."""
  if isinstance(lhs, qarray.QArray):
    lhs_value = lhs.qvalue
    lhs_scale = lhs.scale
    lhs_zero_point = lhs.zero_point
    lhs_tile_size = qarray.get_single_tile_size(lhs)
  else:
    lhs_value = lhs
    lhs_scale = None
    lhs_zero_point = None
    lhs_tile_size = None
  if isinstance(rhs, qarray.QArray):
    rhs_value = rhs.qvalue
    rhs_scale = rhs.scale
    rhs_zero_point = rhs.zero_point
    rhs_tile_size = qarray.get_single_tile_size(rhs)
  else:
    rhs_value = rhs
    rhs_scale = None
    rhs_zero_point = None
    rhs_tile_size = None

  if lhs_zero_point is not None and rhs_zero_point is not None:
    raise ValueError('Only one operand can be asymmetric.')

  if lhs_scale is not None and rhs_scale is not None:
    if lhs_tile_size != rhs_tile_size:
      # Different tile sizes are not supported for now.
      raise ValueError(f'{lhs_tile_size=} != {rhs_tile_size=}')

  tile_size = lhs_tile_size or rhs_tile_size

  tiled_sum_axis = None
  if tile_size:
    (lhs_ca, rhs_ca), (lhs_ba, _) = dimension_numbers
    # Split lhs/rhs_value.
    lhs_value = qarray.split_axis(lhs_value, {lhs_ca[0]: tile_size})
    if lhs_zero_point is not None:
      lhs_zero_point = qarray.split_axis(lhs_zero_point, {lhs_ca[0]: 1})
    rhs_value = qarray.split_axis(rhs_value, {rhs_ca[0]: tile_size})
    if rhs_zero_point is not None:
      rhs_zero_point = qarray.split_axis(rhs_zero_point, {rhs_ca[0]: 1})
    tiled_sum_axis = len(lhs_ba)
    dimension_numbers = _apply_subchannel(dimension_numbers)

  # Transpose lhs/rhs_scale.
  ndims = (len(lhs_value.shape), len(rhs_value.shape))
  if lhs_scale is not None:
    transpose = _get_scale_transpose(dimension_numbers, ndims, True, tile_size)  # pytype: disable=wrong-arg-types
    lhs_scale = qarray.transpose_array(lhs_scale, transpose)
  if rhs_scale is not None:
    transpose = _get_scale_transpose(dimension_numbers, ndims, False, tile_size)  # pytype: disable=wrong-arg-types
    rhs_scale = qarray.transpose_array(rhs_scale, transpose)

  # We want to override the preferred_element_type to int32 for int x int
  # dot_general, or bfloat16/float32 for fp x fp dot_general.
  if all('int' in x.dtype.name for x in (lhs_value, rhs_value)):
    preferred_element_type = jnp.int32
  elif lhs_scale is not None:
    preferred_element_type = lhs_scale.dtype
  elif rhs_scale is not None:
    preferred_element_type = rhs_scale.dtype

  res = jax.lax.dot_general(
      lhs_value,
      rhs_value,
      dimension_numbers=dimension_numbers,
      preferred_element_type=preferred_element_type,
      **kwargs,
  )

  if lhs_zero_point is not None:
    # TODO(zhuyunx): This value can be constant folded in SRQ scenarios.
    res -= jax.lax.dot_general(
        jnp.broadcast_to(lhs_zero_point, lhs_value.shape),
        rhs_value,
        dimension_numbers=dimension_numbers,
        preferred_element_type=preferred_element_type,
        **kwargs,
    )

  if rhs_zero_point is not None:
    res -= jax.lax.dot_general(
        lhs_value,
        jnp.broadcast_to(rhs_zero_point, rhs_value.shape),
        dimension_numbers=dimension_numbers,
        preferred_element_type=preferred_element_type,
        **kwargs,
    )

  if lhs_scale is not None:
    res *= lhs_scale
  if rhs_scale is not None:
    res *= rhs_scale
  if tiled_sum_axis is not None:
    res = jnp.sum(res, axis=tiled_sum_axis)
  return res


def _slow_dot_general(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    **kwargs,
) -> jax.Array:
  """Dot general in slow path by dequantizing first then computing in floating-point types."""
  if isinstance(lhs, qarray.QArray):
    lhs_value = qarray.dequantize(lhs)
  else:
    lhs_value = lhs
  if isinstance(rhs, qarray.QArray):
    rhs_value = qarray.dequantize(rhs)
  else:
    rhs_value = rhs
  return jax.lax.dot_general(lhs_value, rhs_value, dimension_numbers, **kwargs)


def dot_general(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    **kwargs,
) -> jax.Array:
  """Quantized jax.lax.dot_general.

  Args:
    lhs: The left-hand side, either a jax.Array or QArray.
    rhs: The right-hand side, either a jax.Array or QArray.
    dimension_numbers: The dimension numbers passed to dot_general.
    precision: The precision for jax.lax.dot_general.
    preferred_element_type: The preferred element type for jax.lax.dot_general.
    **kwargs: Additional keyword arguments to dot_general.

  Returns:
    a floating-point jax.Array.
  """
  can_optimize = True

  if isinstance(lhs, qarray.QArray) and not numerics.can_dequant_on_output(
      lhs.qtype
  ):
    can_optimize = False
  if isinstance(rhs, qarray.QArray) and not numerics.can_dequant_on_output(
      rhs.qtype
  ):
    can_optimize = False

  if can_optimize:
    return _fast_dot_general(
        lhs,
        rhs,
        dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
        **kwargs,
    )
  else:
    return _slow_dot_general(
        lhs,
        rhs,
        dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
        **kwargs,
    )
