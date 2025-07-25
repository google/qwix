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

from collections.abc import Collection, Sequence

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
) -> tuple[list[int | None], list[int | None]]:
  """Returns the transpose list for lhs_scale and rhs_scale."""
  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
  lhs_ra = sorted(set(range(ndims[0])) - set(lhs_ca) - set(lhs_ba))
  rhs_ra = sorted(set(range(ndims[1])) - set(rhs_ca) - set(rhs_ba))
  return (
      list(lhs_ba) + list(lhs_ra) + [None] * len(rhs_ra),  # lhs_scale_transpose
      list(rhs_ba) + [None] * len(lhs_ra) + list(rhs_ra),  # rhs_scale_transpose
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


def _broadcast_axes(
    array: jax.Array, shape: tuple[int, ...], axes: Collection[int]
) -> jax.Array:
  """Broadcast the given axes in the array to the given shape."""
  target_shape = list(array.shape)
  for a in axes:
    target_shape[a] = shape[a]
  return jnp.broadcast_to(array, target_shape)


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
    lhs_tiled_axes = qarray.get_tiled_axes(lhs)
  else:
    lhs_value = lhs
    lhs_scale = None
    lhs_zero_point = None
    lhs_tiled_axes = {}
  if isinstance(rhs, qarray.QArray):
    rhs_value = rhs.qvalue
    rhs_scale = rhs.scale
    rhs_zero_point = rhs.zero_point
    rhs_tiled_axes = qarray.get_tiled_axes(rhs)
  else:
    rhs_value = rhs
    rhs_scale = None
    rhs_zero_point = None
    rhs_tiled_axes = {}

  if lhs_zero_point is not None and rhs_zero_point is not None:
    raise ValueError('Only one operand can be asymmetric.')

  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers

  if set(lhs_tiled_axes) - set(lhs_ca) or set(rhs_tiled_axes) - set(rhs_ca):
    raise ValueError(
        'Only contracting axes can be tiled for now.'
        f' {lhs_tiled_axes=} {rhs_tiled_axes=} {dimension_numbers=}'
    )

  # Figure out the tiled axes to use for the dot_general. For greater
  # flexibility, we allow a non-tiled axis to be contracted with a tiled axis.
  # However, if both axes are tiled, their tile sizes must be the same.
  for l, r in zip(lhs_ca, rhs_ca):
    lhs_tile_size = lhs_tiled_axes.get(l)
    rhs_tile_size = rhs_tiled_axes.get(r)
    if lhs_tile_size and rhs_tile_size and lhs_tile_size != rhs_tile_size:
      raise ValueError(
          'Contracting axes must be tiled with the same tile size.'
          f' {lhs_tiled_axes=} {rhs_tiled_axes=} {dimension_numbers=}'
      )
    if lhs_tile_size or rhs_tile_size:
      lhs_tiled_axes[l] = lhs_tile_size or rhs_tile_size
      rhs_tiled_axes[r] = lhs_tile_size or rhs_tile_size

  # Split lhs/rhs_value for tiled axes.
  lhs_value = qarray.split_axis(lhs_value, lhs_tiled_axes)
  rhs_value = qarray.split_axis(rhs_value, rhs_tiled_axes)

  # Split lhs/rhs_zero_point for tiled axes.
  if lhs_zero_point is not None:
    lhs_zero_point = qarray.split_axis(
        lhs_zero_point, {a: 1 for a in lhs_tiled_axes}
    )
  if rhs_zero_point is not None:
    rhs_zero_point = qarray.split_axis(
        rhs_zero_point, {a: 1 for a in rhs_tiled_axes}
    )

  # Update dimension_numbers and get sum_axes for tiled axes.
  lhs_ca, lhs_ba, sum_axes = _apply_tiling(lhs_ca, lhs_ba, lhs_tiled_axes)
  rhs_ca, rhs_ba, _ = _apply_tiling(rhs_ca, rhs_ba, rhs_tiled_axes)
  dimension_numbers = (lhs_ca, rhs_ca), (lhs_ba, rhs_ba)

  # Transpose lhs/rhs_scale. This works for tiled axes too.
  lhs_scale_transpose, rhs_scale_transpose = _get_scale_transpose(
      dimension_numbers, (len(lhs_value.shape), len(rhs_value.shape))
  )
  if lhs_scale is not None:
    lhs_scale = qarray.split_axis(lhs_scale, {a: 1 for a in lhs_tiled_axes})
    lhs_scale = qarray.transpose_array(lhs_scale, lhs_scale_transpose)
  if rhs_scale is not None:
    rhs_scale = qarray.split_axis(rhs_scale, {a: 1 for a in rhs_tiled_axes})
    rhs_scale = qarray.transpose_array(rhs_scale, rhs_scale_transpose)

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
        _broadcast_axes(lhs_zero_point, lhs_value.shape, lhs_ca + lhs_ba),
        rhs_value,
        dimension_numbers=dimension_numbers,
        preferred_element_type=preferred_element_type,
        **kwargs,
    )

  if rhs_zero_point is not None:
    res -= jax.lax.dot_general(
        lhs_value,
        _broadcast_axes(rhs_zero_point, rhs_value.shape, rhs_ca + rhs_ba),
        dimension_numbers=dimension_numbers,
        preferred_element_type=preferred_element_type,
        **kwargs,
    )

  if lhs_scale is not None:
    res *= lhs_scale
  if rhs_scale is not None:
    res *= rhs_scale
  if sum_axes:
    res = jnp.sum(res, axis=sum_axes)
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
