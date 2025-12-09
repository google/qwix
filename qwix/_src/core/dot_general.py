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
import itertools
from typing import Any
import jax
from jax import numpy as jnp
from qwix._src.core import numerics
from qwix._src.core import qarray


def get_how_to_quantize(
    *,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    ndims: tuple[int, int],
    for_lhs: bool,
    tile_size: int | float | None,
    **kwargs: Any,
) -> qarray.HowToQuantize:
  """Get how to quantize from dimension_numbers and remaining_dims.

  By default, use channelwise for all non-contraction axes, and subchannel
  for contraction axes if a tile_size is given.

  Args:
    dimension_numbers: The dimension numbers passed to dot_general.
    ndims: The number of dimensions for lhs and rhs.
    for_lhs: Whether to quantize lhs or rhs.
    tile_size: The tile size for subchannel quantization.
    **kwargs: Additional keyword arguments to HowToQuantize.

  Returns:
    How to quantize.
  """
  if for_lhs:
    ndim = ndims[0]
    contracting_axes = dimension_numbers[0][0]
  else:
    ndim = ndims[1]
    contracting_axes = dimension_numbers[0][1]

  channelwise_axes = sorted(set(range(ndim)) - set(contracting_axes))
  tiled_axes = {}
  if tile_size:
    tiled_axes = {contracting_axes[0]: tile_size}

  return qarray.HowToQuantize(
      channelwise_axes=channelwise_axes,
      tiled_axes=tiled_axes,
      **kwargs,
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
  lhs_value = qarray.split_axis(lhs_value, lhs_tiled_ca)
  rhs_value = qarray.split_axis(rhs_value, rhs_tiled_ca)

  # Split lhs/rhs_zero_point for tiled axes.
  if lhs_zero_point is not None:
    lhs_zero_point = qarray.split_axis(
        lhs_zero_point, {a: 1 for a in lhs_tiled_ca}
    )
  if rhs_zero_point is not None:
    rhs_zero_point = qarray.split_axis(
        rhs_zero_point, {a: 1 for a in rhs_tiled_ca}
    )

  # Update dimension_numbers and get sum_axes for tiled axes.
  lhs_ca, lhs_ba, sum_axes = _apply_tiling(lhs_ca, lhs_ba, lhs_tiled_ca)
  rhs_ca, rhs_ba, _ = _apply_tiling(rhs_ca, rhs_ba, rhs_tiled_ca)
  dimension_numbers = (lhs_ca, rhs_ca), (lhs_ba, rhs_ba)

  # Transpose lhs/rhs_scale. This works for tiled axes too.
  lhs_scale_transpose, rhs_scale_transpose = _get_scale_transpose(
      dimension_numbers, (len(lhs_value.shape), len(rhs_value.shape))
  )
  if lhs_scale is not None:
    lhs_scale = qarray.split_axis(lhs_scale, {a: 1 for a in lhs_tiled_ca})
    lhs_scale = qarray.transpose_array(lhs_scale, lhs_scale_transpose)
  if rhs_scale is not None:
    rhs_scale = qarray.split_axis(rhs_scale, {a: 1 for a in rhs_tiled_ca})
    rhs_scale = qarray.transpose_array(rhs_scale, rhs_scale_transpose)

  preferred_element_type, result_type = qarray.get_accumulator_and_result_type(
      lhs, rhs, preferred_element_type=preferred_element_type
  )

  res = jax.lax.dot_general(
      lhs_value,
      rhs_value,
      dimension_numbers=dimension_numbers,
      preferred_element_type=preferred_element_type,
      **kwargs,
  )

  if lhs_zero_point is not None:
    # TODO(zhuyunx): This value can be constant folded in SRQ scenarios.
    res = qarray.call_with_generic_broadcast(
        jnp.subtract,
        res,
        jax.lax.dot_general(
            _broadcast_axes(lhs_zero_point, lhs_value.shape, lhs_ca + lhs_ba),
            rhs_value,
            dimension_numbers=dimension_numbers,
            preferred_element_type=preferred_element_type,
            **kwargs,
        ),
    )

  if rhs_zero_point is not None:
    res = qarray.call_with_generic_broadcast(
        jnp.subtract,
        res,
        jax.lax.dot_general(
            lhs_value,
            _broadcast_axes(rhs_zero_point, rhs_value.shape, rhs_ca + rhs_ba),
            dimension_numbers=dimension_numbers,
            preferred_element_type=preferred_element_type,
            **kwargs,
        ),
    )

  if lhs_scale is not None:
    res = qarray.call_with_generic_broadcast(jnp.multiply, res, lhs_scale)
  if rhs_scale is not None:
    res = qarray.call_with_generic_broadcast(jnp.multiply, res, rhs_scale)
  if sum_axes:
    res = jnp.sum(res, axis=sum_axes)
  return res.astype(result_type)


def _slow_dot_general(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    **kwargs,
) -> jax.Array:
  """Dot general in slow path by dequantizing first then computing in floating-point types."""
  if isinstance(lhs, qarray.QArray):
    lhs = qarray.dequantize(lhs)
  if isinstance(rhs, qarray.QArray):
    rhs = qarray.dequantize(rhs)
  return jax.lax.dot_general(lhs, rhs, dimension_numbers, **kwargs)


def loop_dot_general(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    **kwargs,
) -> jax.Array:
  """Loop-based tiled dot general implementation for the internal accumulation loop.

  This function computes the dot product by iterating over the contracting
  dimensions
  (the temporal loop), while assuming the inputs are already spatially sharded.

  **Input State:**
  The `lhs` and `rhs` inputs to this function are expected to be **Spatial
  Shards**
  (chunks of the M and N dimensions) determined by the Pallas grid, but they
  contain the **Full Contracting Dimension** (K) (or large chunks of it).

  **Why loop?**
  We cannot simply pass the full K-dimension to the hardware in one shot
  because:
  1.  **MXU Constraints:** The TPU Matrix Unit (MXU) operates on fixed tile
  sizes
      (typically 128x128). A large contracting dimension (e.g., K=512 or K=1024)
      must be broken down into these smaller hardware-native tiles.
  2.  **Quantization/Accumulation:** For quantized operations, we often need to
      dequantize and accumulate partial results periodically (e.g., every 128
      steps)
      to prevent integer overflow or precision loss in the accumulator.

  Args:
    lhs: The left-hand side input (spatially sharded).
    rhs: The right-hand side input (spatially sharded).
    dimension_numbers: The contracting/batch dims.
    preferred_element_type: Accumulator dtype.
    **kwargs: Extra args for jax.lax.dot_general.

  Returns:
    The accumulated result of the dot product.
  """
  if isinstance(lhs, qarray.QArray):
    lhs_value = lhs.qvalue
    lhs_scale = lhs.scale
    assert lhs.zero_point is None
    lhs_tiled_axes = qarray.get_tiled_axes(lhs)
  else:
    lhs_value = lhs
    lhs_scale = None
    lhs_tiled_axes = {}
  if isinstance(rhs, qarray.QArray):
    rhs_value = rhs.qvalue
    rhs_scale = rhs.scale
    assert rhs.zero_point is None
    rhs_tiled_axes = qarray.get_tiled_axes(rhs)
  else:
    rhs_value = rhs
    rhs_scale = None
    rhs_tiled_axes = {}

  lhs_ca, rhs_ca = dimension_numbers[0]

  # Allow non-tiled axes to be contracted with tiled axes.
  ca_tile_counts = []  # number of tiles for each contracting axis.
  for l, r in zip(lhs_ca, rhs_ca):
    lhs_tile_size = lhs_tiled_axes.get(l)
    rhs_tile_size = rhs_tiled_axes.get(r)
    if lhs_tile_size and rhs_tile_size and lhs_tile_size != rhs_tile_size:
      raise ValueError(
          'Contracting axes must be tiled with the same tile size.'
          f' {lhs_tiled_axes=} {rhs_tiled_axes=} {dimension_numbers=}'
      )
    tile_size = lhs_tile_size or rhs_tile_size
    if tile_size is not None:
      ca_tile_counts.append(lhs_value.shape[l] // tile_size)
    else:
      ca_tile_counts.append(1)

  preferred_element_type, result_type = qarray.get_accumulator_and_result_type(
      lhs, rhs, preferred_element_type=preferred_element_type
  )

  lhs_scale_transpose, rhs_scale_transpose = _get_scale_transpose(
      dimension_numbers, (len(lhs_value.shape), len(rhs_value.shape))
  )

  def take_slice(
      array: jax.Array, ca: Sequence[int], ca_tile_indices: Sequence[int]
  ) -> jax.Array:
    indices = []
    for i, s in enumerate(array.shape):
      if i not in ca or s == 1:
        indices.append(slice(None))
        continue
      index = ca_tile_indices[ca.index(i)]
      count = ca_tile_counts[ca.index(i)]
      assert s % count == 0
      size = s // count
      indices.append(slice(index * size, (index + 1) * size))
    return array[tuple(indices)]

  acc = None
  for ca_tile_indices in itertools.product(*map(range, ca_tile_counts)):
    # ca_tile_indices is a list of tile indices for each contracting axis.
    out = jax.lax.dot_general(
        take_slice(lhs_value, lhs_ca, ca_tile_indices),
        take_slice(rhs_value, rhs_ca, ca_tile_indices),
        dimension_numbers=dimension_numbers,
        preferred_element_type=preferred_element_type,
        **kwargs,
    )
    if lhs_scale is not None:
      scale = take_slice(lhs_scale, lhs_ca, ca_tile_indices)
      scale = qarray.transpose_array(scale, lhs_scale_transpose)
      out = qarray.call_with_generic_broadcast(jnp.multiply, out, scale)
    if rhs_scale is not None:
      scale = take_slice(rhs_scale, rhs_ca, ca_tile_indices)
      scale = qarray.transpose_array(scale, rhs_scale_transpose)
      out = qarray.call_with_generic_broadcast(jnp.multiply, out, scale)
    acc = out if acc is None else acc + out
  assert acc is not None
  return acc.astype(result_type)


# If a contracting dimension has a tile size smaller than this threshold, tiled
# dot general will be inefficient and we should dequantize the input first.
MIN_TILE_SIZE_TO_DEQUANT_ON_OUTPUT = 128


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
  # We need to choose between slow_dot_general, which dequantizes first and
  # then computes in floating-point types, and fast_dot_general, which
  # computes in quantized types first and then dequantize.
  use_fast_dot_general = True
  for operand, ca in zip((lhs, rhs), dimension_numbers[0]):
    if not isinstance(operand, qarray.QArray):
      if numerics.should_quantize(operand.dtype):
        # Always dequantize on inputs if any of the operands is in bf16/fp32,
        # because XLA is able to fuse the dequantize and the matmul. The slow
        # path is usually not slower than the fast path, since both use fp
        # matmul, and will be significantly faster when subchannel or zero_point
        # is used.
        use_fast_dot_general = False
        break
      # For raw arrays in lower precision, e.g. fp8, int4, bool, using fast path
      # may be beneficial.
      continue

    qarray.validate_qarray(operand)

    # qtypes like nf4 cannot be dequantized on output.
    if not numerics.can_dequant_on_output(operand.qtype):
      use_fast_dot_general = False
      break

    # If a contracting dimension is tiled too small, tiled dot general will
    # be inefficient and we should dequantize the input first. This is critical
    # when a contracting dimension is channelwise quantized, e.g. tile_size=1.
    for axis in ca:
      if operand.scale.shape[axis] > 1:
        tile_size = operand.qvalue.shape[axis] // operand.scale.shape[axis]
        if tile_size < MIN_TILE_SIZE_TO_DEQUANT_ON_OUTPUT:
          use_fast_dot_general = False
          break

  if use_fast_dot_general:
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
