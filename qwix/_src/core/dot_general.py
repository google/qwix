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

# pylint: disable=line-too-long

from collections.abc import Collection, Mapping, Sequence
import itertools
from typing import Any
import jax
from jax import numpy as jnp
from qwix._src.core import mxfp_dot
from qwix._src.core import numerics
from qwix._src.core import qarray


def get_how_to_quantize(
    *,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    ndims: tuple[int, int],
    for_lhs: bool,
    tile_size: Mapping[int, int | float] | int | float | None,
    **kwargs: Any,
) -> qarray.HowToQuantize:
  """Get how to quantize from dimension_numbers and remaining_dims.

  By default, use channelwise for all non-contraction axes, and subchannel
  for the innermost (last) contraction axis if a tile_size is given.

  Args:
    dimension_numbers: The dimension numbers passed to dot_general.
    ndims: The number of dimensions for lhs and rhs.
    for_lhs: Whether to quantize lhs or rhs.
    tile_size: The tile size for subchannel quantization. If the tile_size is a
      Mapping, then this uses the tiling specified by tile_size. Otherwise this
      is interpreted as the tile size for the innermost (last) axis in
      contracting_axes.
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

  if isinstance(tile_size, Mapping):
    tiled_axes = tile_size
  else:
    tiled_axes = {}
    if tile_size:
      tiled_axes = {contracting_axes[-1]: tile_size}

  channelwise_axes = sorted(
      set(range(ndim)) - set(contracting_axes) - set(tiled_axes.keys())
  )

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


def apply_rhs_scale_to_lhs(
    lhs: jax.Array,
    rhs_scale: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers,
) -> jax.Array:
  """Multiplies the scale of rhs into lhs.

  dot(lhs, rhs_qvalue * rhs_scale) == dot(lhs * lhs_scale, rhs_qvalue), where
  lhs_scale is rhs_scale transposed onto the contracting and batch axes of lhs,
  as long as rhs_scale has size 1 on the remaining (non-contracting, non-batch)
  axes of rhs, which lhs doesn't have. Along the contracting and batch axes,
  the scale can have any number of values (channelwise or tiled), since it is
  generic broadcasted to lhs.

  Args:
    lhs: The left-hand side of dot_general.
    rhs_scale: The scale of the right-hand side. It must have size 1 on the
      remaining axes of rhs.
    dimension_numbers: The dimension numbers of dot_general.

  Returns:
    lhs multiplied by the scale, in the promoted dtype of lhs and rhs_scale.
  """
  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
  lhs_axes_to_rhs = dict(zip(lhs_ca, rhs_ca)) | dict(zip(lhs_ba, rhs_ba))
  lhs_scale = qarray.transpose_array(
      rhs_scale, [lhs_axes_to_rhs.get(a) for a in range(lhs.ndim)]
  )
  return qarray.call_with_generic_broadcast(jnp.multiply, lhs, lhs_scale)


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
  lhs_value = qarray.split_axis(lhs_value, lhs_tiled_ca)  # pyrefly: ignore[bad-argument-type]
  rhs_value = qarray.split_axis(rhs_value, rhs_tiled_ca)  # pyrefly: ignore[bad-argument-type]

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


def _fold_scale_dot_general(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    **kwargs,
) -> jax.Array:
  """Dot general that folds the scale of the QArray into the other operand.

  Exactly one operand must be a QArray, without zero_point and with a scale of
  size 1 on all of its remaining (non-contracting, non-batch) axes, see
  _should_fold_scale_into_other_operand. Its scale is multiplied into the other,
  unquantized operand, so that the matmul consumes the raw quantized values as
  in _fast_dot_general, even though the scale varies along a contracting axis
  and cannot be applied to the output. The unquantized operand is scaled in the
  promoted dtype of itself and the scale, i.e. the dtype in which
  _slow_dot_general would compute the matmul.

  Args:
    lhs: The left-hand side, either a jax.Array or QArray.
    rhs: The right-hand side, either a jax.Array or QArray.
    dimension_numbers: The dimension numbers passed to dot_general.
    preferred_element_type: The preferred element type for dot_general.
    **kwargs: Additional keyword arguments to jax.lax.dot_general.

  Returns:
    a floating-point jax.Array.
  """
  preferred_element_type, result_type = qarray.get_accumulator_and_result_type(
      lhs, rhs, preferred_element_type=preferred_element_type
  )
  if isinstance(rhs, qarray.QArray):
    assert rhs.zero_point is None and not isinstance(lhs, qarray.QArray)
    lhs = apply_rhs_scale_to_lhs(lhs, rhs.scale, dimension_numbers)
    rhs = rhs.qvalue
  else:
    assert isinstance(lhs, qarray.QArray) and lhs.zero_point is None
    (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
    rhs = apply_rhs_scale_to_lhs(
        rhs, lhs.scale, ((rhs_ca, lhs_ca), (rhs_ba, lhs_ba))
    )
    lhs = lhs.qvalue
  res = jax.lax.dot_general(
      lhs,
      rhs,
      dimension_numbers=dimension_numbers,
      preferred_element_type=preferred_element_type,
      **kwargs,
  )
  return res.astype(result_type)


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


# Default minimum tile size for architectures without a specific entry (128).
_DEFAULT_MIN_TILE_SIZE_TO_DEQUANT_ON_OUTPUT = 128

# Hardware-specific minimum contracting tile sizes for output dequantization.
_MIN_TILE_SIZE_BY_DEVICE_KIND: dict[str, int] = {
    'TPU v6 lite': 256,
    'TPU v6e': 256,
    'TPU7': 512,
    'TPU7x': 512,
    # TODO: b/556893676 - The TPU8i/TPU8t values are provisional and may change
    # once they are benchmarked.
    'TPU8i': 256,
    'TPU8t': 256,
}


def _get_device_kind() -> str:
  """Returns the abstract mesh's device kind, else the default backend's.

  Note: Ahead-of-time / compile-only tracing without an active abstract mesh
  falls back to the default backend device (e.g. host CPU), which may differ
  from the target topology.
  """
  if (ad := jax.sharding.get_abstract_mesh().abstract_device) is not None:
    return ad.device_kind
  # Uses the default backend's first device; does not honor jax.default_device.
  return jax.devices()[0].device_kind


def get_min_tile_size_to_dequant_on_output() -> int:
  """Returns the minimum contracting tile size for output dequantization.

  Returns:
    The minimum contracting tile size based on active device kind, falling
    back to 128 for unrecognized architectures.
  """
  kind = _get_device_kind().split('\n', 1)[0]
  return _MIN_TILE_SIZE_BY_DEVICE_KIND.get(
      kind, _DEFAULT_MIN_TILE_SIZE_TO_DEQUANT_ON_OUTPUT
  )


def _should_use_fast_dot_general(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    dimension_numbers: jax.lax.DotDimensionNumbers,
) -> bool:
  """Returns whether dot_general should use the fast path.

  The fast path computes in quantized types and applies the scales of the
  QArray operands on the output, instead of dequantizing them first.

  Args:
    lhs: The left-hand side of dot_general.
    rhs: The right-hand side of dot_general.
    dimension_numbers: The dimension numbers of dot_general.

  Returns:
    True if at least one operand is a QArray and it is beneficial to apply the
    scales of every QArray operand on the output. Without a QArray operand there
    is nothing to dequantize, so this returns False.
  """
  # Without a QArray operand there is nothing to dequantize, so fall back to
  # jax.lax.dot_general (slow path) and follow JAX's default dtype rules.
  if not isinstance(lhs, qarray.QArray) and not isinstance(rhs, qarray.QArray):
    return False
  for operand, contracting_axes, other_operand in zip(
      (lhs, rhs), dimension_numbers[0], (rhs, lhs)
  ):
    # Only QArray operands have scales to check. A jax.Array operand is
    # handled when checking the other operand, the QArray.
    if not isinstance(operand, qarray.QArray):
      continue
    qarray.validate_qarray(operand)
    # qtypes like nf4 cannot be dequantized on output.
    if not numerics.can_dequant_on_output(operand.qtype):
      return False
    # Tile sizes of the contracting axes with more than one scale (subchannel).
    tile_sizes = [
        operand.qvalue.shape[axis] // operand.scale.shape[axis]
        for axis in contracting_axes
        if operand.scale.shape[axis] > 1
    ]
    # The other operand is a jax.Array in a dtype that Qwix would quantize
    # (bf16/f16/f32/f64). Raw low-precision arrays (e.g. fp8, int8) are treated
    # as quantized with an implicit scale of 1.
    is_other_operand_unquantized = not isinstance(
        other_operand, qarray.QArray
    ) and numerics.should_quantize(other_operand.dtype)
    # With an unquantized other operand, the matmul runs in floating point
    # either way, so only dequantize on output if it adds no work: zero_point
    # needs an extra matmul and tiled contracting axes need per-tile rescaling.
    # Otherwise, this passes the raw quantized values to XLA, e.g. for int4
    # nibble loading on newer TPUs. Scales along a contracting axis that can be
    # folded into the unquantized operand instead are handled by
    # _should_fold_scale_into_other_operand.
    # TODO: b/570154253 - Unquantized x sub-channel QArray could also use the
    # fast path for small M (decode); this needs an M-aware heuristic or a
    # kernel.
    if is_other_operand_unquantized and (
        operand.zero_point is not None or tile_sizes
    ):
      return False
    # Small tiles make the tiled matmul inefficient, e.g. tile_size=1 when a
    # contracting axis is channelwise quantized. The threshold is device
    # dependent.
    if tile_sizes and (
        min(tile_sizes) < get_min_tile_size_to_dequant_on_output()
    ):
      return False
  return True


def _should_fold_scale_into_other_operand(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    dimension_numbers: jax.lax.DotDimensionNumbers,
) -> bool:
  """Returns whether dot_general should fold the scale into the other operand.

  This covers an unquantized x QArray dot_general that the fast path rejects
  because the scale of the QArray varies along a contracting axis, e.g.
  attention values with per-token scales, or channelwise weights contracted
  along their channel axis in the QT backward pass. Instead of dequantizing the
  QArray first, which XLA may materialize in HBM, the scale is multiplied into
  the unquantized operand and the matmul consumes the raw quantized values, as
  in the fast path.

  Args:
    lhs: The left-hand side of dot_general.
    rhs: The right-hand side of dot_general.
    dimension_numbers: The dimension numbers of dot_general.

  Returns:
    True if exactly one operand is a QArray without zero_point that can be
    dequantized on output, the other operand is an unquantized jax.Array, and
    the scale of the QArray has more than one value along a contracting axis but
    size 1 on all of its remaining (non-contracting, non-batch) axes, so that it
    can be moved to the other operand without changing the result.
  """
  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
  if isinstance(lhs, qarray.QArray) and not isinstance(rhs, qarray.QArray):
    operand, other_operand, ca, ba = lhs, rhs, lhs_ca, lhs_ba
  elif isinstance(rhs, qarray.QArray) and not isinstance(lhs, qarray.QArray):
    operand, other_operand, ca, ba = rhs, lhs, rhs_ca, rhs_ba
  else:
    return False
  # Raw low-precision arrays (e.g. fp8, int8) are treated as quantized with an
  # implicit scale of 1 and are handled by the fast or slow path.
  if not numerics.should_quantize(other_operand.dtype):
    return False
  qarray.validate_qarray(operand)
  # A zero_point cannot be folded into the other operand, and qtypes like nf4
  # cannot be dequantized on output.
  if operand.zero_point is not None or not numerics.can_dequant_on_output(
      operand.qtype
  ):
    return False
  # Scales that are constant along the contracting axes are applied on the
  # output by the fast path instead.
  if all(operand.scale.shape[axis] == 1 for axis in ca):
    return False
  # The scale must not vary along the remaining axes, which the other operand
  # doesn't have. Along the contracting and batch axes it can have any number
  # of values (channelwise or tiled), since it is generic broadcasted to the
  # other operand, which costs the same for any tile size.
  remaining_axes = set(range(operand.ndim)) - set(ca) - set(ba)
  return all(operand.scale.shape[axis] == 1 for axis in remaining_axes)


def dot_general(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    **kwargs,
) -> jax.Array:
  """Computes a general dot product with support for ``QArray`` inputs.

  This function serves as a drop-in replacement for
  `jax.lax.dot_general
  <https://docs.jax.dev/en/latest/_autosummary/jax.lax.dot_general.html>`_.

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
  if kwargs.get('multipass_mode') is not None:
    if isinstance(lhs, qarray.QArray) or isinstance(rhs, qarray.QArray):
      raise ValueError(
          'Inputs to multipass dot_general must strictly be unquantized'
          f' jax.Array, but got lhs={type(lhs)}, rhs={type(rhs)}.'
      )
    from qwix._src.core import multipass_dot  # pylint: disable=g-import-not-at-top

    return multipass_dot.multipass_dot_general(
        lhs,
        rhs,
        dimension_numbers=dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
        **kwargs,
    )

  # Try hardware-accelerated MXFP dot.
  mxfp_result = mxfp_dot.mxfp_dot_general(
      lhs, rhs, dimension_numbers, preferred_element_type
  )
  if mxfp_result is not None:
    return mxfp_result

  # We need to choose between three implementations:
  #   * fast_dot_general computes in quantized types first and then applies
  #     the scales of the QArray operands to the output.
  #   * fold_scale_dot_general, for unquantized x QArray only, multiplies the
  #     scale of the QArray into the unquantized operand and then computes on
  #     the raw quantized values. This covers scales that vary along a
  #     contracting axis, which cannot be applied to the output, without
  #     dequantizing the QArray, which XLA may otherwise materialize in HBM.
  #   * slow_dot_general dequantizes first and then computes in floating-point
  #     types.
  # Without a QArray there is nothing to dequantize, so use a plain dot_general.
  if _should_use_fast_dot_general(lhs, rhs, dimension_numbers):
    return _fast_dot_general(
        lhs,
        rhs,
        dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
        **kwargs,
    )
  elif _should_fold_scale_into_other_operand(lhs, rhs, dimension_numbers):
    return _fold_scale_dot_general(
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
