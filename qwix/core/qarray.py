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
"""Quantized Array."""

import dataclasses
from typing import Collection, Mapping, Sequence, TypeAlias
import flax.struct
import jax
from jax import numpy as jnp
from qwix.core import numerics


@flax.struct.dataclass
class QArray:
  """A quantized array implementation with subchannel support.

  The following conditions hold:
    * qvalue.shape == original.shape
    * len(scale.shape) == len(original.shape)
    * len(scale.shape) == len(zero_point.shape)
    * To enable subchannel quantization, scale and zero_point can be
      "generic broadcasted" to original.shape, which means
        all(o % s == 0 for o, s in zip(original.shape, scale.shape))
    * original ≈ (qvalue - zero_point) * generic_broadcast(
        scale, original.shape)

  Attributes:
    qvalue: The quantized value.
    scale: The scale used to quantize the value.
    zero_point: The quantization value that represents the exact floating-point
      value 0, or None if in symmetric quantization.
    qtype: The logical type of the qvalue, which could be different from the
      dtype used for storage in qvalue.
  """

  qvalue: jax.Array
  scale: jax.Array
  zero_point: jax.Array | None
  qtype: jax.typing.DTypeLike = flax.struct.field(pytree_node=False)


@flax.struct.dataclass
class TransposedQArray(QArray):
  """A QArray that optimizes the inference performance.

  * Tiled axes in qvalue are split into (tile_count, tile_size).
  * The scale is transposed to match the output shape of a specific op
    (e.g. einsum, dot_general, conv_general).
  * Tiled axes in zero_point are split into (tile_count, 1).
  """

  # Because of the transpose, we need to store the original shape.
  original_shape: tuple[int, ...] = flax.struct.field(pytree_node=False)


@dataclasses.dataclass(slots=True, frozen=True, kw_only=True)
class HowToQuantize:
  """Determines how to quantize an array."""

  # The the logical type of the qvalue.
  # E.g. jnp.int8, jnp.int4, jnp.float8_*, nf4, etc.
  # Actual qvalue dtype is determined by the quantization method.
  qtype: jax.typing.DTypeLike
  # Channelwise axes will have individual scales.
  channelwise_axes: Collection[int]
  # Tiled axes have subchannel quantization enabled. The value is a mapping
  # from tiled axis to tile size.
  tiled_axes: Mapping[int, int]
  # Batch axes have shared scales, but are treated differently in calibration,
  # i.e., the mean of quant stats is calculated along the batch axes. An axis
  # can appear only in one of channelwise_axes, tiled_axes, or batch_axes.
  batch_axes: Collection[int]
  # Transpose the scale to match the target side. If set, it should be a list
  # of the same length as the output shape, with elements being the input axis
  # or None.
  scale_transpose: Sequence[int | None] | None
  # The calibration method to use. The format is <method>[,<args>], e.g.
  # "absmax" or "fixed,-10,10". Check calibrate() for supported methods.
  calibration_method: str


ShapeT: TypeAlias = Sequence[int]
MaybeQArray: TypeAlias = jax.Array | QArray


def get_scale_shape(array_shape: ShapeT, how: HowToQuantize) -> ShapeT:
  """Returns the scale shape."""
  if set(how.channelwise_axes) & how.tiled_axes.keys():
    raise ValueError('The same axis cannot be both channelwise and tiled.')
  scale_shape = []
  for axis, dim in enumerate(array_shape):
    if axis in how.channelwise_axes:
      scale_shape.append(dim)
    elif axis in how.tiled_axes:
      if dim % how.tiled_axes[axis] != 0:
        raise ValueError(f'{array_shape} cannot be tiled as {how.tiled_axes}.')
      scale_shape.append(dim // how.tiled_axes[axis])
    else:
      scale_shape.append(1)
  return tuple(scale_shape)


def get_original_shape(array: QArray) -> ShapeT:
  """Returns the original shape given a QArray."""
  if isinstance(array, TransposedQArray):
    return array.original_shape
  return array.qvalue.shape


def transpose_array(
    array: jax.Array, transpose: Sequence[int | None]
) -> jax.Array:
  """Similar to jnp.transpose, but allows missing and new axes in the transpose list."""
  for axis, dim in enumerate(array.shape):
    if axis not in transpose and dim > 1:
      raise ValueError(
          'Diminishing axes must have a dimension of 1, got '
          f'{array.shape=} {transpose=}'
      )
  padded = tuple(transpose) + (None,) * (len(array.shape) - len(transpose))
  missing_idx = set(range(len(padded))) - set(padded)
  padded = tuple(missing_idx.pop() if i is None else i for i in padded)
  if len(array.shape) < len(padded):  # pad the input scale.
    array = array.reshape(array.shape + (1,) * (len(padded) - len(array.shape)))
  array = array.transpose(*padded)
  if len(transpose) < len(padded):  # unpad the output scale.
    array = array.reshape(array.shape[: len(transpose)])
  return array


def split_axis(array: jax.Array, tiled_axes: Mapping[int, int]) -> jax.Array:
  """Reshape the array where the axis is split into (tile_count, tile_size)."""
  new_shape = []
  for axis, dim in enumerate(array.shape):
    if axis in tiled_axes:
      if dim % tiled_axes[axis] != 0:
        raise ValueError(f'{array.shape} cannot be tiled as {tiled_axes}.')
      new_shape.append(dim // tiled_axes[axis])
      new_shape.append(tiled_axes[axis])
    else:
      new_shape.append(dim)
  return array.reshape(new_shape)


def get_tiled_axes(array: QArray) -> Mapping[int, int]:
  """Infers the tiled axes from a QArray.

  Args:
    array: The QArray to infer the tiled axes from.

  Returns:
    A mapping from tiled axis to tile size.
  """
  if isinstance(array, TransposedQArray):
    tiled_axes = {}
    for i, s in enumerate(array.original_shape):
      j = i + len(tiled_axes)  # add the offset of prior tiled axes.
      if array.qvalue.shape[j] != s:
        # j: tile_count, j+1: tile_size
        assert array.qvalue.shape[j] * array.qvalue.shape[j + 1] == s
        tiled_axes[i] = array.qvalue.shape[j + 1]
    return tiled_axes
  tiled_axes = {}
  for i, (j, k) in enumerate(zip(array.qvalue.shape, array.scale.shape)):
    if j != k and k != 1:
      tiled_axes[i] = j // k
  return tiled_axes


def get_single_tile_size(array: QArray) -> int | None:
  """A helper function to get the single tile size from a QArray."""
  tiled_axes = get_tiled_axes(array)
  if not tiled_axes:
    return None
  if len(tiled_axes) > 1:
    raise ValueError('Only a single tiled axis is supported for now.')
  return next(iter(tiled_axes.values()))


def calibrate(array: jax.Array, how: HowToQuantize) -> dict[str, jax.Array]:
  """Calibrates the array.

  Args:
    array: The array to calibrate.
    how: How to quantize the array.

  Returns:
    A dict of quantization statistics, e.g. {'min': ..., 'max': ...} for
    asymmetric quantization, or {'absmax': ...} for symmetric quantization.
  """
  reduce_axes = []  # axes to calibrate.
  batch_axes = []  # axes to calculate the mean of quant stats.
  tiled_axes_offset = 0
  for axis, _ in enumerate(array.shape):
    if axis in how.channelwise_axes:
      continue  # no reduce needed.
    if axis in how.batch_axes:
      batch_axes.append(axis + tiled_axes_offset)
      continue  # batch axes are reduced differently.
    if axis in how.tiled_axes:
      tiled_axes_offset += 1  # reduce the tile_size rather than num_tiles.
    reduce_axes.append(axis + tiled_axes_offset)
  array = split_axis(array, how.tiled_axes)

  # Parse the calibration method.
  method, *args = how.calibration_method.lower().split(',')
  args = [float(a) for a in args]
  if method == 'minmax':
    min_array = jnp.min(array, axis=reduce_axes, keepdims=True)
    max_array = jnp.max(array, axis=reduce_axes, keepdims=True)
    if batch_axes:
      min_array = jnp.mean(min_array, axis=batch_axes, keepdims=True)
      max_array = jnp.mean(max_array, axis=batch_axes, keepdims=True)
    # Ensure min_array <= 0 <= max_array so that 0 can be accurately quantized.
    min_array = jnp.clip(min_array, max=0)
    max_array = jnp.clip(max_array, min=0)
    if args:  # args[0] is the scale factor.
      min_array = min_array * args[0]
      max_array = max_array * args[0]
    return {'min': min_array, 'max': max_array}
  elif method == 'absmax':
    absmax = jnp.max(jnp.abs(array), axis=reduce_axes, keepdims=True)
    if batch_axes:
      absmax = jnp.mean(absmax, axis=batch_axes, keepdims=True)
    if args:  # args[0] is the scale factor.
      absmax = absmax * args[0]
    return {'absmax': absmax}
  elif method == 'rms':
    rms = jnp.sqrt(jnp.mean(jnp.square(array), axis=reduce_axes, keepdims=True))
    if batch_axes:
      rms = jnp.mean(rms, axis=batch_axes, keepdims=True)
    if not args:
      raise ValueError('A scale factor is required for RMS calibration.')
    return {'absmax': rms * args[0]}
  elif method == 'fixed':
    if len(args) != 2:
      raise ValueError('A fixed range is required for fixed calibration.')
    if args[0] > 0 or args[1] < 0 or args[0] >= args[1]:
      raise ValueError('The range must contain 0 and be non-empty.')
    if args[0] + args[1] == 0:
      return {'absmax': jnp.array(args[1], dtype=array.dtype)}
    return {
        'min': jnp.array(args[0], dtype=array.dtype),
        'max': jnp.array(args[1], dtype=array.dtype),
    }
  else:
    raise ValueError(f'Unsupported calibration: {how.calibration_method}')


def compute_scale_zero_point(
    calibration: Mapping[str, jax.Array], qtype: jax.typing.DTypeLike
) -> tuple[jax.Array, jax.Array | None]:
  """Computes the scale and zero_point from the calibration result.

  Args:
    calibration: The calibration returned by calibrate().
    qtype: The dtype of the qvalue.

  Returns:
    A tuple of the scale and zero_point. The zero_point is None in symmetric
    quantization.
  """
  if 'min' in calibration and 'max' in calibration:
    qmin, qmax = numerics.get_asymmetric_bound(qtype)
    scale = (calibration['max'] - calibration['min']) / (qmax - qmin)
    scale = jnp.where(scale == 0, 1, scale)  # Scale shouldn't be 0.
    zero_point = qmin - calibration['min'] / scale
    zero_point = numerics.convert_to(zero_point, qtype)
  elif 'absmax' in calibration:
    qmax = numerics.get_symmetric_bound(qtype)
    scale = calibration['absmax'] / qmax
    # Maybe adding an epsilon (1e-7) is faster?
    scale = jnp.where(scale == 0, 1, scale)  # Scale shouldn't be 0.
    zero_point = None
  else:
    raise ValueError(f'Unsupported calibration: {calibration}')
  return scale, zero_point


def quantize_with_scale_zero_point(
    array: jax.Array,
    how: HowToQuantize,
    scale: jax.Array,
    zero_point: jax.Array | None,
) -> QArray:
  """Quantizes an array with the given scale and zero_point.

  Args:
    array: The array to quantize.
    how: The quantization rule.
    scale: The scale to use.
    zero_point: The zero_point to use.

  Returns:
    The quantized array.
  """
  if not numerics.should_quantize(array.dtype):
    raise ValueError(f'Refuse to quantize: {array.dtype}')
  original_shape = array.shape
  array = split_axis(array, how.tiled_axes)
  # Ensure that the scale has the same dtype as the fp array, because
  # dequantize() uses the scale dtype to reconstruct the original array.
  scale = scale.astype(array.dtype)
  qvalue = array / scale
  if zero_point is not None:
    qvalue = qvalue + zero_point.astype(qvalue.dtype)
  qvalue = numerics.convert_to(qvalue, how.qtype)
  scale_shape = get_scale_shape(original_shape, how)
  scale = scale.reshape(scale_shape)
  if how.scale_transpose:
    scale = transpose_array(scale, how.scale_transpose)
    return TransposedQArray(
        qvalue, scale, zero_point, how.qtype, original_shape=original_shape
    )
  if zero_point is not None:
    zero_point = zero_point.reshape(scale_shape)
  return QArray(qvalue.reshape(original_shape), scale, zero_point, how.qtype)


def quantize(
    array: jax.Array,
    how: HowToQuantize,
) -> QArray:
  """Quantizes an array using a dynamic range."""
  calibration = calibrate(array, how)
  scale, zero_point = compute_scale_zero_point(calibration, how.qtype)
  return quantize_with_scale_zero_point(array, how, scale, zero_point)


def dequantize(array: QArray) -> jax.Array:
  """Dequantizes an array. The reverse of |quantize|.

  Args:
    array: The quantized array to dequantize.

  Returns:
    The dequantized array with the same dtype as the input's scale.
  """
  if isinstance(array, TransposedQArray):
    raise ValueError('TransposedQArray cannot be dequantized.')

  qvalue = numerics.convert_from(array.qvalue, array.qtype)
  qvalue = qvalue.astype(array.scale.dtype)
  tiled_axes = get_tiled_axes(array)
  if not tiled_axes:
    if array.zero_point is not None:
      qvalue -= array.zero_point.astype(array.scale.dtype)
    return qvalue * array.scale
  original_shape = qvalue.shape
  qvalue = split_axis(qvalue, tiled_axes)
  scale = split_axis(array.scale, {a: 1 for a in tiled_axes})
  if array.zero_point is not None:
    zero_point = split_axis(array.zero_point, {a: 1 for a in tiled_axes})
    qvalue -= zero_point.astype(scale.dtype)
  return (qvalue * scale).reshape(original_shape)


def clip_to_calibration(
    array: jax.Array,
    calibration: Mapping[str, jax.Array],
    tiled_axes: Mapping[int, int],
) -> jax.Array:
  """Clips an array to the calibration range."""
  original_shape = array.shape
  array = split_axis(array, tiled_axes)
  if 'min' in calibration and 'max' in calibration:
    array = jnp.clip(array, calibration['min'], calibration['max'])
  elif 'absmax' in calibration:
    array = jnp.clip(array, -calibration['absmax'], calibration['absmax'])
  else:
    raise ValueError(f'Unsupported calibration: {calibration}')
  return array.reshape(original_shape)
