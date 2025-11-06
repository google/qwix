"""Provides a QArray implementation that supports padding to tile sizes.

This module extends the base qwix.QArray with `PaddedQArray`, which
automatically pads arrays to be multiples of specified tile sizes along
certain axes before quantization and dequantization. It also provides
wrappers for `dot_general` and PTQ functions to utilize this padding
behavior.
"""

from __future__ import annotations

import functools
import os
import sys
from typing import Any, Mapping, TypeAlias

import flax.struct
import jax
import jax.numpy as jnp
import qwix
from qwix._src.core import dot_general as core_dot_general
from qwix._src.core import numerics
from qwix._src.core import qarray
from qwix._src.providers import ptq as _ptq


calibrate = qarray.calibrate
HowToQuantize = qarray.HowToQuantize


# ---------------------------
# Padded QArray implementation
# ---------------------------


@flax.struct.dataclass
class PaddedQArray(qarray.QArray):
  """Quantized array supporting padding and tracking original tile sizes.

  Additional Attributes:
    tile_axes: field to store the original tile sizes.
  """

  tile_axes: Mapping[int, int] | float = flax.struct.field(
      pytree_node=False, default_factory=dict
  )


MaybeQArray: TypeAlias = jax.Array | qarray.QArray | PaddedQArray


def pad_to_tile(
    array: jax.Array, tiled_axes: Mapping[int, int | float]
) -> jax.Array:
  """Pads array along tiled axes so each dimension is a multiple of tile size."""
  if not tiled_axes:
    return array
  pad_width = [(0, 0)] * array.ndim
  for axis, tile_size in tiled_axes.items():
    if isinstance(tile_size, float):
      tile_size = round(array.shape[axis] * tile_size)
    dim = array.shape[axis]
    remainder = dim % tile_size
    if remainder > 0:
      pad_width[axis] = (0, tile_size - remainder)
  if all(p == (0, 0) for p in pad_width):
    return array
  return jnp.pad(array, pad_width, constant_values=0)


def get_tiled_axes(array: MaybeQArray) -> Mapping[int, int] | float:
  """Gets the tiled axes from a MaybeQArray."""
  if isinstance(array, PaddedQArray):
    return dict(array.tile_axes)
  if isinstance(array, qarray.QArray):
    return qarray.get_tiled_axes(array)
  return {}


def quantize_with_scale_zero_point(
    array: jax.Array,
    qtype: jax.typing.DTypeLike,
    scale: jax.Array,
    zero_point: jax.Array | None,
    noise_fn: numerics.NoiseFn | None = None,
    tile_axes: Mapping[int, int] | None = None,
    original_shape: tuple[int, ...] | None = None,
) -> PaddedQArray:
  """Quantizes an array with the given scale and zero_point with padding support.

  Applies quantization to padded values and stores tile_axes.
  Optionally saves qvalues in padded form based on env `QARRAY_STORE_PADDED`.

  Args:
    array: The array to quantize.
    qtype: The quantized dtype.
    scale: The scale factor for quantization.
    zero_point: The zero point for quantization, or None.
    noise_fn: Optional function to add noise during quantization.
    tile_axes: Mapping from axis to tile size for padding.
    original_shape: The original shape of the array before padding, if
      applicable.

  Returns:
    A PaddedQArray instance.
  """
  if not numerics.should_quantize(array.dtype):
    raise ValueError(f'Refuse to quantize: {array.dtype}')
  if zero_point is not None and zero_point.shape != scale.shape:
    raise ValueError(
        f'Expect zero_point shape {scale.shape} but got {zero_point.shape}'
    )

  # Ensure that the scale has the same dtype as the fp array, because
  # dequantize() uses the scale dtype to reconstruct the original array.
  scale = scale.astype(array.dtype)

  tile_axes = tile_axes or {}
  padded_array = pad_to_tile(array, tile_axes)
  qvalue = qarray.call_with_generic_broadcast(jnp.divide, padded_array, scale)
  if zero_point is not None:
    qvalue = qarray.call_with_generic_broadcast(
        jnp.add, qvalue, zero_point.astype(qvalue.dtype)
    )
  qvalue = numerics.convert_to(qvalue, qtype, noise_fn)

  # Slice back to original shape if not storing padded version
  store_padded = os.environ.get('QARRAY_STORE_PADDED', '0') == '1'
  if not store_padded and original_shape is not None:
    qvalue = qvalue[tuple(slice(0, dim) for dim in original_shape)]

  return PaddedQArray(qvalue, scale, zero_point, qtype, tile_axes=tile_axes)


def quantize(array: jax.Array, how: HowToQuantize) -> PaddedQArray:
  """Quantizes an array using a dynamic range with padding support."""
  padded_array = pad_to_tile(array, how.tiled_axes)
  calibration = calibrate(padded_array, how)
  scale, zero_point = qarray.compute_scale_zero_point(calibration, how.qtype)
  return quantize_with_scale_zero_point(
      padded_array,
      how.qtype,
      scale,
      zero_point,
      how.noise_fn,
      tile_axes=how.tiled_axes,
      original_shape=array.shape,
  )


def dequantize(array: PaddedQArray) -> jax.Array:
  """Dequantizes an array. The reverse of |quantize|."""
  qarray.validate_qarray(array)
  padded_qvalue = pad_to_tile(array.qvalue, dict(array.tile_axes))
  qvalue = numerics.convert_from(padded_qvalue, array.qtype)
  qvalue = qvalue.astype(array.scale.dtype)
  if array.zero_point is not None:
    qvalue = qarray.call_with_generic_broadcast(
        jnp.subtract, qvalue, array.zero_point.astype(qvalue.dtype)
    )
  return qarray.call_with_generic_broadcast(jnp.multiply, qvalue, array.scale)


# ---------------------------
# dot_general wrappers
# ---------------------------


def pad_if_needed(arr):
  """Pads the qvalue of a PaddedQArray if tile_axes are present.

  Args:
    arr: The array to potentially pad. Can be a PaddedQArray or other types.

  Returns:
    The array with its qvalue padded if it's a PaddedQArray with tile_axes,
    otherwise the original array.
  """
  if isinstance(arr, PaddedQArray):
    tile_axes = get_tiled_axes(arr)
    if tile_axes:
      padded_qvalue = pad_to_tile(arr.qvalue, tile_axes)
      if padded_qvalue.shape != arr.qvalue.shape:
        arr = PaddedQArray(
            padded_qvalue,
            arr.scale,
            arr.zero_point,
            arr.qtype,
            tile_axes=tile_axes,
        )
  return arr


def dot_general(
    lhs: MaybeQArray,
    rhs: MaybeQArray,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    **kwargs,
) -> jax.Array:
  """Quantized jax.lax.dot_general with padding support via PaddedQArray.

  Pads PaddedQArray operands to tile-aligned shapes and lets the user choose
  fast vs slow path via env `QARRAY_USE_FAST_DOT_GENERAL`.

  Args:
    lhs: The left-hand side operand, potentially a PaddedQArray.
    rhs: The right-hand side operand, potentially a PaddedQArray.
    dimension_numbers: A tuple of `((lhs_contracting_dims,
      rhs_contracting_dims), (lhs_batch_dims, rhs_batch_dims))`.
    precision: Optional. See `jax.lax.dot_general`.
    preferred_element_type: Optional. See `jax.lax.dot_general`.
    **kwargs: Additional keyword arguments passed to the underlying
      `core_dot_general`.

  Returns:
    The result of the dot_general operation.
  """
  lhs = pad_if_needed(lhs)
  rhs = pad_if_needed(rhs)
  use_fast = os.environ.get('QARRAY_USE_FAST_DOT_GENERAL', '1') == '1'
  if use_fast:
    return core_dot_general._fast_dot_general(  # pylint: disable=protected-access
        lhs,
        rhs,
        dimension_numbers,
        preferred_element_type=preferred_element_type,
        precision=precision,
        **kwargs,
    )
  else:
    return core_dot_general._slow_dot_general(  # pylint: disable=protected-access
        lhs,
        rhs,
        dimension_numbers,
        preferred_element_type=preferred_element_type,
        precision=precision,
        **kwargs,
    )


def quantize_act(
    array: jax.Array,
    how: HowToQuantize,
    rule,
    act_name: str | None,
):
  """Wrapper to reuse PTQ.quantize_act with this module as qarray backend."""
  return _ptq.quantize_act(
      array, how, rule, act_name, _qarray_module=sys.modules[__name__]
  )


def create_quantized_param(
    name: str,
    value: jax.Array,
    how: HowToQuantize,
) -> _ptq.WithAux[qarray.QArray]:
  """Wrapper that delegates to PTQ.create_quantized_param using this backend."""
  return _ptq.create_quantized_param(
      name, value, how, _qarray_module=sys.modules[__name__]
  )


def quantize_params(
    params: Any,
    abstract_quantized_params: Any,
    quant_stats: Any = flax.core.FrozenDict(),
) -> Any:
  """Wrapper that delegates to PTQ.quantize_params using this backend."""
  return _ptq.quantize_params(
      params,
      abstract_quantized_params,
      quant_stats,
      _qarray_module=sys.modules[__name__],
  )


# ---------------------------
# Provider
# ---------------------------

PaddedPtqProvider = functools.partial(
    qwix.PtqProvider,
    _qarray_module=sys.modules[__name__],
    _dot_general_fn=dot_general,
)


__all__ = [
    'PaddedPtqProvider',
    'PaddedQArray',
]
