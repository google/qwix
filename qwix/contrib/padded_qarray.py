"""Provides a QArray implementation that supports padding to tile sizes.

This module extends the base qwix.QArray with `PaddedQArray`, which
automatically pads arrays to be multiples of specified tile sizes along
certain axes before quantization and dequantization. It also provides
wrappers for `dot_general` and PTQ functions to utilize this padding
behavior.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import sys
from typing import Any, Mapping, TypeAlias

import flax.struct
import jax
import jax.numpy as jnp
from qwix._src.core import dot_general as core_dot_general
from qwix._src.core import einsum as core_einsum
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


def pad_to_tile(array: jax.Array, tiled_axes: Mapping[int, int | float]) -> jax.Array:
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


def quantize_with_scale_zero_point(
    array: jax.Array,
    qtype: jax.typing.DTypeLike,
    scale: jax.Array,
    zero_point: jax.Array | None,
    noise_fn: numerics.NoiseFn | None = None,
    tile_axes: Mapping[int, int] | None = None
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
  original_shape = array.shape

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
  )


def dequantize(array: PaddedQArray) -> jax.Array:
  """Dequantizes an array. The reverse of |quantize|."""
  qarray.validate_qarray(array)
  original_shape = array.shape
  padded_qvalue = pad_to_tile(array.qvalue, dict(array.tile_axes))
  out = qarray.dequantize(dataclasses.replace(array, qvalue=padded_qvalue))
  # If we padded qvalues, crop back to original shape for user output.
  if out.shape != original_shape:
    out = out[tuple(slice(0, d) for d in original_shape)]
  return out


# ---------------------------
# dot_general and einsum wrappers
# ---------------------------


def _pad_operand(x, tiled_axes):
  if isinstance(x, PaddedQArray):
    padded_q = pad_to_tile(x.qvalue, tiled_axes)
    return dataclasses.replace(x, qvalue=padded_q)
  else:
    return pad_to_tile(x, tiled_axes)

def dot_general(
    lhs: MaybeQArray,
    rhs: MaybeQArray,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    **kwargs,
) -> jax.Array:
  """Pad online based on dimension_numbers and tile_size, then delegate.

  We infer tiled axes using the same helper as core.get_how_to_quantize,
  using the provided `_tile_size`. This lets us pad even if stored qvalues
  are not padded, and it pads raw arrays when only the other side is quantized.
  """
  
  # Infer tile size by inspecting existing quantized operands.
  _tile_size = None
  if isinstance(rhs, PaddedQArray):
    _tile_size = next(iter(rhs.tile_axes.values()))
  elif isinstance(lhs, PaddedQArray):
    _tile_size = next(iter(lhs.tile_axes.values()))

  get_how_to_quantize = functools.partial(
        core_dot_general.get_how_to_quantize,
        dimension_numbers=dimension_numbers,
        ndims=(len(lhs.shape), len(rhs.shape)),
    )
  how_lhs = get_how_to_quantize(
      for_lhs=True,
      qtype=None,
      tile_size=_tile_size,
      calibration_method=None,
  )
  how_rhs = get_how_to_quantize(
      for_lhs=False,
      qtype=None,
      tile_size=_tile_size,
      calibration_method=None,
  )

  lhs = _pad_operand(lhs, how_lhs.tiled_axes)
  rhs = _pad_operand(rhs, how_rhs.tiled_axes)

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
  return core_dot_general._slow_dot_general(  # pylint: disable=protected-access
      lhs,
      rhs,
      dimension_numbers,
      preferred_element_type=preferred_element_type,
      precision=precision,
      **kwargs,
  )


def einsum(
    einsum_str: str,
    lhs: MaybeQArray,
    rhs: MaybeQArray,
    *,
    _qwix_dot_general=core_dot_general.dot_general,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    **kwargs,
) -> jax.Array:
  """Pad online based on einsum_str and tile_size, then delegate to core."""

  # Infer tile size by inspecting existing quantized operands.
  _tile_size = None
  if isinstance(rhs, PaddedQArray):
    _tile_size = next(iter(rhs.tile_axes.values()))
  elif isinstance(lhs, PaddedQArray):
    _tile_size = next(iter(lhs.tile_axes.values()))

  get_how_to_quantize = functools.partial(
        core_einsum.get_how_to_quantize,
        einsum_str=einsum_str,
        ndims=(len(lhs.shape), len(rhs.shape)),
    )
  how_lhs = get_how_to_quantize(
      for_lhs=True,
      qtype=None,
      tile_size=_tile_size,
      calibration_method=None,
  )
  how_rhs = get_how_to_quantize(
      for_lhs=False,
      qtype=None,
      tile_size=_tile_size,
      calibration_method=None,
  )

  lhs = _pad_operand(lhs, how_lhs.tiled_axes)
  rhs = _pad_operand(rhs, how_rhs.tiled_axes)
  
  return core_einsum.einsum(
      einsum_str,
      lhs,
      rhs,
      _qwix_dot_general=_qwix_dot_general,
      preferred_element_type=preferred_element_type,
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

from qwix._src.providers.ptq import PtqProvider
PaddedPtqProvider = functools.partial(
    PtqProvider,
    _qarray_module=sys.modules[__name__],
    _dot_general_fn=dot_general,
    _einsum_fn=einsum,
)


__all__ = [
    'PaddedPtqProvider',
    'PaddedQArray',
]
