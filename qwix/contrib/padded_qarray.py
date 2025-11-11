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
"""QArray implementation with padding to tile sizes.

Extends qwix.QArray with PaddedQArray for automatic padding to tile size
multiples before quantization. Provides wrappers for dot_general and PTQ
functions.
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
from qwix._src.core import qarray
from qwix._src.providers import ptq as _ptq


PtqProvider = _ptq.PtqProvider
calibrate = qarray.calibrate
HowToQuantize = qarray.HowToQuantize

# Permanently keep shapes in padded form.
_QARRAY_KEEP_PADDED_SHAPE = False
# Which dot_general implementation to use.
_QARRAY_USE_FAST_DOT_GENERAL = True


# ---------------------------
# Padded QArray implementation
# ---------------------------


@flax.struct.dataclass
class PaddedQArray(qarray.QArray):
  """Quantized array with padding support.

  Attributes:
    tile_axes: Maps axis to tile size for padding.
  """

  tile_axes: Mapping[int, int] | float = flax.struct.field(
      pytree_node=False, default_factory=dict
  )
  original_shape: tuple[int, ...] = flax.struct.field(
      pytree_node=False, default=()
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


def quantize(array: jax.Array, how: HowToQuantize) -> PaddedQArray:
  """Quantizes an array using a dynamic range with padding support."""
  original_shape = array.shape
  array = pad_to_tile(array, how.tiled_axes)
  array = qarray.quantize(array, how)
  if not _QARRAY_KEEP_PADDED_SHAPE:
    array = dataclasses.replace(
        array,
        qvalue=array.qvalue[tuple(slice(0, dim) for dim in original_shape)],
    )
  return PaddedQArray(
      **dataclasses.asdict(array),
      tile_axes=how.tiled_axes,
      original_shape=original_shape,
  )


def dequantize(array: PaddedQArray) -> jax.Array:
  """Dequantizes an array. The reverse of |quantize|."""
  qarray.validate_qarray(array)
  original_shape = array.shape
  padded_qvalue = pad_to_tile(array.qvalue, dict(array.tile_axes))
  out = qarray.dequantize(dataclasses.replace(array, qvalue=padded_qvalue))
  if out.shape != array.original_shape and not _QARRAY_KEEP_PADDED_SHAPE:
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
  """Pad operands online based on dimension_numbers, then delegate.

  Infers tile axes from quantized operands to pad on contracting dimensions.
  Pads raw arrays when only one operand is quantized.

  Args:
    lhs: Left-hand side operand.
    rhs: Right-hand side operand.
    dimension_numbers: Dimension numbers for dot_general.
    precision: Optional precision.
    preferred_element_type: Optional element type.
    **kwargs: Additional arguments.

  Returns:
    Result of dot_general operation.
  """

  # Infer tile size by inspecting existing quantized operands.
  tile_size = None
  if isinstance(rhs, PaddedQArray):
    tile_size = next(iter(dict(rhs.tile_axes).values()))
  elif isinstance(lhs, PaddedQArray):
    tile_size = next(iter(dict(lhs.tile_axes).values()))

  get_how_to_quantize = functools.partial(
      core_dot_general.get_how_to_quantize,
      dimension_numbers=dimension_numbers,
      ndims=(len(lhs.shape), len(rhs.shape)),
  )
  how_lhs = get_how_to_quantize(
      for_lhs=True,
      qtype=None,
      tile_size=tile_size,
      calibration_method=None,
  )
  how_rhs = get_how_to_quantize(
      for_lhs=False,
      qtype=None,
      tile_size=tile_size,
      calibration_method=None,
  )

  lhs = _pad_operand(lhs, how_lhs.tiled_axes)
  rhs = _pad_operand(rhs, how_rhs.tiled_axes)

  if _QARRAY_USE_FAST_DOT_GENERAL:
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
  """Pad operands online based on einsum_str, then delegate."""

  # Infer tile size by inspecting existing quantized operands.
  tile_size = None
  if isinstance(rhs, PaddedQArray):
    tile_size = next(iter(dict(rhs.tile_axes).values()))
  elif isinstance(lhs, PaddedQArray):
    tile_size = next(iter(dict(lhs.tile_axes).values()))

  get_how_to_quantize = functools.partial(
      core_einsum.get_how_to_quantize,
      einsum_str=einsum_str,
      ndims=(len(lhs.shape), len(rhs.shape)),
  )
  how_lhs = get_how_to_quantize(
      for_lhs=True,
      qtype=None,
      tile_size=tile_size,
      calibration_method=None,
  )
  how_rhs = get_how_to_quantize(
      for_lhs=False,
      qtype=None,
      tile_size=tile_size,
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
