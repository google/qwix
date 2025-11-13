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
"""PTQ extension with unevenly subchannel quantization support.

Extends qwix.QArray with PaddedQArray for automatic padding to tile size
multiples before quantization. Provides wrappers for dot_general and PTQ
functions.
"""

from __future__ import annotations

import dataclasses
import functools
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

# Whether to keep shape of qvalue in padded form during "quantize"
QARRAY_KEEP_PADDED_SHAPE = False


# ---------------------------
# Padded QArray implementation
# ---------------------------


@flax.struct.dataclass
class PaddedQArray(qarray.QArray):
  """Quantized array with padding support.

  The qvalue stored can be either padded or unpadded, i.e.
  qvalue.shape == padded_shape or qvalue.shape == original_shape

  Attributes:
    padded_shape: The shape after padding to tile boundaries.
    original_shape: The original shape before padding.
  """

  padded_shape: tuple[int, ...] = flax.struct.field(
      pytree_node=False, default=()
  )
  original_shape: tuple[int, ...] = flax.struct.field(
      pytree_node=False, default=()
  )


MaybeQArray: TypeAlias = jax.Array | qarray.QArray | PaddedQArray


def pad_to_shape(array: jax.Array, target_shape: tuple[int, ...]) -> jax.Array:
  """Pads array to target shape."""
  if array.shape == target_shape:
    return array
  pad_width = [
      (0, target - current)
      for current, target in zip(array.shape, target_shape)
  ]
  return jnp.pad(array, pad_width, constant_values=0)


def get_padded_shape(
    original_shape: tuple[int, ...],
    tiled_axes: Mapping[int, int | float],
) -> tuple[int, ...]:
  """Computes the target padded shape given tiled axes.

  For each axis in tiled_axes, pads the dimension up to the next multiple of
  the tile size. If the tile size is a float, it follows the QWIX convention
  of interpreting it as 1 / tile_count, i.e., tile_size = round(dim * tile).

  Args:
    original_shape: The original shape of the array.
    tiled_axes: A mapping from axis index to the tile size or tile count.

  Returns:
    The shape after padding.
  """
  if not tiled_axes:
    return tuple(original_shape)
  out = list(original_shape)
  for axis, tile in tiled_axes.items():
    dim = original_shape[axis]
    tile_size = round(dim * tile) if isinstance(tile, float) else int(tile)
    if tile_size <= 0:
      continue
    remainder = dim % tile_size
    if remainder:
      out[axis] = dim + (tile_size - remainder)
  return tuple(out)


def quantize(array: jax.Array, how: HowToQuantize) -> PaddedQArray:
  """Quantizes an array using a dynamic range with padding support."""
  original_shape = array.shape
  array = pad_to_shape(array, get_padded_shape(array.shape, how.tiled_axes))
  padded_shape = array.shape
  array = qarray.quantize(array, how)
  if not QARRAY_KEEP_PADDED_SHAPE:
    array = dataclasses.replace(
        array,
        qvalue=array.qvalue[tuple(slice(0, dim) for dim in original_shape)],
    )
  return PaddedQArray(
      **dataclasses.asdict(array),
      padded_shape=padded_shape,
      original_shape=original_shape,
  )


def dequantize(array: PaddedQArray) -> jax.Array:
  """Dequantizes an array. The reverse of |quantize|."""
  qarray.validate_qarray(array)
  padded_qvalue = pad_to_shape(array.qvalue, array.padded_shape)
  out = qarray.dequantize(dataclasses.replace(array, qvalue=padded_qvalue))
  if out.shape != array.original_shape:
    out = out[tuple(slice(0, d) for d in array.original_shape)]
  return out


# ---------------------------
# dot_general and einsum wrappers
# ---------------------------


def _pad_operand_if_qarray(x):
  if isinstance(x, PaddedQArray):
    padded_q = pad_to_shape(x.qvalue, x.padded_shape)
    return dataclasses.replace(x, qvalue=padded_q)
  return x


def dot_general(
    lhs: MaybeQArray,
    rhs: MaybeQArray,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    **kwargs,
) -> jax.Array:
  """Pad operands and delegate to core dot_general.

  PaddedQArray operands are padded to their stored padded_shape. Regular arrays
  are padded along contraction dimensions to match PaddedQArray operands.

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

  # Pad PaddedQArray operands to their padded_shape
  lhs = _pad_operand_if_qarray(lhs)
  rhs = _pad_operand_if_qarray(rhs)

  # If only one operand is PaddedQArray, pad the other to match along
  # contraction dims
  (lhs_contract, rhs_contract), _ = dimension_numbers

  if not isinstance(rhs, PaddedQArray):
    target_shape = list(rhs.shape)
    for rhs_axis, lhs_axis in zip(rhs_contract, lhs_contract):
      target_shape[rhs_axis] = lhs.shape[lhs_axis]
    rhs = pad_to_shape(rhs, tuple(target_shape))

  if not isinstance(lhs, PaddedQArray):
    target_shape = list(lhs.shape)
    for lhs_axis, rhs_axis in zip(lhs_contract, rhs_contract):
      target_shape[lhs_axis] = rhs.shape[rhs_axis]
    lhs = pad_to_shape(lhs, tuple(target_shape))

  return core_dot_general.dot_general(
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
    preferred_element_type: jax.typing.DTypeLike | None = None,
    **kwargs,
) -> jax.Array:
  """Pad operands and delegate to core einsum.

  PaddedQArray operands are padded to their stored padded_shape. Regular arrays
  are padded along contraction dimensions to match PaddedQArray operands.

  Args:
    einsum_str: The einsum equation string.
    lhs: Left-hand side operand.
    rhs: Right-hand side operand.
    preferred_element_type: Optional element type.
    **kwargs: Additional arguments.

  Returns:
    Result of the einsum operation.
  """

  # Pad PaddedQArray operands to their padded_shape
  lhs = _pad_operand_if_qarray(lhs)
  rhs = _pad_operand_if_qarray(rhs)

  # If only one operand is PaddedQArray, pad the other to match along
  # contraction dims
  info = core_einsum.get_einsum_info(einsum_str, (lhs.ndim, rhs.ndim))
  if not isinstance(rhs, PaddedQArray):
    target_shape = list(rhs.shape)
    for axis, name in enumerate(info.rhs):
      if name in info.contractions:
        lhs_axis = info.lhs.index(name)
        target_shape[axis] = lhs.shape[lhs_axis]
    rhs = pad_to_shape(rhs, tuple(target_shape))

  if not isinstance(lhs, PaddedQArray):
    target_shape = list(lhs.shape)
    for axis, name in enumerate(info.lhs):
      if name in info.contractions:
        rhs_axis = info.rhs.index(name)
        target_shape[axis] = rhs.shape[rhs_axis]
    lhs = pad_to_shape(lhs, tuple(target_shape))

  return core_einsum.einsum(
      einsum_str,
      lhs,
      rhs,
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
