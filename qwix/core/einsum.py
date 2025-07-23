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
"""Quantized einsum with subchannel support."""

import dataclasses
from typing import Collection

import jax
from jax import numpy as jnp
from qwix.core import numerics
from qwix.core import qarray


@dataclasses.dataclass(slots=True)
class EinsumInfo:
  lhs: str
  rhs: str
  out: str
  contractions: Collection[str]


def get_einsum_info(einsum_str: str, ndims: tuple[int, int]) -> EinsumInfo:
  """Gets einsum info from an einsum string."""
  einsum_str = einsum_str.replace(' ', '')
  inputs, out = einsum_str.split('->')
  lhs, rhs = inputs.split(',')
  if '...' in lhs or '...' in rhs:
    ndim = ndims[0] - len(lhs) + 3 if '...' in lhs else ndims[1] - len(rhs) + 3
    assert ndim <= 10, f'{ndim=} {einsum_str=}'
    digits = ''.join(map(str, range(ndim)))
    assert not set(digits) & set(einsum_str), f'{digits=} {einsum_str=}'
    lhs = lhs.replace('...', digits)
    rhs = rhs.replace('...', digits)
    out = out.replace('...', digits)
  return EinsumInfo(lhs, rhs, out, set(lhs) & set(rhs) - set(out))


def get_how_to_quantize(
    *,
    einsum_str: str,
    ndims: tuple[int, int],
    for_lhs: bool,
    qtype: jax.typing.DTypeLike,
    tile_size: int | None,
    calibration_method: str,
    batch_axes: Collection[int],
) -> qarray.HowToQuantize:
  """Get how to quantize from an einsum string.

  By default, use channelwise for all non-contraction axes, and subchannel for
  contraction axes if a tile_size is given.

  Args:
    einsum_str: The einsum string.
    ndims: The number of dimensions of the lhs and rhs array. This is needed
      when ellipsis is in subscripts and we need to determine the number of
      dimensions represented by ellipsis.
    for_lhs: Whether to quantize lhs or rhs.
    qtype: The logical type for quantized value.
    tile_size: The tile size for subchannel quantization.
    calibration_method: The calibration method to use.
    batch_axes: Batch axes used for calibration.

  Returns:
    How to quantize the lhs or rhs.
  """
  info = get_einsum_info(einsum_str, ndims)
  subs = info.lhs if for_lhs else info.rhs
  channelwise_axes = []
  tiled_axes = {}
  for axis, name in enumerate(subs):
    if name not in info.contractions and name not in batch_axes:
      channelwise_axes.append(axis)
    elif tile_size:
      tiled_axes[axis] = tile_size

  return qarray.HowToQuantize(
      qtype=qtype,
      channelwise_axes=channelwise_axes,
      tiled_axes=tiled_axes,
      batch_axes=batch_axes,
      calibration_method=calibration_method,
  )


def _get_transpose(src: str, dst: str) -> list[int | None]:
  """Returns the transpose list for the given src and dst."""
  return [src.index(d) if d in src else None for d in dst]


def _assert_and_get_single_contraction(info: EinsumInfo) -> str:
  if len(info.contractions) != 1:
    raise ValueError('Only a single tiled axis is supported for now.')
  return next(iter(info.contractions))


def _apply_subchannel(info: EinsumInfo) -> EinsumInfo:
  """Apply subchannel to einsum info."""
  ca = _assert_and_get_single_contraction(info)
  # This is faster than putting ca at the beginning of the out string.
  # NOTE: This may change the numerics! (b/373012343)
  tiled_idx = next(i for i, a in enumerate(info.out) if a not in info.lhs)
  assert '*' not in info.lhs + info.rhs + info.out, f'{info=}'
  return dataclasses.replace(
      info,
      # Use "*" as tile_size axis, which becomes the new contraction axis.
      lhs=info.lhs.replace(ca, ca + '*'),
      rhs=info.rhs.replace(ca, ca + '*'),
      out=info.out[:tiled_idx] + ca + info.out[tiled_idx:],
  )


def _fast_einsum(
    einsum_str: str, lhs: qarray.MaybeQArray, rhs: qarray.MaybeQArray
):
  """Einsum in faster path by computing in quantized types first then dequantize."""
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

  if rhs_zero_point is not None:
    raise ValueError('Asymmetric quantization for rhs is not supported.')

  if lhs_scale is not None and rhs_scale is not None:
    if lhs_tile_size != rhs_tile_size:
      # Different tile sizes are not supported for now.
      raise ValueError(f'{lhs_tile_size=} != {rhs_tile_size=}')

  tile_size = lhs_tile_size or rhs_tile_size
  lhs_ndim = len(lhs_value.shape)
  rhs_ndim = len(rhs_value.shape)
  info = get_einsum_info(einsum_str, (lhs_ndim, rhs_ndim))

  tiled_sum_axis = None
  einsum_out = info.out
  if tile_size:
    contraction = _assert_and_get_single_contraction(info)
    # Prepare einsum_str, e.g., "gecf,efd->gecd" becomes "gecf*,ef*d->gecfd".
    tiled_info = _apply_subchannel(info)
    einsum_out = tiled_info.out
    tiled_sum_axis = tiled_info.out.index(contraction)
    einsum_str = tiled_info.lhs + ',' + tiled_info.rhs + '->' + tiled_info.out
    # Split lhs/rhs_value.
    lhs_ca = info.lhs.index(contraction)
    lhs_value = qarray.split_axis(lhs_value, {lhs_ca: tile_size})
    if lhs_zero_point is not None:
      lhs_zero_point = qarray.split_axis(lhs_zero_point, {lhs_ca: 1})
    rhs_ca = info.rhs.index(contraction)
    rhs_value = qarray.split_axis(rhs_value, {rhs_ca: tile_size})

  # Transpose lhs/rhs_scale.
  if lhs_scale is not None:
    transpose = _get_transpose(info.lhs, einsum_out)
    lhs_scale = qarray.transpose_array(lhs_scale, transpose)
  if rhs_scale is not None:
    transpose = _get_transpose(info.rhs, einsum_out)
    rhs_scale = qarray.transpose_array(rhs_scale, transpose)

  if all(x.dtype.name.startswith('int') for x in (lhs_value, rhs_value)):
    acc_type = jnp.int32
  elif lhs_scale is not None:
    acc_type = lhs_scale.dtype
  elif rhs_scale is not None:
    acc_type = rhs_scale.dtype
  else:
    acc_type = None  # let jnp.einsum decide.

  res = jnp.einsum(
      einsum_str, lhs_value, rhs_value, preferred_element_type=acc_type
  )
  if lhs_zero_point is not None:
    # TODO(zhuyunx): This value can be constant folded in SRQ scenarios.
    res -= jnp.einsum(
        einsum_str,
        jnp.broadcast_to(lhs_zero_point, lhs_value.shape),
        rhs_value,
        preferred_element_type=acc_type,
    )

  if lhs_scale is not None:
    res *= lhs_scale
  if rhs_scale is not None:
    res *= rhs_scale
  if tiled_sum_axis is not None:
    res = jnp.sum(res, axis=tiled_sum_axis)
  return res


def _slow_einsum(
    einsum_str: str, lhs: qarray.MaybeQArray, rhs: qarray.MaybeQArray
) -> jax.Array:
  """Einsum in slow path by dequantizing first then computing in floating-point types."""
  if isinstance(lhs, qarray.QArray):
    lhs_value = qarray.dequantize(lhs)
  else:
    lhs_value = lhs
  if isinstance(rhs, qarray.QArray):
    rhs_value = qarray.dequantize(rhs)
  else:
    rhs_value = rhs
  return jnp.einsum(einsum_str, lhs_value, rhs_value)


def einsum(
    einsum_str: str, lhs: qarray.MaybeQArray, rhs: qarray.MaybeQArray
) -> jax.Array:
  """Quantized einsum that takes jax.Array or QArray and returns floating-point jax.Array.

  Args:
    einsum_str: The einsum string.
    lhs: The left-hand side, either a jax.Array or QArray.
    rhs: The right-hand side, either a jax.Array or QArray.

  Returns:
    The result of the einsum, a floating-point jax.Array.
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
    return _fast_einsum(einsum_str, lhs, rhs)
  else:
    return _slow_einsum(einsum_str, lhs, rhs)
