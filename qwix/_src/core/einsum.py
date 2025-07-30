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
from qwix._src.core import dot_general
from qwix._src.core import qarray


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
    tile_size: int | float | None,
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


def einsum(
    *args, _qwix_dot_general=dot_general.dot_general, **kwargs
) -> jax.Array:
  """Quantized einsum that can take QArrays and returns floating-point jax.Array.

  Args:
    *args: Arguments to einsum.
    _qwix_dot_general: The dot_general function to use.
    **kwargs: Keyword arguments to einsum.

  Returns:
    The result of the einsum, a floating-point jax.Array.
  """
  # We want to use jnp.einsum with quantized dot_general to avoid duplicating
  # the implementation. However, jnp.einsum will check the inputs to be
  # jax Arrays. To work around this, we send the qvalue to jnp.einsum and
  # restore the actual QArray in a wrapper. preferred_element_type needs to be
  # set so that jnp.einsum won't convert the output to some qvalue types.
  args = list(args)
  qvalue_to_qarray = {}
  for i, arg in enumerate(args):
    if isinstance(arg, qarray.QArray):
      args[i] = arg.qvalue
      qvalue_to_qarray[id(arg.qvalue)] = arg
      kwargs['preferred_element_type'] = arg.scale.dtype

  def _dot_general(*args, **kwargs):
    args = [qvalue_to_qarray.pop(id(a), a) for a in args]
    return _qwix_dot_general(*args, **kwargs)

  # Disabling JIT is necessary so that args in _dot_general are not tracers.
  with jax.disable_jit():
    out = jnp.einsum(*args, _dot_general=_dot_general, **kwargs)
  assert not qvalue_to_qarray, 'All qvalues should be consumed.'
  return out
