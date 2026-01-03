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
# pylint: disable=line-too-long

import dataclasses
from typing import Any, Collection

import jax
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
    tile_size: int | float | None,
    **kwargs: Any,
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
    tile_size: The tile size for subchannel quantization.
    **kwargs: Additional keyword arguments to HowToQuantize.

  Returns:
    How to quantize the lhs or rhs.
  """
  info = get_einsum_info(einsum_str, ndims)
  subs = info.lhs if for_lhs else info.rhs
  channelwise_axes = []
  tiled_axes = {}
  for axis, name in enumerate(subs):
    if name not in info.contractions:
      channelwise_axes.append(axis)
    elif tile_size and not tiled_axes:  # Only tile the first contraction axis.
      tiled_axes[axis] = tile_size

  return qarray.HowToQuantize(
      channelwise_axes=channelwise_axes,
      tiled_axes=tiled_axes,
      **kwargs,
  )


def einsum(
    *args,
    _qwix_dot_general=dot_general.dot_general,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    **kwargs,
) -> jax.Array:
  """Computes Einstein summation convention with support for ``QArray`` inputs.

  This function serves as a drop-in replacement for
  `jax.numpy.einsum
  <https://docs.jax.dev/en/latest/_autosummary/jax.numpy.einsum.html>`_.

  Args:
    *args: Arguments to einsum.
    _qwix_dot_general: The dot_general function to use.
    preferred_element_type: The preferred element type for jax.lax.dot_general.
    **kwargs: Keyword arguments to einsum.

  Returns:
    The result of the einsum, a floating-point jax.Array.
  """
  # preferred_element_type has to be set for jnp.einsum so that it won't infer
  # the type from qvalue x qvalue.
  _, preferred_element_type = qarray.get_accumulator_and_result_type(
      *[a for a in args if isinstance(a, qarray.MaybeQArray)],
      preferred_element_type=preferred_element_type,
  )

  # Separate string and operands
  # The `args` tuple is expected to be (einsum_str, lhs, rhs)
  if len(args) != 3:
    raise ValueError(
        f'Expected 3 arguments (einsum_str, lhs, rhs), but got {len(args)}'
    )
  einsum_str, lhs, rhs = args
  info = get_einsum_info(einsum_str, (lhs.ndim, rhs.ndim))

  batch_chars = sorted(list(set(info.lhs) & set(info.rhs) & set(info.out)))
  contract_chars = sorted(list(set(info.lhs) & set(info.rhs) - set(info.out)))

  # Construct dimension_numbers for dot_general
  lhs_map = {c: i for i, c in enumerate(info.lhs)}
  rhs_map = {c: i for i, c in enumerate(info.rhs)}
  lhs_contract = [lhs_map[c] for c in contract_chars]
  rhs_contract = [rhs_map[c] for c in contract_chars]
  lhs_batch = [lhs_map[c] for c in batch_chars]
  rhs_batch = [rhs_map[c] for c in batch_chars]
  dimension_numbers = (
      (tuple(lhs_contract), tuple(rhs_contract)),
      (tuple(lhs_batch), tuple(rhs_batch)),
  )

  # Call dot_general
  output = _qwix_dot_general(
      lhs,
      rhs,
      dimension_numbers,
      preferred_element_type=preferred_element_type,
      **kwargs,
  )

  # Transpose output
  lhs_free_chars = [
      c for c in info.lhs if c in info.out and c not in batch_chars
  ]
  rhs_free_chars = [
      c for c in info.rhs if c in info.out and c not in batch_chars
  ]
  current_out_chars = batch_chars + lhs_free_chars + rhs_free_chars
  current_pos_map = {c: i for i, c in enumerate(current_out_chars)}
  perm = [current_pos_map[c] for c in info.out]
  output = output.transpose(perm)

  return output
