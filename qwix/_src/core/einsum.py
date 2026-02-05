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

from typing import Any, Callable

import jax
from jax import numpy as jnp
import opt_einsum
from qwix._src.core import dot_general
from qwix._src.core import einsum_info
from qwix._src.core import qarray


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
  info = einsum_info.EinsumInfo.parse(einsum_str, ndims=ndims)
  operand_subs = info.lhs if for_lhs else info.rhs

  channelwise_axes = []
  tiled_axes = {}
  for axis, char in enumerate(operand_subs):
    if char not in info.contract_chars:
      channelwise_axes.append(axis)
    elif tile_size and not tiled_axes:  # Only tile the first contraction axis.
      tiled_axes[axis] = tile_size

  return qarray.HowToQuantize(
      channelwise_axes=channelwise_axes,
      tiled_axes=tiled_axes,
      **kwargs,
  )


def _perform_binary_einsum(
    lhs: qarray.MaybeQArray,
    rhs: qarray.MaybeQArray,
    einsum_str: str,
    dot_general_func: Callable[..., jax.Array],
    preferred_element_type: jax.typing.DTypeLike | None,
    **kwargs: Any,
) -> jax.Array:
  """Performs a binary einsum using the given dot_general function."""
  info = einsum_info.EinsumInfo.parse(einsum_str)
  output = dot_general_func(
      lhs,
      rhs,
      info.dimension_numbers,
      preferred_element_type=preferred_element_type,
      **kwargs,
  )
  if info.output_perm is not None:
    output = output.transpose(info.output_perm)
  return output


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

  # Prepare inputs using opt_einsum and broadcast_operands.
  # args: ("ij,jk->ik", qarray, qarray)
  # input_subs: "ij,jk",  output_subs: "ik", operands: [qarray, qarray]
  input_subs, output_subs, operands = opt_einsum.parser.parse_einsum_input(args)
  input_subs_list = input_subs.split(',')
  assert len(input_subs_list) == len(operands)
  operands = einsum_info.broadcast_operands(operands, input_subs_list)

  # Execution using opt_einsum path
  _, contractions = opt_einsum.contract_path(
      f'{input_subs}->{output_subs}',
      *operands,
      einsum_call=True,  # This is necessary for opt_einsum to return the contraction list.
  )
  for contraction in contractions:  # pytype: disable=attribute-error
    # operand_indices: (0, 1), einsum_str: "ij,jk->ik"
    operand_indices, _, einsum_str = contraction[:3]
    if len(operand_indices) == 1:
      # Fallback to dequantization for unary ops.
      op0 = operands.pop(operand_indices[0])
      op0 = qarray.dequantize(op0)
      res = jnp.einsum(einsum_str, op0)
      operands.append(res)
    elif len(operand_indices) == 2:
      idx0, idx1 = operand_indices
      op0 = operands[idx0]
      op1 = operands[idx1]
      for idx in sorted(operand_indices, reverse=True):
        # Ensure we pop in correct order to avoid index shift issues
        operands.pop(idx)
      res = _perform_binary_einsum(
          op0,
          op1,
          einsum_str,
          _qwix_dot_general,
          preferred_element_type=preferred_element_type,
          **kwargs,
      )
      operands.append(res)
    else:
      raise NotImplementedError(
          'A single opt_einsum contract path should contain at most 2'
          f' operands. Got {len(operand_indices)} operands: {operand_indices}'
      )

  assert isinstance(operands[0], jax.Array)
  return operands[0]
