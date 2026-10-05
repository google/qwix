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

"""Basic functionalities for introducing sparsity in neural networks."""

import dataclasses
from typing import Optional

import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True, kw_only=True)
class SparsityRule:
  """Sparsity rules that match and configure the sparsity behavior."""

  # N:M sparsity for weights and activations. If m > 0, sparsity is enabled.
  # 1:4/2:4 and 3:4 sparsity are supported.
  # TODO(shivaniagrawal): Raise value error if
  # weight_sparsity_n >= weight_sparsity_m.
  weight_sparsity_n: int = 0
  weight_sparsity_m: int = 0
  # Apply pruning using this index order. See
  # sparsity.jax.sparsity_core.get_sparsity_mask for details.
  weight_sparsity_order: str = 'R'
  weight_sparsity_block_size: int = 0
  weight_sparsity_offset: int = 0
  weight_sparsity_start_step: int = 0
  weight_sparsity_update_step: int = 1

  eval_mode: bool = False

  activation_sparsity_n: int = 0
  activation_sparsity_m: int = 0
  activation_sparsity_order: str = 'R'
  activation_sparsity_block_size: int = 0
  activation_sparsity_offset: int = 0
  compress_weights: bool = False
  sparse_axis: int = 0


def apply_sparsity(
    inputs: jax.Array,
    mask: jax.Array,
    is_channelwise: bool = False,
    pruned_value: Optional[jax.Array] = None,
) -> jax.Array:
  """Returns sparsified inputs based on input mask.

  Args:
    inputs: The input tensor.
    mask: The mask to be applied to the input tensor.
    is_channelwise: If true the mask will be treated as a channelwise mask and
      will be broadcast applied to the input tensor.
    pruned_value: If not None, it replaces the value of pruned elements (zeros)
      with the given pruned_value.

  Returns: The masked input tensor.
  """
  if is_channelwise:
    mask = mask * jnp.ones_like(inputs).astype(jnp.bool_)
  if pruned_value is not None:
    return jnp.multiply(inputs, ((~mask) * pruned_value + mask))
  else:
    return jnp.where(
        mask,
        inputs,
        jnp.zeros(inputs.shape, inputs.dtype),
    )


def get_sparsity_mask(
    inputs: jax.Array,
    n_sparsity: int = 0,
    m_sparsity: int = 0,
    order: str = 'R',
    block_size: int = 0,
    offset: int = 0,
) -> jax.Array:
  """Returns sparsified inputs for n:m structured pruning.

  Args:
    inputs: Input array for which N:M pruning mask is computed.
    n_sparsity: Maximum number of non-zero values in each block.
    m_sparsity: Number of values in each block.
    order: Apply pruning using this index order. Supported values are `C`, `R`.
      `C` and `R` indicate column-wise and row-wise masking, respectively.
      Default is `R` indicating to applying N:M sparsity across rows of the
      input matrix. Default is `C` indicating to applying N:M sparsity across
      columns of the input matrix. The choice may intersect with hardware
      capabilities. For a weight tensor `C` corresponds to the reduction
      dimension, and `R' for activations.
    block_size: Number of values in each weight block.
    offset: Indicates the offset between the group of M elements on which
      N:M sparsity is applied. The default is `0` (narrowly-separated),
        indicating that `M` elements are selected from adjacent values in the
        input matrix. Generally, because of the XLA layout (lanes 128/sublanes
        8), another value for offset would be 128 (widely-separated). If offset
        > 0, we only support scenarios where the input array size is equal to
        (offset * m). Offset != 128 may not be best optimized for the memory
        layout.

  Returns:
    A mask that indicates the pruning locations (`0`: no pruning, `1`: pruned).
  """
  assert (
      n_sparsity <= m_sparsity
  ), f'N must be lower than M for N:M ({n_sparsity}:{m_sparsity}) sparsity.'
  if order not in ['C', 'R']:
    raise ValueError(f'Index order {order} not supported.')
  if offset < 0:
    raise ValueError(f'Offset value must be positive. You provided {offset}.')

  length = jnp.size(inputs)
  if length % m_sparsity != 0:
    raise ValueError(
        f'inputs size must be divisible by m, provided {length} and'
        f' {m_sparsity}'
    )
  if order not in ['C', 'R']:
    raise ValueError(f'Index order {order} not supported.')

  if block_size > 1:
    blocks = int(length / block_size)
    original_shape = inputs.shape
    if order == 'R':
      inputs_block = inputs.reshape(blocks, block_size, order='C')
    else:
      inputs_trans = jnp.einsum('...ij->...ji', inputs)
      original_shape = inputs_trans.shape
      inputs_block = inputs_trans.reshape(blocks, block_size, order='C')

    def block_score(inputs: jax.Array):
      return jnp.sum(jnp.abs(inputs), axis=-1)

    inputs_block_temp = jnp.apply_along_axis(
        block_score, axis=-1, arr=inputs_block
    )
    mask_shape = tuple((
        original_shape[i]
        if i != len(original_shape) - 1
        else int(original_shape[i] / block_size)
        for i in range(len(original_shape))
    ))
    if order == 'R':
      new_inputs = inputs_block_temp.reshape(mask_shape, order='C')
    else:
      new_inputs = jnp.einsum(
          '...ij->...ji', inputs_block_temp.reshape(mask_shape, order='C')
      )
    inputs = new_inputs

  length = jnp.size(inputs)
  if offset > 0 and length % (offset * m_sparsity) != 0:
    raise ValueError(
        'When offset > 0, we only support an array size (length) equal to '
        f'(offset * m_sparsity). Provided offset = {offset}, '
        f'm_sparsity = {m_sparsity}, length = {length}.'
    )

  inputs = jnp.abs(inputs)
  original_shape = inputs.shape

  if order == 'C':
    inputs = jnp.einsum('...ij->...ji', inputs)
    original_shape = inputs.shape

  prac_offset = 1 if offset == 0 else offset
  if original_shape[-1] % (m_sparsity * prac_offset) == 0:
    group = original_shape[-1] // m_sparsity
    # TODO(shivaniagrawal): we can always split in 3D with offset=1 too and
    # do top-K in -2 dimension.
    if offset > 1:
      new_shape = (*original_shape[:-1], group // offset, m_sparsity, offset)
      inputs = inputs.reshape(new_shape)
      inputs = jnp.einsum('...ij->...ji', inputs)

    new_shape = (*original_shape[:-1], group, m_sparsity)
    inputs_temp = inputs.reshape(new_shape)

  else:
    group = int(length / m_sparsity)
    if offset > 0:
      inputs = inputs.reshape((group // offset, m_sparsity, offset))
      inputs = jnp.einsum('...ij->...ji', inputs)

    inputs_temp = inputs.reshape(group, m_sparsity, order='C')

  _, top_k_indices = jax.lax.top_k(inputs_temp, k=n_sparsity)
  mask = jnp.any(
      jax.nn.one_hot(top_k_indices, m_sparsity, dtype=jnp.bool_), axis=-2
  )

  if offset > 0:
    # NOTE: without meeting this condition, we had flattened the whole matrix
    # and mask as well.
    if original_shape[-1] % (m_sparsity * offset) == 0:
      # group = original_shape[-1] // m_sparsity in this case
      mask = mask.reshape(
          (*original_shape[:-1], group // offset, offset, m_sparsity)
      )
    else:
      # group = length // m_sparsity in this case
      mask = mask.reshape((group // offset, offset, m_sparsity))
    mask = jnp.einsum('...ij->...ji', mask)

  if order == 'R':
    result_mask = mask.reshape(original_shape, order='C')
  else:
    result_mask = jnp.einsum(
        '...ij->...ji', mask.reshape(original_shape, order='C')
    )

  if block_size > 0:
    if order == 'R':
      expanded_mask = jnp.repeat(result_mask, block_size, axis=-1)
    else:
      expanded_mask = jnp.repeat(result_mask, block_size, axis=-2)
    return expanded_mask
  else:
    return result_mask


def get_sparsity_mask_unstructured(
    inputs: jax.Array,
    mask: jax.Array | None,
    prune_rate: jax.Array | float,
) -> jax.Array:
  """Computes a sparisty mask to prune the required percentage of weights.

  The mask is calculated by thresholding the absolute values of inputs. The
  threshold is the lowest value greater than prune_rate percent of weights, i.e.
  the corresponding percentile.

  The newly pruned weights form a superset of the currently pruned weights if
  the current mask is provided.

  Args:
      inputs: Input tensor.
      mask: Current mask.
      prune_rate: Percentage of weights to prune, value between 0 and 100.

  Returns:
      Sparsity mask.
  """
  if mask is not None:
    inputs = apply_sparsity(inputs, mask)
  inputs_abs = jnp.abs(inputs)
  threshold = jnp.percentile(inputs_abs, prune_rate)
  return jnp.greater(inputs_abs, threshold)


def prune_inputs_n_m(
    inputs: jax.Array,
    *,
    n: int,
    m: int,
    order: str = 'R',
    offset: int = 0,
) -> jax.Array:
  """Returns pruned array with N:M (structured) pruning.

  N:M pruning makes at most N values non-zero in each block of M consecutive
  values.

  Args:
    inputs: Input array for which N:M pruning mask is computed.
    n: Maximum number of non-zero values in each block.
    m: Number of values in each block.
    order: Apply pruning using this index order. Supported values are `C`, `R`.
      `C` and `R` indicate column-wise and row-wise masking, respectively.
      Default is `R` indicating to applying N:M sparsity across rows of the
      input matrix. The choice may intersect with hardware capabilities. For a
      weight tensor `C` corresponds to the reduction dimension, and `R' for
      activations.
    offset: Indicates the offset between the group of M elements on which
      N:M sparsity is applied. The default is `0` (narrowly-separated),
        indicating that `M` elements are selected from adjacent values in the
        input matrix. Generally, because of the XLA layout (lanes 128/sublanes
        8), another value for offset would be 128 (widely-separated). If offset
        > 0, we only support scenarios where the input array size is equal to
        (offset * m). Offset != 128 may not be best optimized for the memory
        layout.

  Returns:
    An array with the same shape as inputs pruned with N:M strategy.
  """
  mask = get_sparsity_mask(inputs, n, m, order=order, offset=offset)
  return jnp.where(mask, inputs, jnp.zeros(inputs.shape, inputs.dtype))


def pack_u2_indices(indices: jax.Array, pack_axis: int = 0) -> jax.Array:
  """Packs 4 consecutive 2-bit index values [0..3] along pack_axis into uint8."""
  axis = pack_axis % indices.ndim
  if indices.shape[axis] % 4 != 0:
    raise ValueError(
        f'Axis {axis} size ({indices.shape[axis]}) must be divisible by 4 for'
        ' u2 index packing.'
    )
  grouped_shape = (
      indices.shape[:axis]
      + (indices.shape[axis] // 4, 4)
      + indices.shape[axis + 1 :]
  )
  grouped = indices.reshape(grouped_shape).astype(jnp.uint8)
  i0 = jnp.take(grouped, 0, axis=axis + 1) & 0x3
  i1 = (jnp.take(grouped, 1, axis=axis + 1) & 0x3) << 2
  i2 = (jnp.take(grouped, 2, axis=axis + 1) & 0x3) << 4
  i3 = (jnp.take(grouped, 3, axis=axis + 1) & 0x3) << 6
  return (i0 | i1 | i2 | i3).astype(jnp.uint8)


def unpack_u2_indices(
    packed_indices: jax.Array, pack_axis: int = 0
) -> jax.Array:
  """Unpacks uint8 elements along pack_axis into 4 consecutive uint8 values [0..3]."""
  axis = pack_axis % packed_indices.ndim
  packed_u8 = packed_indices.astype(jnp.uint8)
  i0 = packed_u8 & 0x3
  i1 = (packed_u8 >> 2) & 0x3
  i2 = (packed_u8 >> 4) & 0x3
  i3 = (packed_u8 >> 6) & 0x3
  stacked = jnp.stack([i0, i1, i2, i3], axis=axis + 1)
  unpacked_shape = (
      packed_indices.shape[:axis]
      + (packed_indices.shape[axis] * 4,)
      + packed_indices.shape[axis + 1 :]
  )
  return stacked.reshape(unpacked_shape).astype(jnp.uint8)


def compress_1_4(
    array: jax.Array,
    *,
    axis: int = 0,
    order: str = 'C',
    pack_indices: bool = False,
) -> tuple[jax.Array, jax.Array]:
  """Compresses an array along axis using 1:4 structured sparsity.

  Args:
    array: The input tensor whose size along `axis` must be divisible by 4.
    axis: The dimension along which to apply 1:4 compression.
    order: Grouping order ('C' or 'R').
    pack_indices: If True, packs 4 consecutive 2-bit indices along `axis` into a
      single uint8 (requiring `array.shape[axis]` to be divisible by 16).

  Returns:
    A tuple of `(compressed_values, indices)` where `compressed_values` has
    `shape[axis] == array.shape[axis] // 4` and `indices` has dtype `uint8`
    with `shape[axis] == array.shape[axis] // 4` (or `array.shape[axis] // 16`
    when `pack_indices=True`).
  """
  if order not in ('C', 'R'):
    raise ValueError(f'Index order {order} not supported.')
  axis = axis % array.ndim
  k = array.shape[axis]
  if k % 4 != 0:
    raise ValueError(
        f'Axis {axis} size ({k}) must be divisible by 4 for 1:4 compression.'
    )
  grouped_shape = array.shape[:axis] + (k // 4, 4) + array.shape[axis + 1 :]
  grouped = array.reshape(grouped_shape)
  indices = jnp.argmax(
      jnp.abs(grouped.astype(jnp.float32)), axis=axis + 1
  ).astype(jnp.uint8)
  compressed_values = jnp.take_along_axis(
      grouped, jnp.expand_dims(indices, axis=axis + 1), axis=axis + 1
  ).squeeze(axis=axis + 1)
  if pack_indices:
    indices = pack_u2_indices(indices, pack_axis=axis)
  return compressed_values, indices


def decompress_1_4(
    values: jax.Array,
    indices: jax.Array,
    *,
    axis: int = 0,
) -> jax.Array:
  """Reconstructs the dense tensor from 1:4 compressed values and indices.

  Args:
    values: Compressed values tensor of shape `[..., K // 4, ...]`.
    indices: Index tensor of shape `[..., K // 4, ...]` (unpacked uint8 in
      `[0..3]`) or `[..., K // 16, ...]` (packed u2 in uint8).
    axis: The compressed dimension.

  Returns:
    The decompressed tensor of shape `[..., K, ...]` with dtype `values.dtype`.
  """
  axis = axis % values.ndim
  if indices.ndim != values.ndim:
    raise ValueError(
        f'indices.ndim ({indices.ndim}) must match values.ndim ({values.ndim}).'
    )
  if indices.shape[axis] * 4 == values.shape[axis]:
    indices = unpack_u2_indices(indices, pack_axis=axis)
  if indices.shape != values.shape:
    raise ValueError(
        f'Unpacked indices shape {indices.shape} must match values shape'
        f' {values.shape}.'
    )
  slots_shape = (1,) * (axis + 1) + (4,) + (1,) * (values.ndim - axis - 1)
  slots = jnp.arange(4, dtype=indices.dtype).reshape(slots_shape)
  mask = jnp.expand_dims(indices, axis=axis + 1) == slots
  grouped = jnp.where(
      mask,
      jnp.expand_dims(values, axis=axis + 1),
      jnp.zeros(
          values.shape[:axis]
          + (values.shape[axis], 4)
          + values.shape[axis + 1 :],
          dtype=values.dtype,
      ),
  )
  dense_shape = (
      values.shape[:axis] + (values.shape[axis] * 4,) + values.shape[axis + 1 :]
  )
  return grouped.reshape(dense_shape)
