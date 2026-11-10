# Copyright 2026 Google LLC
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
"""N:M structured sparse array utilities."""

import dataclasses

from flax import nnx
import flax.struct
import jax
from jax import numpy as jnp
from jax.experimental import layout
from qwix._src.core import qarray
from qwix._src.core import sparsity as sparsity_lib


def _validate_sparsity_args(
    shape: tuple[int, ...], *, n: int, m: int, axis: int
) -> int:
  """Validates `n:m`, non-empty rank >= 2 `shape`, and `axis`."""
  if not 0 < n < m <= 4:
    raise ValueError(f'Expected 0 < n < m <= 4; got {n=}, {m=}.')
  ndim = len(shape)
  if ndim < 2:
    raise ValueError(f'Expected rank >= 2 for N:M sparsity; got {ndim=}.')
  if 0 in shape:
    raise ValueError(
        f'Expected non-empty dimensions for N:M sparsity; got {shape=}.'
    )
  if not -ndim <= axis < ndim:
    raise ValueError(f'Axis {axis} is out of range for rank {ndim}.')
  return axis % ndim


def _tpu_u2_major_to_minor(x: jax.Array) -> tuple[int, ...] | None:
  """Returns TPU's default `uint2` `major_to_minor` order, or `None` off TPU."""
  devices = jax.devices()
  if not devices or devices[0].platform != 'tpu':
    return None
  sharding = jax.typeof(x).sharding
  if not sharding.mesh.empty and sharding.mesh.are_all_axes_explicit:
    shard_shape = sharding.shard_shape(x.shape)
  else:
    shard_shape = x.shape
  pjrt_layout = devices[0].client.get_default_layout(
      jnp.dtype(jnp.uint2), shard_shape, devices[0]
  )
  return layout.Layout.from_pjrt_layout(pjrt_layout).major_to_minor


def _tpu_to_u2(x: jax.Array) -> jax.Array:
  """Aligns `x` to TPU's `uint2` dimension order before casting to `uint2`."""
  # Pre-align in `x.dtype` to avoid an XLA:TPU 3D+ `uint2` relayout failure.
  u2_order = _tpu_u2_major_to_minor(x)
  if u2_order is not None:
    x = layout.with_layout_constraint(x, layout.Layout(u2_order))
  return x.astype(jnp.uint2)


def _tpu_from_u2(x: jax.Array) -> jax.Array:
  """Pins `uint2` `x` to HBM before casting to `uint8`."""
  # Keep `x` in HBM to avoid an XLA:TPU VMEM `uint2` unpack compiler failure.
  u2_order = _tpu_u2_major_to_minor(x)
  if u2_order is not None:
    x = jax.ffi.ffi_call(
        'xla_color_next_input_hbm',
        x,
        vmap_method='broadcast_all',
        input_layouts=(u2_order,),
        output_layouts=u2_order,
    )(x)
  return x.astype(jnp.uint8)


# `platform_dependent` traces all branches on every host and lowers by array
# placement; `_tpu_u2_major_to_minor` skips TPU PJRT queries on non-TPU hosts.
@jax.jit
def _to_u2(x: jax.Array) -> jax.Array:
  """Casts `x` to `jnp.uint2`, aligning layout on TPU."""
  return jax.lax.platform_dependent(
      x, tpu=_tpu_to_u2, default=lambda y: y.astype(jnp.uint2)
  )


@jax.jit
def _from_u2(indices: jax.Array) -> jax.Array:
  """Casts `jnp.uint2` `indices` to `jnp.uint8`, reading from HBM on TPU."""
  return jax.lax.platform_dependent(
      indices, tpu=_tpu_from_u2, default=lambda y: y.astype(jnp.uint8)
  )


def compress(
    x: jax.Array,
    mask: jax.Array,
    *,
    n: int,
    m: int,
    axis: int,
) -> tuple[jax.Array, jax.Array]:
  """Compresses `x` under a contiguous `n:m` boolean `mask` along `axis`.

  `mask` is expected to contain at most `n` `True` entries in every contiguous
  group of `m` elements along `axis` (if more than `n` entries are `True`, the
  lowest `n` offsets are selected). When a group has fewer than `n` `True`
  entries, lower group offsets are padded in ascending index order with zeroed
  values.

  Args:
    x: Non-empty dense input array of rank >= 2.
    mask: Boolean mask with the same shape as `x`.
    n: Number of kept elements per group (`0 < n < m <= 4`).
    m: Group size along `axis` (`0 < n < m <= 4`).
    axis: Axis along which contiguous groups of `m` elements are compressed.

  Returns:
    A `(values, indices)` pair, each with shape matching `x` except dimension
    `axis` has size `x.shape[axis] * n // m`. `values` preserves `x.dtype` and
    `indices` has dtype `jnp.uint2` with ascending offsets in `[0, m)` per
    group.

  Raises:
    ValueError: If `0 < n < m <= 4` is violated, `x` has rank < 2 or a zero-size
      dimension, `axis` is out of range, `x.shape != mask.shape`, `mask` is not
      boolean, or `x.shape[axis]` is not divisible by `m`.
  """
  axis = _validate_sparsity_args(x.shape, n=n, m=m, axis=axis)
  if x.shape != mask.shape:
    raise ValueError(
        f'x and mask shapes must match; got {x.shape} vs {mask.shape}.'
    )
  if jnp.dtype(mask.dtype) != jnp.bool_:
    raise ValueError(f'mask must have dtype bool; got {mask.dtype}.')
  if x.shape[axis] % m != 0:
    raise ValueError(
        f'Dimension {x.shape[axis]} along axis {axis} of shape {x.shape} is'
        f' not divisible by {m=}.'
    )
  x_moved = jnp.moveaxis(x, axis, -1)
  mask_moved = jnp.moveaxis(mask, axis, -1)
  leading = x_moved.shape[:-1]
  num_groups = x_moved.shape[-1] // m
  x_grouped = x_moved.reshape(*leading, num_groups, m)
  mask_grouped = mask_moved.reshape(*leading, num_groups, m)
  # Break ties toward lower offsets so `< n` True entries pad lower slots in
  # ascending order after `jnp.sort`, portable across all JAX backends.
  scores = mask_grouped.astype(jnp.int32) * m - jnp.arange(m, dtype=jnp.int32)
  _, selected_indices = jax.lax.top_k(scores, k=n)
  selected_indices = jnp.sort(selected_indices, axis=-1)
  masked = jnp.where(mask_grouped, x_grouped, jnp.zeros_like(x_grouped))
  values = jnp.take_along_axis(masked, selected_indices, axis=-1).reshape(
      *leading, num_groups * n
  )
  indices = selected_indices.reshape(*leading, num_groups * n)
  return (
      jnp.moveaxis(values, -1, axis),
      _to_u2(jnp.moveaxis(indices, -1, axis)),
  )


def _validate_compressed_buffers(
    values_shape: tuple[int, ...],
    indices: jax.Array | None = None,
    *,
    n: int,
    m: int,
    axis: int,
) -> int:
  """Validates compressed `(values, indices)` shapes, dtype, and `axis`."""
  axis = _validate_sparsity_args(values_shape, n=n, m=m, axis=axis)
  if values_shape[axis] % n != 0:
    raise ValueError(
        f'Compressed dimension {values_shape[axis]} along axis {axis} of'
        f' shape {values_shape} is not divisible by {n=}.'
    )
  if indices is not None:
    if values_shape != indices.shape:
      raise ValueError(
          f'values and indices shapes must match; got {values_shape} vs'
          f' {indices.shape}.'
      )
    if jnp.dtype(indices.dtype) != jnp.uint2:
      raise ValueError(f'indices must have dtype uint2; got {indices.dtype}.')
  return axis


@flax.struct.dataclass
class SparseArray:
  """An N:M structured sparse array with compressed values and 2-bit indices.

  Attributes:
    values: Compressed non-zero values (`jax.Array` or symmetric `QArray`) with
      `values.shape[axis] == shape[axis] * sparsity.n // sparsity.m`.
    indices: `jnp.uint2` group-local indices in `[0, sparsity.m)` matching the
      compressed value leaf shape.
    sparsity: Static `StructuredSparsity` metadata with normalized negative
      `dimension`.
  """

  values: jax.Array | qarray.QArray
  indices: jax.Array
  sparsity: sparsity_lib.StructuredSparsity = flax.struct.field(
      pytree_node=False
  )

  def __post_init__(self):
    if self.sparsity.dimension is None:
      raise ValueError('SparseArray requires sparsity.dimension to be set.')
    if self.sparsity.dimension >= 0:
      val_shape = self.values.shape
      ax = _validate_compressed_buffers(
          val_shape,
          n=self.sparsity.n,
          m=self.sparsity.m,
          axis=self.sparsity.dimension,
      )
      object.__setattr__(
          self,
          'sparsity',
          dataclasses.replace(self.sparsity, dimension=ax - len(val_shape)),
      )

  @property
  def axis(self) -> int:
    """Returns the normalized negative sparse axis."""
    assert self.sparsity.dimension is not None
    return self.sparsity.dimension

  @property
  def shape(self) -> tuple[int, ...]:
    """Returns the logical dense shape."""
    val_shape = self.values.shape
    ax = _validate_compressed_buffers(
        val_shape, n=self.sparsity.n, m=self.sparsity.m, axis=self.axis
    )
    dense_dim = (val_shape[ax] * self.sparsity.m) // self.sparsity.n
    return val_shape[:ax] + (dense_dim,) + val_shape[ax + 1 :]

  @property
  def ndim(self) -> int:
    """Returns the rank of the logical dense array."""
    return self.values.ndim

  @property
  def dtype(self) -> jnp.dtype:
    """Returns the logical dequantized dtype."""
    return self.values.dtype

  def astype(self, dtype: jax.typing.DTypeLike) -> 'SparseArray':
    """Casts the logical dtype of `values`."""
    return dataclasses.replace(self, values=self.values.astype(dtype))


nnx.register_data_type(SparseArray)


def validate_sparse_array(array: SparseArray) -> None:
  """Validates shape, dtype, and scale invariants of a `SparseArray`."""
  if isinstance(array.values, qarray.QArray):
    raw_val = array.values.qvalue
    if not isinstance(array.values.scale, jax.Array):
      return
  else:
    raw_val = array.values
  if not isinstance(raw_val, jax.Array) or not isinstance(
      array.indices, jax.Array
  ):
    return
  axis = _validate_compressed_buffers(
      raw_val.shape,
      array.indices,
      n=array.sparsity.n,
      m=array.sparsity.m,
      axis=array.axis,
  )
  if isinstance(array.values, qarray.QArray):
    if array.values.zero_point is not None:
      raise ValueError(
          'Asymmetric quantization (zero_point) is not supported in'
          ' SparseArray.'
      )
    qarray.validate_qarray(array.values)
    if array.values.scale.shape[axis] != 1:
      raise ValueError(
          'Subchannel scale along sparse axis is not supported; got'
          f' scale.shape={array.values.scale.shape} for axis {array.axis}.'
      )
    if qarray.get_tiled_axes(array.values):
      raise ValueError(
          'Only per-channel or per-tensor scales are supported in'
          f' SparseArray; got scale.shape={array.values.scale.shape} vs'
          f' qvalue.shape={raw_val.shape}.'
      )


def decompress(
    values: jax.Array,
    indices: jax.Array,
    *,
    n: int,
    m: int,
    axis: int,
) -> jax.Array:
  """Reconstructs a dense array from `n:m` compressed `(values, indices)`.

  `indices` must contain distinct offsets in `[0, m)` within each group of `n`
  elements along `axis`.

  Args:
    values: Non-empty compressed values of rank >= 2.
    indices: `jnp.uint2` group-local indices in `[0, m)` matching
      `values.shape`.
    n: Number of kept elements per group (`0 < n < m <= 4`).
    m: Expanded group size along `axis` (`0 < n < m <= 4`).
    axis: Compressed axis to expand from `L` to `L * m // n`.

  Returns:
    Dense array of dtype `values.dtype` with `axis` expanded to
    `values.shape[axis] * m // n` and zeros placed at unselected positions.

  Raises:
    ValueError: If `0 < n < m <= 4` is violated, `values` has rank < 2 or a
      zero-size dimension, `axis` is out of range, `values.shape !=
      indices.shape`, `indices` does not have dtype `uint2`, or
      `values.shape[axis]` is not divisible by `n`.
  """
  axis = _validate_compressed_buffers(
      values.shape, indices, n=n, m=m, axis=axis
  )
  indices_moved = jnp.moveaxis(_from_u2(indices), axis, -1)
  values_moved = jnp.moveaxis(values, axis, -1)
  leading = values_moved.shape[:-1]
  num_groups = values_moved.shape[-1] // n
  values_grouped = values_moved.reshape(*leading, num_groups, n)
  indices_grouped = indices_moved.reshape(*leading, num_groups, n)
  out = jnp.zeros((*leading, num_groups, m), dtype=values.dtype)
  iota = jnp.arange(m, dtype=jnp.uint8)
  for i in range(n):
    out = jnp.where(
        indices_grouped[..., i : i + 1] == iota,
        values_grouped[..., i : i + 1],
        out,
    )
  return jnp.moveaxis(out.reshape(*leading, num_groups * m), -1, axis)
