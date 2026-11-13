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

import jax
from jax import numpy as jnp
from jax.experimental import layout


def _validate_sparsity_args(
    shape: tuple[int, ...], *, n: int, m: int, axis: int
) -> int:
  """Validates `1:m` sparsity, non-empty rank >= 2 `shape`, and `axis`."""
  if not 0 < n < m <= 4:
    raise ValueError(f'Expected 0 < n < m <= 4; got {n=}, {m=}.')
  if n != 1:
    raise NotImplementedError(f'Only n == 1 is supported; got {n=}.')
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
  """Compresses `x` under a contiguous `1:m` boolean `mask` along `axis`.

  `mask` is expected to contain at most one `True` entry in every contiguous
  group of `m` elements along `axis` (if multiple entries are `True`, the
  lowest offset is selected; if all entries are `False`, offset `0` is selected
  with a zeroed value).

  Args:
    x: Non-empty dense input array of rank >= 2.
    mask: Boolean mask with the same shape as `x`.
    n: Number of kept elements per group (only `n == 1` is supported).
    m: Group size along `axis` (`2 <= m <= 4`).
    axis: Axis along which contiguous groups of `m` elements are compressed.

  Returns:
    A `(values, indices)` pair, each with shape matching `x` except dimension
    `axis` has size `x.shape[axis] // m`. `values` preserves `x.dtype` and
    `indices` has dtype `jnp.uint2` with group-local offsets in `[0, m)`.

  Raises:
    ValueError: If `0 < n < m <= 4` is violated, `x` has rank < 2 or a zero-size
      dimension, `axis` is out of range, `x.shape != mask.shape`, `mask` is not
      boolean, or `x.shape[axis]` is not divisible by `m`.
    NotImplementedError: If `n != 1`.
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
  # `jnp.argmax` returns the smallest index on ties (first `True`, or `0` if
  # all `False`), portable across all JAX backends.
  selected_indices = jnp.argmax(mask_grouped, axis=-1, keepdims=True)
  masked = jnp.where(mask_grouped, x_grouped, jnp.zeros_like(x_grouped))
  values = jnp.take_along_axis(masked, selected_indices, axis=-1).reshape(
      *leading, num_groups
  )
  indices = selected_indices.reshape(*leading, num_groups)
  return (
      jnp.moveaxis(values, -1, axis),
      _to_u2(jnp.moveaxis(indices, -1, axis)),
  )


def decompress(
    values: jax.Array,
    indices: jax.Array,
    *,
    n: int,
    m: int,
    axis: int,
) -> jax.Array:
  """Reconstructs a dense array from `1:m` compressed `(values, indices)`.

  Args:
    values: Non-empty compressed values of rank >= 2.
    indices: `jnp.uint2` group-local indices in `[0, m)` matching
      `values.shape`.
    n: Number of kept elements per group (only `n == 1` is supported).
    m: Expanded group size along `axis` (`2 <= m <= 4`).
    axis: Compressed axis to expand from `L` to `L * m`.

  Returns:
    Dense array of dtype `values.dtype` with `axis` expanded to
    `values.shape[axis] * m` and zeros placed at unselected positions.

  Raises:
    ValueError: If `0 < n < m <= 4` is violated, `values` has rank < 2 or a
      zero-size dimension, `axis` is out of range, `values.shape !=
      indices.shape`, or `indices` does not have dtype `uint2`.
    NotImplementedError: If `n != 1`.
  """
  axis = _validate_sparsity_args(values.shape, n=n, m=m, axis=axis)
  if values.shape != indices.shape:
    raise ValueError(
        f'values and indices shapes must match; got {values.shape} vs'
        f' {indices.shape}.'
    )
  if jnp.dtype(indices.dtype) != jnp.uint2:
    raise ValueError(f'indices must have dtype uint2; got {indices.dtype}.')
  indices_moved = jnp.moveaxis(_from_u2(indices), axis, -1)
  values_moved = jnp.moveaxis(values, axis, -1)
  leading = values_moved.shape[:-1]
  num_groups = values_moved.shape[-1]
  values_bcast = jnp.broadcast_to(
      values_moved[..., None], (*leading, num_groups, m)
  )
  iota = jnp.arange(m, dtype=jnp.uint8)
  out = jnp.where(
      indices_moved[..., None] == iota,
      values_bcast,
      jnp.zeros_like(values_bcast),
  )
  return jnp.moveaxis(out.reshape(*leading, num_groups * m), -1, axis)
