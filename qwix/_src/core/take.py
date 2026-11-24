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
"""Quantized jax.numpy.take with QArray support."""

import math
import operator
from typing import TypeAlias

import jax
from jax import lax
from jax import numpy as jnp
import numpy as np
from qwix._src.core import qarray

_StaticScalar: TypeAlias = bool | int | float | complex | np.bool_ | np.number
_MAX_ROWS = (1 << 31) - 1


def _canonicalize_axis(axis: int, ndim: int) -> int:
  """Canonicalizes `axis` into `[0, ndim)`."""
  axis_idx = operator.index(axis)
  if not -ndim <= axis_idx < ndim:
    raise ValueError(
        f'axis {axis} is out of bounds for array of dimension {ndim}.'
    )
  return axis_idx + ndim if axis_idx < 0 else axis_idx


def _prepare_indices(
    indices: jax.Array, axis_size: int, mode: str | None
) -> tuple[jax.Array, jax.Array | None]:
  """Normalizes `indices` into in-bounds indices and an optional OOB mask."""
  is_unsigned = jnp.issubdtype(indices.dtype, jnp.unsignedinteger)
  if is_unsigned:
    ids_w = (
        indices.astype(jnp.int32)
        if jnp.iinfo(indices.dtype).bits < 32
        else indices
    )
    if axis_size > jnp.iinfo(indices.dtype).max:
      # Every unsigned index is already strictly in `[0, axis_size)`.
      return ids_w, None
  else:
    target_dtype = (
        jnp.int32
        if axis_size <= _MAX_ROWS
        else jax.dtypes.canonicalize_dtype(jnp.int64)
    )
    ids_w = (
        indices.astype(target_dtype)
        if jnp.iinfo(indices.dtype).bits < jnp.iinfo(target_dtype).bits
        else indices
    )
    if axis_size > jnp.iinfo(ids_w.dtype).max:
      raise OverflowError(
          f'Axis size {axis_size} exceeds maximum {ids_w.dtype} index; enable'
          ' x64 mode or use unsigned indices.'
      )

  axis_size_val = jnp.asarray(axis_size, dtype=ids_w.dtype)
  max_index_val = jnp.asarray(axis_size - 1, dtype=ids_w.dtype)
  index_dtype = jnp.int32 if axis_size <= _MAX_ROWS else ids_w.dtype
  if mode == 'clip':
    if is_unsigned:
      in_bounds = jnp.minimum(ids_w, max_index_val)
    else:
      in_bounds = jnp.clip(ids_w, 0, max_index_val)
    return in_bounds.astype(index_dtype), None
  if mode == 'wrap':
    return jnp.mod(ids_w, axis_size_val).astype(index_dtype), None

  # Default mode (None or 'fill'): wrap negative signed indices and track OOB.
  if is_unsigned:
    oob = ids_w >= axis_size_val
    in_bounds = jnp.minimum(ids_w, max_index_val)
  else:
    wrapped = jnp.where(ids_w < 0, ids_w + axis_size_val, ids_w)
    oob = (wrapped < 0) | (wrapped >= axis_size_val)
    in_bounds = jnp.clip(wrapped, 0, max_index_val)
  return in_bounds.astype(index_dtype), oob


def _gather_axis(
    x: jax.Array,
    in_bounds: jax.Array,
    axis_idx: int,
    *,
    unique_indices: bool,
    indices_are_sorted: bool,
) -> jax.Array:
  """Gathers slices from `x` along `axis_idx` at in-bounds `in_bounds`."""
  index_dims = in_bounds.ndim
  slice_sizes = list(x.shape)
  slice_sizes[axis_idx] = 1
  dnums = lax.GatherDimensionNumbers(
      offset_dims=tuple(
          list(range(axis_idx))
          + list(range(axis_idx + index_dims, x.ndim + index_dims - 1))
      ),
      collapsed_slice_dims=(axis_idx,),
      start_index_map=(axis_idx,),
  )
  return lax.gather(
      x,
      in_bounds[..., None],
      dimension_numbers=dnums,
      slice_sizes=tuple(slice_sizes),
      unique_indices=unique_indices,
      indices_are_sorted=indices_are_sorted,
      mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS,
  )


def _gather_param(
    param: jax.Array,
    full_axis_size: int,
    in_bounds: jax.Array,
    axis_idx: int,
    *,
    unique_indices: bool,
    indices_are_sorted: bool,
) -> jax.Array:
  """Aligns a `scale` or `zero_point` leaf with gathered `qvalue`."""
  param_axis_size = param.shape[axis_idx]
  if param_axis_size == 1:
    new_shape = (
        param.shape[:axis_idx]
        + (1,) * in_bounds.ndim
        + param.shape[axis_idx + 1 :]
    )
    return param.reshape(new_shape)
  tile_size = full_axis_size // param_axis_size
  if tile_size == 1:
    param_indices = in_bounds
  elif tile_size > jnp.iinfo(in_bounds.dtype).max:
    param_indices = jnp.zeros_like(in_bounds)
  else:
    param_indices = in_bounds // jnp.asarray(tile_size, dtype=in_bounds.dtype)
  return _gather_axis(
      param,
      param_indices,
      axis_idx,
      unique_indices=unique_indices and tile_size == 1,
      indices_are_sorted=indices_are_sorted,
  )


def _apply_fill_value(
    rows: jax.Array,
    oob: jax.Array | None,
    axis_idx: int,
    ndim: int,
    fill_value: _StaticScalar | None,
) -> jax.Array:
  """Masks OOB entries in `rows` with `fill_value` (or `NaN` if `None`)."""
  if oob is None:
    return rows
  fill = (
      jnp.asarray(jnp.nan, dtype=rows.dtype)
      if fill_value is None
      else jnp.asarray(fill_value, dtype=rows.dtype)
  )
  oob_mask = oob.reshape(
      (1,) * axis_idx + oob.shape + (1,) * (ndim - 1 - axis_idx)
  )
  return jnp.where(oob_mask, fill, rows)


def take(
    a: qarray.MaybeQArray,
    indices: jax.typing.ArrayLike,
    axis: int | None = None,
    out: None = None,
    mode: str | None = None,
    unique_indices: bool = False,
    indices_are_sorted: bool = False,
    fill_value: _StaticScalar | None = None,
    *,
    out_sharding: (
        jax.sharding.NamedSharding | jax.sharding.PartitionSpec | None
    ) = None,
) -> jax.Array:
  """Takes elements from an array along an axis with ``QArray`` support.

  This function serves as a drop-in replacement for
  `jax.numpy.take
  <https://docs.jax.dev/en/latest/_autosummary/jax.numpy.take.html>`_.

  When ``a`` is a ``QArray``, ``take`` gathers compressed ``qvalue`` and
  corresponding ``scale`` / ``zero_point`` slices before dequantizing only the
  gathered output (avoiding materializing the full dequantized array, except
  when ``axis=None`` on a multi-dimensional ``QArray`` whose scale layout
  cannot be flattened). Negative signed indices wrap by the axis size like
  ``jnp.take`` (for example, ``-1`` reads the last element rather than acting
  as padding); callers needing zero-filled padding on out-of-bounds indices can
  pass ``fill_value=0.0``. Execution placement is left to XLA (which may
  automatically offload eligible gathers on SparseCore TPUs).

  Args:
    a: Array or ``QArray`` to take values from.
    indices: Integer indices of values to take.
    axis: Axis along which to take values. If ``None``, ``a`` is flattened
      before indexing.
    out: Unsupported (mirrors ``jnp.take``).
    mode: Out-of-bounds indexing mode: ``None`` / ``'fill'``, ``'clip'``, or
      ``'wrap'``. ``'raise'`` is unsupported.
    unique_indices: If ``True``, assumes normalized indices are unique.
    indices_are_sorted: If ``True``, assumes normalized indices are sorted.
    fill_value: Value used for out-of-bounds indices when ``mode`` is ``None``
      or ``'fill'``. If ``None``, defaults to ``NaN`` for floating-point
      outputs. Ignored for ``'clip'`` and ``'wrap'``.
    out_sharding: Optional output sharding specification (supported for plain
      arrays; raises ``NotImplementedError`` for ``QArray`` inputs).

  Returns:
    The gathered (and dequantized, for ``QArray``) array.

  Raises:
    NotImplementedError: If ``out`` is not ``None``, ``mode == 'raise'``, or
      ``out_sharding`` is set for a ``QArray``.
    ValueError: If ``mode``, ``indices``, ``axis``, ``fill_value``, or
      ``QArray`` metadata are invalid.
    IndexError: If taking non-empty ``indices`` from a size-0 axis.
    OverflowError: If the target axis size exceeds the representable signed
      index range in the active JAX precision mode.
  """
  if not isinstance(a, qarray.QArray):
    if out_sharding is not None:
      return jnp.take(
          a,
          indices,
          axis=axis,
          out=out,
          mode=mode,
          unique_indices=unique_indices,
          indices_are_sorted=indices_are_sorted,
          fill_value=fill_value,
          out_sharding=out_sharding,
      )
    return jnp.take(
        a,
        indices,
        axis=axis,
        out=out,
        mode=mode,
        unique_indices=unique_indices,
        indices_are_sorted=indices_are_sorted,
        fill_value=fill_value,
    )

  if out is not None:
    raise NotImplementedError("The 'out' argument to take is not supported.")
  if mode == 'raise':
    raise NotImplementedError("The 'raise' mode to take is not supported.")
  if mode not in (None, 'fill', 'clip', 'wrap'):
    raise ValueError(f'Invalid mode {mode!r} for take.')

  indices_arr = jnp.asarray(indices)
  if indices_arr.size != 0 and not jnp.issubdtype(
      indices_arr.dtype, jnp.integer
  ):
    raise ValueError(
        f'Expected integer indices; got dtype {indices_arr.dtype}.'
    )

  if fill_value is not None and np.ndim(fill_value) != 0:
    raise ValueError(
        f'Expected scalar fill_value; got shape {np.shape(fill_value)}.'
    )

  if out_sharding is not None:
    raise NotImplementedError(
        'out_sharding is not supported for QArray inputs to take.'
    )

  qa = qarray.QArray(
      jnp.asarray(a.qvalue),
      jnp.asarray(a.scale),
      None if a.zero_point is None else jnp.asarray(a.zero_point),
      a.qtype,
  )
  qarray.validate_qarray(qa)

  if axis is None:
    if qa.ndim != 1:
      flat_size = math.prod(qa.shape)
      if flat_size == 0:
        if indices_arr.size != 0:
          raise IndexError('Cannot do a non-empty take() from an empty axis.')
        return jnp.zeros(indices_arr.shape, dtype=qa.scale.dtype)
      try:
        qa = qarray.reshape(qa, (flat_size,))
      except ValueError:
        return jnp.take(
            qarray.dequantize(qa),
            indices_arr,
            axis=None,
            mode=mode,
            unique_indices=unique_indices,
            indices_are_sorted=indices_are_sorted,
            fill_value=fill_value,
        )
    axis_idx = 0
  else:
    axis_idx = _canonicalize_axis(axis, qa.ndim)

  axis_size = qa.shape[axis_idx]
  if axis_size == 0 and indices_arr.size != 0:
    raise IndexError('Cannot do a non-empty take() from an empty axis.')

  out_shape = qa.shape[:axis_idx] + indices_arr.shape + qa.shape[axis_idx + 1 :]
  if 0 in out_shape:
    return jnp.zeros(out_shape, dtype=qa.scale.dtype)

  in_bounds, oob = _prepare_indices(indices_arr, axis_size, mode)
  # Clamping OOB indices in fill/clip modes can map distinct inputs onto the
  # same boundary index, violating `lax.gather`'s `unique_indices=True`
  # backward-pass (scatter_add) contract.
  safe_unique_indices = unique_indices and mode == 'wrap'
  qvalue = _gather_axis(
      qa.qvalue,
      in_bounds,
      axis_idx,
      unique_indices=safe_unique_indices,
      indices_are_sorted=indices_are_sorted,
  )
  scale = _gather_param(
      qa.scale,
      axis_size,
      in_bounds,
      axis_idx,
      unique_indices=safe_unique_indices,
      indices_are_sorted=indices_are_sorted,
  )
  if qa.zero_point is None:
    zero_point = None
  else:
    zero_point = _gather_param(
        qa.zero_point,
        axis_size,
        in_bounds,
        axis_idx,
        unique_indices=safe_unique_indices,
        indices_are_sorted=indices_are_sorted,
    )
  rows = qarray.dequantize(qarray.QArray(qvalue, scale, zero_point, qa.qtype))
  return _apply_fill_value(rows, oob, axis_idx, qa.ndim, fill_value)
