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
"""Quantized embedding table schemes, host quantizer, and SparseCore take."""

from collections.abc import Iterator
import dataclasses
import functools
import math
from typing import TypeAlias, cast

import jax
from jax import lax
from jax import numpy as jnp
from jax.experimental import compute_on
import numpy as np
from qwix._src.core import numerics
from qwix._src.core import qarray

_StaticScalar: TypeAlias = bool | int | float | complex | np.bool_ | np.number

_DEFAULT_CHUNK_BYTES = 64 * 1024 * 1024
_ABSMAX_EPSILON = np.float32(1e-7)
_INT8_MAX = np.float32(127.0)
_INV_INT8_MAX = np.float32(1.0) / _INT8_MAX
_UINT8_MAX = np.float32(255.0)
_ZERO_OFFSET = np.float32(128.0)
_MAX_ROWS = int(np.iinfo(np.int32).max)
_ROW_GATHER_DNUMS = lax.GatherDimensionNumbers(
    offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)
)
_ELEM_GATHER_DNUMS = lax.GatherDimensionNumbers(
    offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)
)


@dataclasses.dataclass(frozen=True)
class FixedRangeScheme:
  """Zero-aligned 256-bucket fixed-range int8 embedding quantization scheme.

  Requires bounds with exact zero on bucket 128 of the 256-level grid:
  `lower = -128 * step` and `upper = 127 * step` where
  `step = (upper - lower) / 255` (within `1e-5` relative tolerance), stored as
  signed `int8` codes (`bucket - 128`) with scale `step`. General
  `lower < upper` and literal symmetric `lower == -upper` bounds are rejected;
  to approximate a symmetric range `[-m, m]`, pass
  `lower = -128.0 / 127.0 * m, upper = m`.

  Attributes:
    lower: Inclusive lower quantization bound (`-128 * step`).
    upper: Inclusive upper quantization bound (`127 * step`).
  """

  lower: float
  upper: float

  def __post_init__(self) -> None:
    if not (math.isfinite(self.lower) and math.isfinite(self.upper)):
      raise ValueError(
          f'Bounds must be finite; got lower={self.lower}, upper={self.upper}.'
      )
    if self.lower >= self.upper:
      raise ValueError(
          f'Expected lower < upper; got lower={self.lower}, upper={self.upper}.'
      )
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
      range_f32 = np.float32(self.upper) - np.float32(self.lower)
      step_f32 = np.float32(range_f32 / _UINT8_MAX)
      inv_step_f32 = np.float32(_UINT8_MAX / range_f32)
    if not (
        np.isfinite(range_f32) and step_f32 > 0 and np.isfinite(inv_step_f32)
    ):
      raise ValueError(
          f'Range [{self.lower}, {self.upper}] is not cleanly representable in'
          ' float32.'
      )
    span = self.upper - self.lower
    if abs(128.0 * (span / 255.0) + self.lower) > 1e-5 * span:
      raise ValueError(
          'FixedRangeScheme requires zero-aligned bounds '
          '(lower = -128 * step, upper = 127 * step where '
          f'step = (upper - lower) / 255); got lower={self.lower}, '
          f'upper={self.upper}.'
      )


@dataclasses.dataclass(frozen=True)
class RowAbsmaxScheme:
  """Per-row dynamic symmetric int8 embedding quantization scheme."""


Scheme: TypeAlias = FixedRangeScheme | RowAbsmaxScheme


def _validate_table_shape(shape: tuple[int, ...]) -> tuple[int, int]:
  """Validates a 2D `[N, D]` embedding table shape and returns `(N, D)`."""
  if len(shape) != 2 or not (0 < shape[0] <= _MAX_ROWS) or shape[1] < 1:
    raise ValueError(
        f'Expected 2D table [N, D] with 0 < N <= {_MAX_ROWS} and D >= 1; got'
        f' shape {shape}.'
    )
  return shape[0], shape[1]


def _rows_per_chunk(chunk_bytes: int, row_bytes: int) -> int:
  """Returns the positive row chunk size for a host byte budget."""
  if chunk_bytes <= 0:
    raise ValueError(f'chunk_bytes must be positive; got {chunk_bytes}.')
  return max(1, chunk_bytes // row_bytes)


def _validate_qtable(qtable: qarray.QArray) -> tuple[int, int, bool]:
  """Validates a 2D quantized table and returns `(N, D, per_row_scale)`."""
  num_rows, dim = _validate_table_shape(qtable.shape)
  if qtable.zero_point is not None:
    raise ValueError('Asymmetric quantization (zero_point) is not supported.')
  scale_dtype = jnp.dtype(qtable.scale.dtype)
  if not numerics.should_quantize(scale_dtype):
    raise ValueError(
        f'Unsupported scale dtype {scale_dtype}; expected floating point.'
    )
  if qtable.scale.shape not in ((1, 1), (num_rows, 1)):
    raise ValueError(
        f'Expected scale shape (1, 1) or ({num_rows}, 1); got '
        f'{qtable.scale.shape}.'
    )
  qval_dtype = jnp.dtype(qtable.qvalue.dtype)
  try:
    valid = jnp.dtype(qtable.qtype) == jnp.dtype(
        jnp.int8
    ) and qval_dtype == jnp.dtype(jnp.int8)
  except TypeError:
    valid = False
  if not valid:
    raise ValueError(
        f'Unsupported qtype {qtable.qtype!r} with qvalue dtype {qval_dtype}; '
        'expected int8.'
    )
  return num_rows, dim, qtable.scale.shape[0] > 1


def _iter_f32_chunks(
    table: np.ndarray | jax.Array, rows_per_chunk: int
) -> Iterator[tuple[int, int, np.ndarray]]:
  """Yields `(start, end, f32_chunk)` slices and rejects non-finite values."""
  num_rows = table.shape[0]
  for start in range(0, num_rows, rows_per_chunk):
    end = min(start + rows_per_chunk, num_rows)
    with np.errstate(over='ignore'):
      chunk = np.asarray(table[start:end], dtype=np.float32)
    if not np.isfinite(chunk).all():
      raise ValueError(
          'Expected finite float32-representable values in table '
          f'(rows {start}:{end}).'
      )
    yield start, end, chunk


def _prepare_row_indices(
    indices: jax.Array, num_rows: int, mode: str | None
) -> tuple[jax.Array, jax.Array | None]:
  """Normalizes `indices` into in-bounds `int32` indices and optional OOB."""
  is_unsigned = jnp.issubdtype(indices.dtype, jnp.unsignedinteger)
  ids_w = (
      indices.astype(jnp.int32)
      if jnp.iinfo(indices.dtype).bits < 32
      else indices
  )
  if is_unsigned and num_rows > jnp.iinfo(indices.dtype).max:
    # Every unsigned index is already strictly in `[0, num_rows)`.
    return ids_w, None
  if mode == 'clip':
    in_bounds = (
        jnp.minimum(ids_w, num_rows - 1)
        if is_unsigned
        else jnp.clip(ids_w, 0, num_rows - 1)
    )
    return in_bounds.astype(jnp.int32), None
  if mode == 'wrap':
    return jnp.mod(ids_w, num_rows).astype(jnp.int32), None

  if is_unsigned:
    oob = ids_w >= num_rows
    in_bounds = jnp.minimum(ids_w, num_rows - 1)
  else:
    wrapped = jnp.where(ids_w < 0, ids_w + num_rows, ids_w)
    oob = (wrapped < 0) | (wrapped >= num_rows)
    in_bounds = jnp.clip(wrapped, 0, num_rows - 1)
  return in_bounds.astype(jnp.int32), oob


def _gather_arrays(
    indices: jax.Array, *arrays: jax.Array
) -> tuple[jax.Array, ...]:
  """Gathers elements/rows at 1D `[M]` in-bounds `indices` from 1D/2D arrays."""
  indices_2d = indices[:, None]
  return tuple(
      lax.gather(
          arr,
          indices_2d,
          _ELEM_GATHER_DNUMS if arr.ndim == 1 else _ROW_GATHER_DNUMS,
          slice_sizes=(1,) if arr.ndim == 1 else (1, arr.shape[1]),
          mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS,
      )
      for arr in arrays
  )


@functools.cache
def _sparsecore_gather():
  """Returns the TPU SparseCore-annotated `_gather_arrays` callable."""
  # JAX 0.10.1-0.11.0 exposes the decorator as compute_on2 (renamed in 0.11.1).
  compute_on_fn = (
      getattr(compute_on, 'compute_on2', None) or compute_on.compute_on
  )
  return compute_on_fn(
      compute_type='tpu_sparsecore', out_memory_spaces=jax.memory.Space.Device
  )(_gather_arrays)


def _gather_axis0(
    in_bounds: jax.Array, *arrays: jax.Array
) -> tuple[jax.Array, ...]:
  """Gathers rows along `axis=0` from 2D arrays, offloading `M > 1` to SC."""
  if in_bounds.size == 1:
    # A single-row lookup has no SparseCore benefit and avoids a TPU compiler
    # crash on single-index SparseCore gathers.
    idx = in_bounds.reshape(())
    return tuple(
        lax.dynamic_slice_in_dim(arr, idx, 1, axis=0) for arr in arrays
    )
  # Stage 1D `[M]` indices and 1D `[N]` operands (`[N, 1]` scales and `D == 1`
  # tables) into the SparseCore region; 2D `[K, 1]` operands can crash the
  # TPU compiler for large unaligned `K`.
  operands = tuple(
      arr.reshape((arr.shape[0],)) if arr.shape[1] == 1 else arr
      for arr in arrays
  )
  return lax.platform_dependent(
      in_bounds.reshape(-1),
      *operands,
      tpu=_sparsecore_gather(),
      default=_gather_arrays,
  )


def quantize_table(
    table: jax.typing.ArrayLike,
    scheme: Scheme,
    *,
    chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
) -> qarray.QArray:
  """Quantizes a 2D floating-point embedding table on the host in row chunks.

  Args:
    table: 2D floating-point table of shape `[N, D]` with finite
      float32-representable values.
    scheme: `FixedRangeScheme` or `RowAbsmaxScheme`.
    chunk_bytes: Host chunk budget in bytes for peak-memory control.

  Returns:
    A `QArray` whose leaves are host `np.ndarray`s (no default-device
    allocation).

  Raises:
    ValueError: If `scheme`, `table` shape or dtype, `chunk_bytes`, or any
      value in `table` (non-finite after float32 conversion) is invalid.
  """
  if not isinstance(scheme, (FixedRangeScheme, RowAbsmaxScheme)):
    raise ValueError(f'Unsupported scheme: {scheme!r}.')

  table_arr = (
      table if isinstance(table, (np.ndarray, jax.Array)) else np.asarray(table)
  )
  num_rows, dim = _validate_table_shape(table_arr.shape)
  if not numerics.should_quantize(table_arr.dtype):
    raise ValueError(
        f'Expected floating-point table dtype; got {table_arr.dtype}.'
    )
  rows_per_chunk = _rows_per_chunk(
      chunk_bytes, dim * max(4, table_arr.dtype.itemsize)
  )

  qvalue = np.empty((num_rows, dim), dtype=np.int8)
  if isinstance(scheme, FixedRangeScheme):
    # Match `jax-tpu-embedding` SparseCore fixed-range QAT's float32 op order
    # (`floor((clip(x, lower, upper) - lower) * (255 / range) + 0.5)`), shift
    # by -128 into signed int8 so 0.0 maps to code 0, always emit float32
    # scales, and avoid JAX tracing overhead on host.
    lower_f32 = np.float32(scheme.lower)
    upper_f32 = np.float32(scheme.upper)
    range_f32 = np.float32(upper_f32 - lower_f32)
    step = np.float32(range_f32 / _UINT8_MAX)
    inv_step = np.float32(_UINT8_MAX / range_f32)
    for start, end, chunk in _iter_f32_chunks(table_arr, rows_per_chunk):
      q = np.floor(
          (np.clip(chunk, lower_f32, upper_f32) - lower_f32) * inv_step
          + np.float32(0.5)
      )
      qvalue[start:end] = (
          np.clip(q, np.float32(0.0), _UINT8_MAX) - _ZERO_OFFSET
      ).astype(np.int8)
    scale = np.full((1, 1), step, dtype=np.float32)
  else:
    # Match `jax-tpu-embedding` SparseCore absmax QAT bit-for-bit: symmetric
    # [-127, 127] grid (`qarray.quantize` uses bound 127.5), 1e-7 floor,
    # half-up rounding, reciprocal 1/127 scale, and float32 scales.
    scale = np.empty((num_rows, 1), dtype=np.float32)
    for start, end, chunk in _iter_f32_chunks(table_arr, rows_per_chunk):
      absmax = np.maximum(
          np.max(np.abs(chunk), axis=1, keepdims=True), _ABSMAX_EPSILON
      )
      q = np.floor(chunk * (_INT8_MAX / absmax) + np.float32(0.5))
      qvalue[start:end] = np.clip(q, -_INT8_MAX, _INT8_MAX).astype(np.int8)
      scale[start:end] = absmax * _INV_INT8_MAX
  return qarray.QArray(
      cast(jax.Array, qvalue), cast(jax.Array, scale), qtype=jnp.int8
  )


def dequantize_table(
    qtable: qarray.QArray,
    *,
    chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
) -> np.ndarray:
  """Dequantizes a quantized embedding table on host CPU in row chunks.

  Delegates each chunk to `qarray.dequantize`, which multiplies in
  `qtable.scale.dtype` (e.g. `bfloat16`) before converting to `float32`.

  Args:
    qtable: 2D quantized embedding table of shape `[N, D]`.
    chunk_bytes: Host chunk budget in bytes for peak-memory control.

  Returns:
    Host `float32` `np.ndarray` of shape `[N, D]`.

  Raises:
    ValueError: If `qtable` or `chunk_bytes` is invalid.
  """
  num_rows, dim, per_row_scale = _validate_qtable(qtable)
  rows_per_chunk = _rows_per_chunk(chunk_bytes, dim * 4)
  cpu = jax.devices('cpu')[0]
  out = np.empty((num_rows, dim), dtype=np.float32)
  shared_scale = None if per_row_scale else jax.device_put(qtable.scale, cpu)
  for start in range(0, num_rows, rows_per_chunk):
    end = min(start + rows_per_chunk, num_rows)
    qval_chunk = jax.device_put(qtable.qvalue[start:end], cpu)
    scale_chunk = (
        jax.device_put(qtable.scale[start:end], cpu)
        if shared_scale is None
        else shared_scale
    )
    with jax.default_device(cpu):
      chunk_dq = qarray.dequantize(
          qarray.QArray(qval_chunk, scale_chunk, qtype=qtable.qtype)
      )
    out[start:end] = np.asarray(chunk_dq, dtype=np.float32)
  return out


def sparsecore_take(
    table: qarray.MaybeQArray,
    indices: jax.typing.ArrayLike,
    *,
    mode: str | None = None,
    fill_value: _StaticScalar | None = None,
) -> jax.Array:
  """Gathers rows along `axis=0` on TPU SparseCore, dequantizing on TensorCore.

  Follows `jax.numpy.take(..., axis=0)` indexing semantics while explicitly
  staging the row gather(s) inside a `compute_on('tpu_sparsecore')` region.
  Use this when explicit SparseCore gather offload is needed in cases where
  XLA's automatic gather offload declines (for example, packed `int8` tables
  kept on TensorCore by VMEM/cost heuristics, modules combining other
  SparseCore embedding ops, or `float32` baseline tables). Negative signed
  indices wrap by the table's row count like `jnp.take` (for example, `-1`
  reads the last row rather than acting as padding); callers needing zero-filled
  padding on out-of-bounds indices can pass `fill_value=0.0` (or use an in-range
  ID with weight `0` for weighted bag sums). Requires `libtpu >= 0.0.51` on
  SparseCore TPUs (includes SparseCore signed `int8` conversion and vectorized
  1D pad-emitter support for arbitrary batch sizes). Not supported under
  `jax.vmap` on SparseCore TPUs (pass multi-dimensional `indices` directly
  instead). On non-TPU backends or TPUs without SparseCore, falls back to
  standard execution.

  Args:
    table: 2D `[N, D]` floating-point array or symmetric `int8` `QArray` with
      `float32` scale of shape `(1, 1)` or `(N, 1)`.
    indices: Integer indices of rows to gather.
    mode: Out-of-bounds indexing mode: `None` / `'fill'`, `'clip'`, or `'wrap'`.
      `'raise'` is unsupported.
    fill_value: Value used for out-of-bounds indices when `mode` is `None` or
      `'fill'`. Defaults to `NaN` if `None`. Ignored for `'clip'` and `'wrap'`.

  Returns:
    Gathered (and dequantized, for `QArray`) rows of shape
    `(*indices.shape, D)`.

  Raises:
    NotImplementedError: If `mode == 'raise'`.
    ValueError: If `mode`, `indices`, `fill_value`, or `table` preconditions
      are invalid.
  """
  if mode == 'raise':
    raise NotImplementedError(
        "The 'raise' mode to sparsecore_take is not supported."
    )
  if mode not in (None, 'fill', 'clip', 'wrap'):
    raise ValueError(f'Invalid mode {mode!r} for sparsecore_take.')

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

  if isinstance(table, qarray.QArray):
    num_rows, dim, per_row_scale = _validate_qtable(table)
    if jnp.dtype(table.scale.dtype) != jnp.dtype(jnp.float32):
      raise ValueError(
          f'sparsecore_take requires float32 scale; got {table.scale.dtype}.'
      )
    if indices_arr.size == 0:
      return jnp.zeros((*indices_arr.shape, dim), dtype=jnp.float32)
    in_bounds, oob = _prepare_row_indices(indices_arr, num_rows, mode)
    qvalue = jnp.asarray(table.qvalue)
    scale = jnp.asarray(table.scale)
    if per_row_scale:
      codes, scale_rows = _gather_axis0(in_bounds, qvalue, scale)
      scale_rows = scale_rows.reshape((*in_bounds.shape, 1))
    else:
      (codes,) = _gather_axis0(in_bounds, qvalue)
      scale_rows = scale.reshape((1,) * in_bounds.ndim + (1,))
    # Reshape gathered codes before dequantizing so XLA keeps them in VMEM and
    # fuses TensorCore dequantization with downstream OOB masking/reductions.
    rows = qarray.dequantize(
        qarray.QArray(
            codes.reshape((*in_bounds.shape, dim)),
            scale_rows,
            qtype=table.qtype,
        )
    )
  else:
    arr = jnp.asarray(table)
    num_rows, dim = _validate_table_shape(arr.shape)
    if not numerics.should_quantize(arr.dtype):
      raise ValueError(
          'sparsecore_take requires a floating-point array or int8 QArray; '
          f'got dtype {arr.dtype}.'
      )
    if indices_arr.size == 0:
      return jnp.zeros((*indices_arr.shape, dim), dtype=arr.dtype)
    in_bounds, oob = _prepare_row_indices(indices_arr, num_rows, mode)
    (rows,) = _gather_axis0(in_bounds, arr)
    rows = rows.reshape((*in_bounds.shape, dim))

  if oob is not None:
    fill = (
        jnp.asarray(jnp.nan, dtype=rows.dtype)
        if fill_value is None
        else jnp.asarray(fill_value, dtype=rows.dtype)
    )
    rows = jnp.where(oob[..., None], fill, rows)
  return rows
