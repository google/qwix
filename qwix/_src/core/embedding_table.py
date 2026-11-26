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
"""Quantized embedding table schemes and host (de)quantization helpers."""

import dataclasses
import math
from typing import TypeAlias, cast

import jax
from jax import numpy as jnp
import numpy as np
from qwix._src.core import numerics
from qwix._src.core import qarray

_DEFAULT_CHUNK_BYTES = 64 * 1024 * 1024
_ABSMAX_EPSILON = np.float32(1e-7)
_INT8_MAX = np.float32(127.0)
_INV_INT8_MAX = np.float32(1.0) / _INT8_MAX
_UINT8_MAX = np.float32(255.0)
_MAX_ROWS = int(np.iinfo(np.int32).max)


@dataclasses.dataclass(frozen=True)
class FixedRangeScheme:
  """Symmetric 256-bucket fixed-range uint8 embedding quantization scheme."""

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
    if abs(self.lower + self.upper) > 1e-6 * (self.upper - self.lower):
      raise ValueError(
          'FixedRangeScheme requires a symmetric zero-centered range '
          f'(lower == -upper); got lower={self.lower}, upper={self.upper}.'
      )
    with np.errstate(over='ignore', divide='ignore'):
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
  if qtable.qtype == 'uint8_centered':
    valid = qval_dtype == jnp.dtype(jnp.uint8)
  else:
    try:
      valid = jnp.dtype(qtable.qtype) == jnp.dtype(
          jnp.int8
      ) and qval_dtype == jnp.dtype(jnp.int8)
    except TypeError:
      valid = False
  if not valid:
    raise ValueError(
        f'Unsupported qtype {qtable.qtype!r} with qvalue dtype {qval_dtype}; '
        "expected 'uint8_centered' (uint8) or int8."
    )
  return num_rows, dim, qtable.scale.shape[0] > 1


def quantize_table(
    table: jax.typing.ArrayLike,
    scheme: Scheme,
    *,
    chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
) -> qarray.QArray:
  """Quantizes a 2D floating-point embedding table on the host in row chunks.

  Args:
    table: 2D floating-point table of shape `[N, D]`.
    scheme: `FixedRangeScheme` or `RowAbsmaxScheme`.
    chunk_bytes: Host chunk budget in bytes for peak-memory control.

  Returns:
    A `QArray` whose leaves are host `np.ndarray`s (no default-device
    allocation).
  """
  if not isinstance(scheme, (FixedRangeScheme, RowAbsmaxScheme)):
    raise ValueError(f'Unsupported scheme: {scheme!r}.')

  table_np = np.asarray(table)
  num_rows, dim = _validate_table_shape(table_np.shape)
  if not numerics.should_quantize(table_np.dtype):
    raise ValueError(
        f'Expected floating-point table dtype; got {table_np.dtype}.'
    )
  rows_per_chunk = _rows_per_chunk(
      chunk_bytes, dim * max(4, table_np.dtype.itemsize)
  )

  if isinstance(scheme, FixedRangeScheme):
    lower_f32 = np.float32(scheme.lower)
    upper_f32 = np.float32(scheme.upper)
    range_f32 = np.float32(upper_f32 - lower_f32)
    step = np.float32(range_f32 / _UINT8_MAX)
    inv_step = np.float32(_UINT8_MAX / range_f32)
    qvalue = np.empty((num_rows, dim), dtype=np.uint8)
    for start in range(0, num_rows, rows_per_chunk):
      end = min(start + rows_per_chunk, num_rows)
      chunk = np.asarray(table_np[start:end], dtype=np.float32)
      q = np.floor(
          (np.clip(chunk, lower_f32, upper_f32) - lower_f32) * inv_step
          + np.float32(0.5)
      )
      qvalue[start:end] = np.clip(q, np.float32(0.0), _UINT8_MAX).astype(
          np.uint8
      )
    scale = np.full((1, 1), step, dtype=np.float32)
    return qarray.QArray(
        cast(jax.Array, qvalue), cast(jax.Array, scale), qtype='uint8_centered'
    )

  qvalue = np.empty((num_rows, dim), dtype=np.int8)
  scale = np.empty((num_rows, 1), dtype=np.float32)
  for start in range(0, num_rows, rows_per_chunk):
    end = min(start + rows_per_chunk, num_rows)
    chunk = np.asarray(table_np[start:end], dtype=np.float32)
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
  """
  num_rows, dim, per_row_scale = _validate_qtable(qtable)
  rows_per_chunk = _rows_per_chunk(chunk_bytes, dim * 4)
  cpu = jax.devices('cpu')[0]
  out = np.empty((num_rows, dim), dtype=np.float32)
  with jax.default_device(cpu):
    shared_scale = None if per_row_scale else jax.device_put(qtable.scale, cpu)
    for start in range(0, num_rows, rows_per_chunk):
      end = min(start + rows_per_chunk, num_rows)
      qval_chunk = jax.device_put(qtable.qvalue[start:end], cpu)
      scale_chunk = (
          jax.device_put(qtable.scale[start:end], cpu)
          if shared_scale is None
          else shared_scale
      )
      chunk_dq = qarray.dequantize(
          qarray.QArray(qval_chunk, scale_chunk, qtype=qtable.qtype)
      )
      out[start:end] = np.asarray(chunk_dq, dtype=np.float32)
  return out
