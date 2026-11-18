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
"""Quantized embedding table helpers and TPU serving lookup ops."""

from collections.abc import Callable
import dataclasses
import functools
import math
from typing import Literal, TypeAlias, TypeVar, cast

import jax
from jax import lax
from jax import numpy as jnp
from jax.experimental import compute_on
import numpy as np
from qwix._src.core import numerics
from qwix._src.core import qarray

Placement: TypeAlias = Literal['tc', 'sc_gather', 'sc']

_T = TypeVar('_T')
_DEFAULT_CHUNK_BYTES = 64 * 1024 * 1024
_ABSMAX_EPSILON = np.float32(1e-7)
_INT8_MAX = np.float32(127.0)
_INV_INT8_MAX = np.float32(1.0) / _INT8_MAX
_UINT8_MAX = np.float32(255.0)
_MAX_ROWS = int(np.iinfo(np.int32).max)

_PLACEMENT_COMPUTE_TYPE = {
    'tc': 'device',
    'sc_gather': 'tpu_sparsecore',
    'sc': 'tpu_sparsecore',
}

_GATHER_DNUMS = lax.GatherDimensionNumbers(
    offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,)
)
_SCATTER_DNUMS = lax.ScatterDimensionNumbers(
    update_window_dims=(1,),
    inserted_window_dims=(0,),
    scatter_dims_to_operand_dims=(0,),
)


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


def _validate_placement(placement: Placement) -> str:
  """Validates `placement` and returns its JAX `compute_on` compute type."""
  compute_type = _PLACEMENT_COMPUTE_TYPE.get(placement)
  if compute_type is None:
    raise ValueError(
        f"Unsupported placement {placement!r}; expected 'tc', 'sc_gather', or"
        " 'sc'."
    )
  return compute_type


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


def _validate_qtable(
    qtable: qarray.QArray, *, require_f32_scale: bool = False
) -> tuple[int, int, bool]:
  """Validates a 2D quantized table and returns `(N, D, per_row_scale)`."""
  num_rows, dim = _validate_table_shape(qtable.shape)
  if qtable.zero_point is not None:
    raise ValueError('Asymmetric quantization (zero_point) is not supported.')
  scale_dtype = jnp.dtype(qtable.scale.dtype)
  if require_f32_scale:
    if scale_dtype != jnp.dtype(jnp.float32):
      raise ValueError(
          f'Unsupported scale dtype {scale_dtype}; expected float32.'
      )
  elif not numerics.should_quantize(scale_dtype):
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


def _prepare_ids(
    ids: jax.typing.ArrayLike, num_rows: int
) -> tuple[jax.Array, jax.Array]:
  """Normalizes `ids` into clamped `[M, 1]` int32 indices and an OOB mask."""
  ids_arr = jnp.asarray(ids)
  if not jnp.issubdtype(ids_arr.dtype, jnp.integer):
    raise ValueError(f'Expected integer ids; got dtype {ids_arr.dtype}.')
  is_unsigned = jnp.issubdtype(ids_arr.dtype, jnp.unsignedinteger)
  # Widen <32-bit integer dtypes before comparing or wrapping against num_rows
  # so neither signed `ids + num_rows` nor unsigned `ids >= num_rows` overflows
  # or wraps modulo 2**bits.
  ids_w = (
      ids_arr.astype(jnp.int32)
      if jax.dtypes.itemsize_bits(ids_arr.dtype) < 32
      else ids_arr
  )
  if is_unsigned:
    oob = ids_w >= num_rows
    flat_indices = jnp.minimum(ids_w.reshape(-1), num_rows - 1).astype(
        jnp.int32
    )[:, None]
  else:
    wrapped = jnp.where(ids_w < 0, ids_w + num_rows, ids_w)
    oob = (wrapped < 0) | (wrapped >= num_rows)
    flat_indices = jnp.clip(wrapped.reshape(-1), 0, num_rows - 1).astype(
        jnp.int32
    )[:, None]
  return flat_indices, oob


def _gather_rows(table: jax.Array, indices: jax.Array) -> jax.Array:
  """Gathers rows at 2D `[M, 1]` in-bounds `indices` from a 2D `table`."""
  return lax.gather(
      table,
      indices,
      _GATHER_DNUMS,
      slice_sizes=(1, table.shape[1]),
      mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS,
  )


def _dequantize_rows(
    codes: jax.Array, scale: jax.Array, qtype: jax.typing.DTypeLike
) -> jax.Array:
  """Dequantizes gathered codes with `float32` `scale`."""
  return numerics.convert_from(codes, qtype).astype(jnp.float32) * scale


def _gather_dequantize(
    qvalue: jax.Array,
    scale: jax.Array,
    indices: jax.Array,
    *,
    per_row_scale: bool,
    qtype: jax.typing.DTypeLike,
) -> jax.Array:
  """Gathers and dequantizes rows at in-bounds `[M, 1]` `indices`."""
  codes = _gather_rows(qvalue, indices)
  scale_rows = _gather_rows(scale, indices) if per_row_scale else scale
  return _dequantize_rows(codes, scale_rows, qtype)


def _run_in_placement(
    compute_type: str,
    fn: Callable[..., _T],
    *args: jax.Array,
) -> _T:
  """Stages `fn(*args)` inside a TPU `compute_on` region, or directly off TPU."""
  tpu_fn = compute_on.compute_on(
      compute_type=compute_type, out_memory_spaces=jax.memory.Space.Device
  )(fn)
  return jax.lax.platform_dependent(*args, tpu=tpu_fn, default=fn)


def take(
    qtable: qarray.QArray,
    ids: jax.typing.ArrayLike,
    *,
    placement: Placement = 'tc',
) -> jax.Array:
  """Gathers and dequantizes rows at `ids` in `float32`.

  Negative signed IDs wrap by `N`; out-of-bounds IDs return `NaN` rows.

  Args:
    qtable: 2D quantized embedding table of shape `[N, D]` with `float32` scale.
    ids: Integer row indices of arbitrary shape.
    placement: Execution placement (`'tc'`, `'sc_gather'`, or `'sc'`).

  Returns:
    Float32 array of shape `[*ids.shape, D]`.
  """
  compute_type = _validate_placement(placement)
  num_rows, dim, per_row_scale = _validate_qtable(
      qtable, require_f32_scale=True
  )
  flat_indices, oob = _prepare_ids(ids, num_rows)
  qvalue = jnp.asarray(qtable.qvalue)
  scale = jnp.asarray(qtable.scale)
  qtype = qtable.qtype

  if placement == 'sc_gather':
    if per_row_scale:
      codes, scale_rows = _run_in_placement(
          compute_type,
          lambda qv, sc, idx: (_gather_rows(qv, idx), _gather_rows(sc, idx)),
          qvalue,
          scale,
          flat_indices,
      )
    else:
      codes = _run_in_placement(
          compute_type, _gather_rows, qvalue, flat_indices
      )
      scale_rows = scale
    rows = _dequantize_rows(codes, scale_rows, qtype)
  else:
    rows = _run_in_placement(
        compute_type,
        functools.partial(
            _gather_dequantize, per_row_scale=per_row_scale, qtype=qtype
        ),
        qvalue,
        scale,
        flat_indices,
    )

  rows = rows.reshape((*oob.shape, dim))
  return jnp.where(oob[..., None], jnp.float32(jnp.nan), rows)


def lookup(
    qtable: qarray.QArray,
    ids: jax.typing.ArrayLike,
    weights: jax.typing.ArrayLike,
    *,
    placement: Placement = 'tc',
) -> jax.Array:
  """Computes weighted embedding bag sums of shape `[B, D]` in `float32`.

  Negative signed IDs wrap by `N`; any bag containing an out-of-bounds ID
  returns a `NaN` row. `'sc'` does not support `jax.vmap` on TPU; flatten
  batch dimensions into `B`.

  Args:
    qtable: 2D quantized embedding table of shape `[N, D]` with `float32` scale.
    ids: 2D `[B, K]` integer row indices.
    weights: 2D `[B, K]` floating-point per-element bag weights.
    placement: Execution placement (`'tc'`, `'sc_gather'`, or `'sc'`).

  Returns:
    Float32 array of shape `[B, D]`.
  """
  compute_type = _validate_placement(placement)
  ids_arr = jnp.asarray(ids)
  weights_arr = jnp.asarray(weights)
  if ids_arr.ndim != 2 or weights_arr.shape != ids_arr.shape:
    raise ValueError(
        f'Expected matching 2D [B, K] ids and weights; got ids={ids_arr.shape},'
        f' weights={weights_arr.shape}.'
    )
  if not numerics.should_quantize(weights_arr.dtype):
    raise ValueError(
        f'Expected floating-point weights dtype; got {weights_arr.dtype}.'
    )
  weights_f32 = weights_arr.astype(jnp.float32)

  if placement != 'sc':
    rows = take(qtable, ids_arr, placement=placement)
    return jnp.sum(rows * weights_f32[..., None], axis=1)

  num_rows, dim, per_row_scale = _validate_qtable(
      qtable, require_f32_scale=True
  )
  flat_indices, oob = _prepare_ids(ids_arr, num_rows)
  batch, bag_size = ids_arr.shape
  bag_ids = jnp.repeat(jnp.arange(batch, dtype=jnp.int32), bag_size)[:, None]
  init_sums = jnp.zeros((batch, dim), dtype=jnp.float32)

  def _sc_bag_sum(qv, sc, idx, b_ids, w, sums):
    values = _gather_dequantize(
        qv, sc, idx, per_row_scale=per_row_scale, qtype=qtable.qtype
    )
    return lax.scatter_add(
        sums,
        b_ids,
        values * w[:, None],
        _SCATTER_DNUMS,
        indices_are_sorted=True,
        mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )

  sums = _run_in_placement(
      compute_type,
      _sc_bag_sum,
      jnp.asarray(qtable.qvalue),
      jnp.asarray(qtable.scale),
      flat_indices,
      bag_ids,
      weights_f32.reshape(-1),
      init_sums,
  )
  return jnp.where(jnp.any(oob, axis=1)[:, None], jnp.float32(jnp.nan), sums)
