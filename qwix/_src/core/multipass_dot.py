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
"""Multi-pass for fp8 formats.

This module provides multi-pass residual quantization emulation:
- 2-pass residual decomposition:
    Pass 0: X_0 = quantize(X),   R_1 = X - dequantize(X_0)
    Pass 1: X_1 = quantize(R_1)
- Multi-pass matrix multiplication modes:
    1. 'three_pass_fp8': A_0 B_0 + A_0 B_1 + A_1 B_0
       Drops second-order cross-residual A_1 B_1.
    2. 'four_pass_fp8': A_0 B_0 + A_0 B_1 + A_1 B_0 + A_1 B_1
       Evaluates all 4 cross-products.
    3. 'two_pass_lhs_fp8': A_0 B_0 + A_1 B_0
       LHS uses 2 residual passes, RHS uses 1 pass.
    4. 'two_pass_rhs_fp8': A_0 B_0 + A_0 B_1
       LHS uses 1 pass, RHS uses 2 residual passes.

Additionally it supports int8 by decomposing into int4. This could be useful on
devices with high int4 FLOPs. For the purposes of emulation we do this on the
fp8 native path.

All decomposed passes are evaluated on quantized operands directly through the
hardware FP8 path (via `_fast_dot_general` with hardware FP8 accumulation) or
via `jax.nn.scaled_matmul` on supported GPUs.

Scaling factor design choice for residual passes:
While an alternative formulation could independently compute fresh scaling
factors for each residual pass (via block-level max-reduction searches over
R_1), we instead derive residual scales by shifting the initial pass's scale
factor by a fixed power-of-2 exponent offset (e.g. 2^-4 for FP8 E4M3).
This design choice is made because:

1. Hardware efficiency: It eliminates the expensive second reduction
   tree across elements, replacing dynamic exponent search with a single ALU
   integer shift (E_1 = E_0 - shift_bits).
2. Memory bandwidth and storage: The residual scale is implicit rather than a
   separate scale tensor that must be stored and loaded from memory, cutting
   scale metadata traffic.
3. Fixed accumulator alignment: Constant power-of-2 scale offsets allow matrix
   units to align cross-term products with simple arithmetic bit-shifts prior
   to register accumulation, rather than performing variable floating-
   point scale multiplications.
4. Numerical closeness: Because the maximum rounding residual of round-to-
   nearest is bounded by the top bin step size, shifting by these exact bits
   yields similar SQNR to independent scale choices.
"""

from typing import Any, Literal, TypeAlias, get_args

import jax
import jax.numpy as jnp
from qwix._src.core import dot_general as dg
from qwix._src.core import qarray

# Multi-pass quantization is an experimental feature. Modes and APIs are
# subject to change in future releases.
MultiPassMode: TypeAlias = Literal[
    'three_pass_fp8',
    'four_pass_fp8',
    'two_pass_lhs_fp8',
    'two_pass_rhs_fp8',
    'four_pass_int4',
]


def _get_residual_scale_shift_bits(qtype: jax.typing.DTypeLike) -> int | None:
  """Returns the power-of-2 scale shift bits for residual quantization.

  For floating-point formats, the residual quantization scale can be derived
  directly from the preceding pass's scale by shifting its exponent, eliminating
  redundant block-level reduction passes:
  - fp8 / float8_e4m3fn (E4M3): shift by 4 bits (2^-4 = 1/16).
  - float8_e5m2 (E5M2): shift by 2 bits (2^-2 = 1/4).

  For integer formats (e.g. int4, int8), returns None to indicate
  exact algebraic decomposition or independent calibration.

  Args:
    qtype: Quantization format or dtype string.

  Returns:
    The integer exponent shift in bits, or None.
  """
  # Normalize synthetic string aliases to standard JAX dtypes.
  match qtype:
    case 'float8_e4m3' | 'fp8':
      qtype = jnp.float8_e4m3fn

  try:
    dt = jnp.dtype(qtype)
  except (TypeError, ValueError):
    return None

  if dt == jnp.float8_e4m3fn:
    return 4
  if dt == jnp.float8_e5m2:
    return 2
  return None


def _residual_decompose(
    x: qarray.MaybeQArray,
    how: qarray.HowToQuantize,
    n_passes: int = 2,
) -> tuple[qarray.QArray, ...]:
  """Decomposes tensor x into residual quantized passes.

  For floating-point formats (e.g., float8_e4m3fn), residual passes reuse the
  initial pass's scale factor shifted by the format's bit-precision (e.g. 2^-4
  for FP8), eliminating redundant block-level reduction passes. For integer
  formats, passes are calibrated independently.

  If x is already a QArray, it is used as the first pass, and subsequent passes
  are zero.

  Args:
    x: Input array or QArray.
    how: HowToQuantize configuration specifying format, tile size, scaling, etc.
    n_passes: Number of residual passes (default 2).

  Returns:
    Tuple of quantized QArray objects (q_0, q_1, ...).
  """
  if isinstance(x, qarray.QArray):
    raise ValueError(
        'Input to _residual_decompose must strictly be unquantized jax.Array,'
        f' but got {type(x)}.'
    )

  passes = []
  current = x
  shift_bits = _get_residual_scale_shift_bits(how.qtype)

  # Pass 0
  q0 = qarray.quantize(current, how)
  passes.append(q0)
  deq0 = qarray.dequantize(q0)
  current = current - deq0

  base_scale = q0.scale
  for p in range(1, n_passes):
    if shift_bits is not None:
      scale_p = (base_scale * (2.0 ** (-shift_bits * p))).astype(
          base_scale.dtype
      )
      q = qarray.quantize_with_scale_zero_point(
          current,
          how.qtype,
          scale=scale_p,
          zero_point=q0.zero_point,
          noise_fn=how.noise_fn,
      )
    else:
      q = qarray.quantize(current, how)
    passes.append(q)
    deq = qarray.dequantize(q)
    current = current - deq

  return tuple(passes)


def _prep_int8_parts(x: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Decomposes signed INT8 into high and low signed INT4 parts in [-8, 7]."""
  x = x.astype(jnp.int8)
  x_h = x >> 4
  x_l = (x & 0x0F) - 8
  return x_h, x_l


def _emulated_signed_int8_dot_general(
    a: jax.Array,
    b: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers = (((1,), (0,)), ((), ())),
    *,
    compute_dtype: jax.typing.DTypeLike,
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array:
  """Emulated signed int8 GEMM using 4 uncoupled passes."""
  (lhs_ca, rhs_ca), _ = dimension_numbers

  a_h, a_l = _prep_int8_parts(a)
  b_h, b_l = _prep_int8_parts(b)

  # Note: Despite converting to compute_type here, if native int4 is supported
  # it can be used for all four passes.
  a_h_c = a_h.astype(compute_dtype)
  a_l_c = a_l.astype(compute_dtype)
  b_h_c = b_h.astype(compute_dtype)
  b_l_c = b_l.astype(compute_dtype)

  def _dot(x: jax.Array, y: jax.Array) -> jax.Array:
    return jax.lax.dot_general(
        x,
        y,
        dimension_numbers=dimension_numbers,
        preferred_element_type=jnp.float32,
    ).astype(jnp.int32)

  p11 = _dot(a_h_c, b_h_c)
  p10 = _dot(a_h_c, b_l_c)
  p01 = _dot(a_l_c, b_h_c)
  p00 = _dot(a_l_c, b_l_c)

  xy_product = (p11 << 8) + ((p10 + p01) << 4) + p00

  k_size = 1
  for d in lhs_ca:
    k_size *= a.shape[d]

  lhs_transpose, rhs_transpose = dg._get_scale_transpose(  # pylint: disable=protected-access
      dimension_numbers, (a.ndim, b.ndim)
  )
  sum_a = qarray.transpose_array(
      jnp.sum(a, axis=lhs_ca, dtype=jnp.int32, keepdims=True), lhs_transpose
  )
  sum_b = qarray.transpose_array(
      jnp.sum(b, axis=rhs_ca, dtype=jnp.int32, keepdims=True), rhs_transpose
  )

  res = xy_product + (sum_a * 8) + (sum_b * 8) - (64 * k_size)
  if preferred_element_type is not None:
    return res.astype(preferred_element_type)
  return res


def _int8_multipass_dot_general(
    lhs: jax.Array,
    rhs: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers = (((1,), (0,)), ((), ())),
    compute_dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    tile_size: int | None = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array:
  """Executes scaled INT8 (channelwise or subchannel) GEMM via INT4 passes."""
  # Other options will be added.
  dot_fn = _emulated_signed_int8_dot_general

  lhs_how = dg.get_how_to_quantize(
      dimension_numbers=dimension_numbers,
      ndims=(lhs.ndim, rhs.ndim),
      for_lhs=True,
      tile_size=tile_size,
      qtype=jnp.int8,
  )
  q_lhs = qarray.quantize(lhs, lhs_how)

  rhs_how = dg.get_how_to_quantize(
      dimension_numbers=dimension_numbers,
      ndims=(lhs.ndim, rhs.ndim),
      for_lhs=False,
      tile_size=tile_size,
      qtype=jnp.int8,
  )
  q_rhs = qarray.quantize(rhs, rhs_how)

  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
  lhs_value = q_lhs.qvalue
  rhs_value = q_rhs.qvalue
  lhs_scale = q_lhs.scale
  rhs_scale = q_rhs.scale

  lhs_tiled_axes = qarray.get_tiled_axes(q_lhs)
  rhs_tiled_axes = qarray.get_tiled_axes(q_rhs)

  lhs_tiled_ca = {}
  rhs_tiled_ca = {}
  for l, r in zip(lhs_ca, rhs_ca):
    lhs_tile_size = lhs_tiled_axes.get(l)
    rhs_tile_size = rhs_tiled_axes.get(r)
    if lhs_tile_size and rhs_tile_size and lhs_tile_size != rhs_tile_size:
      raise ValueError(
          'Contracting axes must be tiled with the same tile size.'
          f' {lhs_tiled_axes=} {rhs_tiled_axes=} {dimension_numbers=}'
      )
    if lhs_tile_size or rhs_tile_size:
      lhs_tiled_ca[l] = lhs_tile_size or rhs_tile_size
      rhs_tiled_ca[r] = lhs_tile_size or rhs_tile_size

  lhs_value = qarray.split_axis(lhs_value, lhs_tiled_ca)
  rhs_value = qarray.split_axis(rhs_value, rhs_tiled_ca)
  # pylint: disable=protected-access
  lhs_ca, lhs_ba, sum_axes = dg._apply_tiling(lhs_ca, lhs_ba, lhs_tiled_ca)
  rhs_ca, rhs_ba, _ = dg._apply_tiling(rhs_ca, rhs_ba, rhs_tiled_ca)
  dimension_numbers = (lhs_ca, rhs_ca), (lhs_ba, rhs_ba)
  # Transpose lhs/rhs_scale for generic broadcasting.
  lhs_scale_transpose, rhs_scale_transpose = dg._get_scale_transpose(
      dimension_numbers, (len(lhs_value.shape), len(rhs_value.shape))
  )
  # pylint: enable=protected-access
  if lhs_scale is not None:
    lhs_scale = qarray.split_axis(lhs_scale, {a: 1 for a in lhs_tiled_ca})
    lhs_scale = qarray.transpose_array(lhs_scale, lhs_scale_transpose)
  if rhs_scale is not None:
    rhs_scale = qarray.split_axis(rhs_scale, {a: 1 for a in rhs_tiled_ca})
    rhs_scale = qarray.transpose_array(rhs_scale, rhs_scale_transpose)
  # Single batched GEMM across all tiles.
  res = dot_fn(
      lhs_value,
      rhs_value,
      dimension_numbers=dimension_numbers,
      compute_dtype=compute_dtype,
      preferred_element_type=jnp.float32,
  )
  if lhs_scale is not None:
    res = qarray.call_with_generic_broadcast(jnp.multiply, res, lhs_scale)
  if rhs_scale is not None:
    res = qarray.call_with_generic_broadcast(jnp.multiply, res, rhs_scale)
  if sum_axes:
    res = jnp.sum(res, axis=sum_axes)
  _, result_type = qarray.get_accumulator_and_result_type(
      q_lhs, q_rhs, preferred_element_type=preferred_element_type
  )
  return res.astype(result_type)


def _fp8_multipass_dot_general(
    lhs: jax.Array,
    rhs: jax.Array,
    multipass_mode: str,
    dimension_numbers: jax.lax.DotDimensionNumbers = (((1,), (0,)), ((), ())),
    precision: jax.lax.PrecisionLike = None,
    tile_size: int | None = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array:
  """Multipass functions for fp8."""

  lhs_how = dg.get_how_to_quantize(
      dimension_numbers=dimension_numbers,
      ndims=(lhs.ndim, rhs.ndim),
      for_lhs=True,
      qtype=jnp.float8_e4m3fn,
      tile_size=tile_size,
  )
  rhs_how = dg.get_how_to_quantize(
      dimension_numbers=dimension_numbers,
      ndims=(lhs.ndim, rhs.ndim),
      for_lhs=False,
      qtype=jnp.float8_e4m3fn,
      tile_size=tile_size,
  )

  def _dot(a: qarray.QArray, b: qarray.QArray) -> jax.Array:
    return dg.dot_general(
        a,
        b,
        dimension_numbers=dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )

  if multipass_mode == 'two_pass_lhs_fp8':
    a_passes = _residual_decompose(lhs, lhs_how, n_passes=2)
    a0, a1 = a_passes[0], a_passes[1]
    b0 = qarray.quantize(rhs, rhs_how)
    c0 = _dot(a0, b0)
    c1 = _dot(a1, b0)
    return c0 + c1

  elif multipass_mode == 'two_pass_rhs_fp8':
    a0 = qarray.quantize(lhs, lhs_how)
    b_passes = _residual_decompose(rhs, rhs_how, n_passes=2)
    b0, b1 = b_passes[0], b_passes[1]
    c0 = _dot(a0, b0)
    c1 = _dot(a0, b1)
    return c0 + c1

  elif multipass_mode in ('three_pass_fp8', 'four_pass_fp8'):
    a_passes = _residual_decompose(lhs, lhs_how, n_passes=2)
    a0, a1 = a_passes[0], a_passes[1]
    b_passes = _residual_decompose(rhs, rhs_how, n_passes=2)
    b0, b1 = b_passes[0], b_passes[1]
    c00 = _dot(a0, b0)
    c01 = _dot(a0, b1)
    c10 = _dot(a1, b0)
    res = c00 + c01 + c10
    if multipass_mode == 'four_pass_fp8':
      c11 = _dot(a1, b1)
      res = res + c11
    return res
  else:
    raise ValueError(
        "_fp8_multipass_dot_general requires mode 'two_pass_lhs_fp8',"
        " 'two_pass_rhs_fp8', 'three_pass_fp8', or 'four_pass_fp8'"
    )


def multipass_dot_general(
    lhs: jax.Array,
    rhs: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers = (((1,), (0,)), ((), ())),
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    **kwargs: Any,
) -> jax.Array:
  """Executes multi-pass emulated matrix multiplication.

  Inputs must strictly be unquantized jax.Array to allow residual calculation.

  Args:
    lhs: Left-hand side unquantized array.
    rhs: Right-hand side unquantized array.
    dimension_numbers: Standard JAX dot_general dimension specification.
    precision: The precision for dot_general.
    preferred_element_type: Output/accumulator dtype.
    **kwargs: Additional arguments forwarded to dot_general, including optional
      'multipass_mode', 'tile_size'.

  Returns:
    The resulting matrix product array.
  """
  if isinstance(lhs, qarray.QArray) or isinstance(rhs, qarray.QArray):
    raise ValueError(
        'Inputs to multipass_dot_general must strictly be unquantized'
        f' jax.Array, but got lhs={type(lhs)}, rhs={type(rhs)}.'
    )

  kwargs = dict(kwargs)
  multipass_mode: MultiPassMode = kwargs.pop('multipass_mode', None)
  if kwargs.get('lhs_qtype') is not None or kwargs.get('rhs_qtype') is not None:
    raise ValueError(
        "When 'multipass_mode' is specified, 'lhs_qtype' and 'rhs_qtype' must"
        ' not be specified to avoid ambiguity. The precision is determined by'
        f" the mode '{multipass_mode}'."
    )
  if kwargs.get('lhs_how') is not None or kwargs.get('rhs_how') is not None:
    raise ValueError(
        "When 'multipass_mode' is specified, 'lhs_how' and 'rhs_how' must not"
        ' be specified. The quantization configuration is determined'
        f" automatically by the mode '{multipass_mode}'."
    )
  tile_size = kwargs.pop('tile_size', None)

  if multipass_mode in ('four_pass_int4',):
    return _int8_multipass_dot_general(
        lhs,
        rhs,
        dimension_numbers=dimension_numbers,
        compute_dtype=jnp.float8_e4m3fn,
        tile_size=tile_size,
        preferred_element_type=preferred_element_type,
    )
  elif multipass_mode in (
      'three_pass_fp8',
      'four_pass_fp8',
      'two_pass_lhs_fp8',
      'two_pass_rhs_fp8',
  ):
    return _fp8_multipass_dot_general(
        lhs,
        rhs,
        multipass_mode=multipass_mode,
        dimension_numbers=dimension_numbers,
        precision=precision,
        tile_size=tile_size,
        preferred_element_type=preferred_element_type,
    )
  else:
    raise ValueError(
        f'Unknown multipass mode: {multipass_mode!r}. Expected one of:'
        f' {list(get_args(MultiPassMode))}'
    )
