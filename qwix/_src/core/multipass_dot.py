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

- Hybrid types of multi-pass:
For hybrid multi-pass, Pass 1 A0 B0 is evaluated in FP8, while 4-bit formats are
used for cross-pass terms A0 B1 and A1 B0.
Supported hybrid modes:
    1. 'three_pass_fp8_fp4/fp4_fp4/fp4': FP4 cross passes.
    2. 'three_pass_fp8_int4/int4_int4/int4': INT4 cross passes.
    3. 'three_pass_fp8_int4/fp4_fp4/int4': Mixed 4-bit cross passes
    (coarse INT4, residual FP4).
    4. 'three_pass_fp8_fp4/int4_int4/fp4': Mixed 4-bit cross passes
    (coarse FP4, residual INT4).
Additionally, microscaled variants are supported:
    5. 'three_pass_mxfp8_mxfp4/mxfp4_mxfp4/mxfp4'
    6. 'three_pass_mxfp8_mxint4/mxint4_mxint4/mxint4'
    7. 'three_pass_mxfp8_mxint4/mxfp4_mxfp4/mxint4'
    8. 'three_pass_mxfp8_mxfp4/mxint4_mxint4/mxfp4'

Additionally it supports int8 by decomposing into int4. This could be useful on
devices with high int4 FLOPs. For the purposes of emulation we do this on the
fp8 native path. We support the following int8 emulation modes:
    1. 'four_pass_int4': full emulation for int8 x int8 matmul
    2. 'three_pass_int4': Truncated emulation, dropping the low-order cross
       nibble (a_l * b_l)
    3. 'two_pass_lhs_int4': Two passes for the LHS (emulating int8 x int4)
    4. 'two_pass_rhs_int4': Two passes for the RHS (emulating int4 x int8)
    5. 'four_pass_zero_offset_int4': 4-pass zero-offset int4 (radix-16 split
       into 2 signed nibbles in [-8, 7] without +8 offset, evaluating a0b0,
       a0b1, a1b0, a1b1). Use this (+/-119 range with no offset terms) instead
       of 'four_pass_int4' (+/-127 range with +8 offset correction terms) to
       avoid extra row/column sum reductions at a small dynamic-range cost.
    6. 'six_pass_zero_offset_int4': 6-pass zero-offset int4 (radix-16 split into
       3 signed nibbles in [-8, 7] without +8 offset, evaluating a0b0, a0b1,
       a1b0, a1b1, a2b0, a0b2)
    7. 'eight_pass_zero_offset_int4': 8-pass zero-offset int4 (same 3 signed
       nibbles in [-8, 7], evaluating a0b0, a0b1, a1b0, a1b1, a2b0, a0b2, a1b2,
       a2b1)

All decomposed passes are evaluated on quantized operands directly through the
hardware FP8 path (via `_fast_dot_general` with hardware FP8 accumulation) or
via `jax.lax.scaled_dot` on supported GPUs.

Scaling factor design choice for residual and downcast passes:
In pure FP8 multi-pass (`three_pass_fp8`, `four_pass_fp8`) and coarse 4-bit
downcasting (`_downcast_by_shift`), scales are derived directly by shifting the
initial pass's scale factor by a fixed power-of-2 exponent offset (e.g. 2^-4
for FP8 E4M3 residuals, 32 for coarse INT4, 64 for coarse FP4) without a second
reduction tree. In the high-level JAX reference implementation of hybrid FP8 +
4-bit modes (`_hybrid_fp8_4bit_dot_general`), `qarray.quantize` calibrates `r_a`
and `r_b` with `calibration_method = f'absmax,{7.5 / 8.5}'` for INT4/MXINT4
residuals: for power-of-2 `mxint4` block scales, `2 ** ceil(log2(absmax(r) /
8.5))` snaps directly to the exact power-of-2 sub-step `s_res = Delta / 16`
(matching a fixed exponent shift `s0 * 2^-4` or `s0 * 2^-3`), whereas for
floating-point `int4` tile scales `scale = absmax(r) / 8.5` equals `s_res` when
the tile attains the worst-case residual bound `8.5 * s_res`.
This power-of-2 alignment is chosen because:

1. Hardware efficiency: In low-level hardware kernels, fixed power-of-2 scale
   relationships allow deriving scales via an integer exponent shift
   (E_1 = E_0 - shift_bits) without a dynamic reduction tree.
2. Memory bandwidth and storage: An implicit power-of-2 shifted scale avoids
   storing and loading a separate scale tensor from memory.
3. Fixed accumulator alignment: Constant power-of-2 scale offsets allow matrix
   units to align cross-term products with arithmetic bit-shifts prior to
   register accumulation.
4. Numerical closeness: Because the maximum rounding residual of Pass 1 is
   bounded by the top binade step size (`Delta = 16 * s_res`), aligning the
   16 INT4 residual bins `[-8, +7]` to `[-8.5 * s_res, +7.5 * s_res)` matches
   the exact sub-bin grid.
"""

import functools
import math
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
    'three_pass_int4',
    'two_pass_lhs_int4',
    'two_pass_rhs_int4',
    'four_pass_zero_offset_int4',
    'six_pass_zero_offset_int4',
    'eight_pass_zero_offset_int4',
    # Hybrid FP8 + 4-bit modes:
    'three_pass_fp8_fp4/fp4_fp4/fp4',
    'three_pass_fp8_int4/int4_int4/int4',
    'three_pass_fp8_int4/fp4_fp4/int4',
    'three_pass_fp8_fp4/int4_int4/fp4',
    # Microscaled hybrid modes:
    'three_pass_mxfp8_mxfp4/mxfp4_mxfp4/mxfp4',
    'three_pass_mxfp8_mxint4/mxint4_mxint4/mxint4',
    'three_pass_mxfp8_mxint4/mxfp4_mxfp4/mxint4',
    'three_pass_mxfp8_mxfp4/mxint4_mxint4/mxfp4',
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


def _prep_zero_offset_int4_parts(
    x: jax.Array, n_parts: int = 3
) -> tuple[jax.Array, ...]:
  """Decomposes integers into 2 or 3 zero-offset INT4 parts in [-8, 7]."""
  if n_parts == 2:
    # Max positive value for 2 signed INT4 parts in [-8, 7]:
    # 16 * 7 + 7 = 119.
    x = jnp.clip(x.astype(jnp.int32), -119, 119)
    x_1 = ((x + 8) & 0x0F) - 8
    x_0 = (x + 8) >> 4
    return x_0.astype(jnp.int8), x_1.astype(jnp.int8)
  if n_parts == 3:
    # Max positive value for 3 signed INT4 parts in [-8, 7]:
    # 256 * 7 + 16 * 7 + 7 = 1911.
    x = jnp.clip(x.astype(jnp.int32), -1911, 1911)
    x_2 = ((x + 8) & 0x0F) - 8
    x_p = (x + 8) >> 4
    x_1 = ((x_p + 8) & 0x0F) - 8
    x_0 = (x_p + 8) >> 4
    return x_0.astype(jnp.int8), x_1.astype(jnp.int8), x_2.astype(jnp.int8)
  raise ValueError(f'Unsupported n_parts: {n_parts!r}. Expected 2 or 3.')


def _emulated_signed_int8_dot_general(
    a: jax.Array,
    b: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers = (((1,), (0,)), ((), ())),
    *,
    multipass_mode: str,
    compute_dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array:
  """Emulated signed int8/int4 GEMM using INT4 passes."""
  (lhs_ca, rhs_ca), _ = dimension_numbers

  # Accumulate and combine passes in float32 rather than int32: in the worst
  # case (constant or correlated inputs with qmax = 1911), the 8-pass integer
  # sum reaches 1911^2 * K / 16 ~= 228k * K, which overflows int32 for K > ~9.4k
  # (and 4-pass / 6-pass overflow int32 for K > ~133k). The extra float32
  # rounding (~2^-24) is negligible compared to quantization noise.
  def _dot(x: jax.Array, y: jax.Array) -> jax.Array:
    return jax.lax.dot_general(
        x.astype(compute_dtype),
        y.astype(compute_dtype),
        dimension_numbers=dimension_numbers,
        preferred_element_type=jnp.float32,
    )

  lhs_transpose, rhs_transpose = dg._get_scale_transpose(  # pylint: disable=protected-access
      dimension_numbers, (a.ndim, b.ndim)
  )

  match multipass_mode:
    case 'four_pass_int4' | 'three_pass_int4':
      a_h, a_l = _prep_int8_parts(a)
      b_h, b_l = _prep_int8_parts(b)
      p11 = _dot(a_h, b_h)
      p10 = _dot(a_h, b_l)
      p01 = _dot(a_l, b_h)
      xy_product = p11 * 256.0 + (p10 + p01) * 16.0
      if multipass_mode == 'four_pass_int4':
        p00 = _dot(a_l, b_l)
        xy_product = xy_product + p00

      k_size = 1
      for d in lhs_ca:
        k_size *= a.shape[d]

      sum_a = qarray.transpose_array(
          jnp.sum(a, axis=lhs_ca, dtype=jnp.float32, keepdims=True),
          lhs_transpose,
      )
      sum_b = qarray.transpose_array(
          jnp.sum(b, axis=rhs_ca, dtype=jnp.float32, keepdims=True),
          rhs_transpose,
      )
      res = xy_product + (sum_a * 8.0) + (sum_b * 8.0) - (64.0 * k_size)

    case 'two_pass_lhs_int4':
      a_h, a_l = _prep_int8_parts(a)
      p1 = _dot(a_h, b)
      p0 = _dot(a_l, b)
      xy_product = p1 * 16.0 + p0

      sum_b = qarray.transpose_array(
          jnp.sum(b, axis=rhs_ca, dtype=jnp.float32, keepdims=True),
          rhs_transpose,
      )
      res = xy_product + (sum_b * 8.0)

    case 'two_pass_rhs_int4':
      b_h, b_l = _prep_int8_parts(b)
      p1 = _dot(a, b_h)
      p0 = _dot(a, b_l)
      xy_product = p1 * 16.0 + p0

      sum_a = qarray.transpose_array(
          jnp.sum(a, axis=lhs_ca, dtype=jnp.float32, keepdims=True),
          lhs_transpose,
      )
      res = xy_product + (sum_a * 8.0)

    case 'four_pass_zero_offset_int4':
      a0, a1 = _prep_zero_offset_int4_parts(a, n_parts=2)
      b0, b1 = _prep_zero_offset_int4_parts(b, n_parts=2)
      p00 = _dot(a0, b0)
      p01 = _dot(a0, b1)
      p10 = _dot(a1, b0)
      p11 = _dot(a1, b1)
      res = p00 * 256.0 + (p01 + p10) * 16.0 + p11

    case 'six_pass_zero_offset_int4' | 'eight_pass_zero_offset_int4':
      a0, a1, a2 = _prep_zero_offset_int4_parts(a, n_parts=3)
      b0, b1, b2 = _prep_zero_offset_int4_parts(b, n_parts=3)
      p00 = _dot(a0, b0)
      p01 = _dot(a0, b1)
      p10 = _dot(a1, b0)
      p11 = _dot(a1, b1)
      p20 = _dot(a2, b0)
      p02 = _dot(a0, b2)
      res = p00 * 256.0 + (p01 + p10) * 16.0 + (p11 + p20 + p02)
      if multipass_mode == 'eight_pass_zero_offset_int4':
        p12 = _dot(a1, b2)
        p21 = _dot(a2, b1)
        res = res * 16.0 + (p12 + p21)

    case _:
      raise ValueError(f'Unsupported multipass_mode: {multipass_mode!r}')

  if preferred_element_type is not None:
    return res.astype(preferred_element_type)
  return res.astype(jnp.int32)


def _int8_multipass_dot_general(
    lhs: jax.Array,
    rhs: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers = (((1,), (0,)), ((), ())),
    *,
    multipass_mode: str = 'four_pass_int4',
    compute_dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    tile_size: int | None = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array:
  """Executes scaled integer (channelwise or subchannel) GEMM via INT4 passes."""
  calib = 'absmax'
  scale_mult = 1.0
  quantize_in_f32 = False
  match multipass_mode:
    case 'four_pass_int4' | 'three_pass_int4':
      lhs_qtype = jnp.int8
      rhs_qtype = jnp.int8
    case 'two_pass_lhs_int4':
      lhs_qtype = jnp.int8
      rhs_qtype = jnp.int4
    case 'two_pass_rhs_int4':
      lhs_qtype = jnp.int4
      rhs_qtype = jnp.int8
    case 'four_pass_zero_offset_int4':
      lhs_qtype = jnp.int8
      rhs_qtype = jnp.int8
      # Override int8 bound (127.5) so scale = max|x| / 119.0
      # (where 119 = 16 * 7 + 7 for 2 INT4 parts).
      calib = f'absmax,{127.5 / 119.0}'
      quantize_in_f32 = True
    case 'six_pass_zero_offset_int4':
      lhs_qtype = jnp.int16
      rhs_qtype = jnp.int16
      # Override int16 bound (32768.0) so scale = max|x| / 1911.0
      # (where 1911 = 256 * 7 + 16 * 7 + 7 for 3 INT4 parts).
      calib = f'absmax,{32768.0 / 1911.0}'
      # Order-2 terms have weight 256 = 16^2; absorb 16x into each scale.
      scale_mult = 16.0
      quantize_in_f32 = True
    case 'eight_pass_zero_offset_int4':
      lhs_qtype = jnp.int16
      rhs_qtype = jnp.int16
      calib = f'absmax,{32768.0 / 1911.0}'
      # Order-3 terms have weight 16 = 4^2; absorb 4x into each scale.
      scale_mult = 4.0
      quantize_in_f32 = True
    case _:
      raise ValueError(f'Unknown multipass_mode: {multipass_mode!r}')

  dot_fn = functools.partial(
      _emulated_signed_int8_dot_general,
      multipass_mode=multipass_mode,
  )

  lhs_how = dg.get_how_to_quantize(
      dimension_numbers=dimension_numbers,
      ndims=(lhs.ndim, rhs.ndim),
      for_lhs=True,
      tile_size=tile_size,
      qtype=lhs_qtype,
      calibration_method=calib,
  )
  rhs_how = dg.get_how_to_quantize(
      dimension_numbers=dimension_numbers,
      ndims=(lhs.ndim, rhs.ndim),
      for_lhs=False,
      tile_size=tile_size,
      qtype=rhs_qtype,
      calibration_method=calib,
  )
  if quantize_in_f32:
    q_lhs = qarray.quantize(lhs.astype(jnp.float32), lhs_how)
    q_rhs = qarray.quantize(rhs.astype(jnp.float32), rhs_how)
  else:
    q_lhs = qarray.quantize(lhs, lhs_how)
    q_rhs = qarray.quantize(rhs, rhs_how)
  lhs_scale = q_lhs.scale * scale_mult
  rhs_scale = q_rhs.scale * scale_mult

  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
  lhs_value = q_lhs.qvalue
  rhs_value = q_rhs.qvalue

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
      lhs, rhs, preferred_element_type=preferred_element_type
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


def _downcast_by_shift(
    operand: qarray.QArray,
    target_qtype: jax.typing.DTypeLike,
) -> qarray.QArray:
  """Downcasts an FP8 QArray into 4-bit (FP4 or INT4) via scale-shifting.

  This helper converts FP8 first-pass operands (A0, B0) into 4-bit formats
  for the secondary cross passes (A0 B1 and A1 B0) by reusing the in-register
  quantized operand from Pass 1, rather than reloading and re-quantizing the
  original floating-point input tensor.

  Key architectural considerations:
    1. In-Register Reuse: Avoids reloading the unquantized floating-point
       tensor from memory and eliminates redundant memory traffic.
    2. Skipping Abs-Max Reductions: Recomputing an abs-max reduction pass can
       be skipped when the block size of FP8 is less than or equal to that of
       the 4-bit format (e.g., block 16 FP8 vs block 32 for 4-bit).
    3. Fixed Power-of-2 Scaling: Shifting by a fixed power-of-2 factor (64 for
       FP4, 32 for INT4) preserves residual error cancellation and is
       substantially cheaper in lower-level hardware implementations (e.g.,
       executable via an in-register exponent subtraction / vector multiply for
       FP4, or vector integer right shift `vshra.i8` for INT4, rather than
       full float division and comparison trees).
    4. Implementation Note: In this high-level JAX reference implementation,
       `.astype(jnp.float4_e2m1fn)` is called to satisfy JAX's type system, as
       JAX lacks a direct in-register cast to `float4`. In a lower-level or
       native hardware kernel (e.g., Pallas/Mosaic), this mapping would be
       realized via direct in-register exponent or bit manipulation without
       floating-point comparison trees.

  Args:
    operand: The input FP8 QArray to downcast.
    target_qtype: Target 4-bit quantization type ('fp4'/'mxfp4' or
      'int4'/'mxint4').

  Returns:
    A QArray with the downcast 4-bit values and scaled factor.
  """
  if target_qtype in (jnp.float4_e2m1fn, 'fp4', 'mxfp4'):
    shift = 64.0
    scaled_val = operand.qvalue.astype(jnp.float32) / shift
    new_qval = jnp.clip(scaled_val, -6.0, 6.0).astype(jnp.float4_e2m1fn)
    actual_qtype = jnp.float4_e2m1fn
  elif target_qtype in (jnp.int4, 'int4', 'mxint4'):
    shift = 32.0
    scaled_val = operand.qvalue.astype(jnp.float32) / shift
    # Use standard round-to-even (jnp.round) clipped to two's-complement INT4
    # [-8.0, 7.0]:
    # - Enabling -8.0 (instead of symmetric [-7.0, 7.0]) preserves negative
    #   top-binade anchors (-256 -> -8, -240 -> -8) with zero net anchor bias
    #   across [-256, -128].
    # - Because a0_downcast and b0_downcast only multiply the zero-mean
    #   residuals b1 and a1 (c01 = a0_downcast @ b1, c10 = a1 @ b0_downcast),
    #   any positive top-binade clipping at +240, +256 -> +7 is attenuated by
    #   E[b1] = 0, so standard round-to-even avoids 3 extra VPU instructions
    #   (compare, add, select) and achieves +0.06 to +0.10 dB higher Gaussian
    #   SQNR than upward tie-breaking on [4.0, 8.0].
    new_qval = jnp.clip(jnp.round(scaled_val), -8.0, 7.0).astype(jnp.int4)
    actual_qtype = jnp.int4
  else:
    raise ValueError(
        '_downcast_by_shift requires a 4-bit target type but got'
        f' {target_qtype}.'
    )
  return qarray.QArray(
      qvalue=new_qval, scale=operand.scale * shift, qtype=actual_qtype
  )


def _get_fp8_shift_factor(
    qtype: jax.typing.DTypeLike, target_dtype: jax.typing.DTypeLike
) -> float:
  """Returns the maximum power-of-2 shift factor that fits without overflow."""
  if qtype in (jnp.float8_e4m3fn, 'fp8', 'mxfp8', 'mxfp8_16'):
    max_qval = 448.0
  elif qtype in (jnp.float4_e2m1fn, 'fp4', 'mxfp4'):
    max_qval = 6.0
  elif qtype in (jnp.int4, 'int4', 'mxint4'):
    max_qval = 8.0
  else:
    raise ValueError(
        f'_get_fp8_shift_factor requires a supported qtype but got {qtype}.'
    )

  target_max = float(jnp.finfo(target_dtype).max)
  k = int(math.floor(math.log2(target_max / max_qval)))
  return 2.0**k


def _block_shift_to_fp8(
    operand: qarray.QArray,
    target_dtype: jax.typing.DTypeLike | None = None,
    dimension_numbers: jax.lax.DotDimensionNumbers = (((1,), (0,)), ((), ())),
    for_lhs: bool = True,
    tile_size: int | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Absorbs block scales into FP8 exponents.

  When tile_size is specified and smaller than the contracting dimension, fine
  blocks are coalesced into coarse tiles of size tile_size using the largest
  scaling factor within each coarse tile, allowing sub-blocks with smaller
  scales to underflow into the FP8 representation. When tile_size is None,
  blocks along the contracting axes are coalesced into channel-wise scales.

  Args:
    operand: The input QArray whose block scales are to be absorbed.
    target_dtype: The target floating-point dtype (defaults to float8_e4m3fn for
      FP8 inputs and float8_e5m2 for 4-bit inputs).
    dimension_numbers: Standard JAX dot_general dimension numbers.
    for_lhs: Whether the array is the LHS or RHS operand.
    tile_size: Optional coarse tile size along the contracting dimension.

  Returns:
    A tuple of (rel_scaled_val, fp8_block_scale).
  """
  if target_dtype is None:
    if operand.qtype in (jnp.float8_e4m3fn, 'fp8', 'mxfp8', 'mxfp8_16'):
      target_dtype = jnp.float8_e4m3fn
    elif operand.qtype in (
        jnp.float4_e2m1fn,
        'fp4',
        'mxfp4',
        jnp.int4,
        'int4',
        'mxint4',
    ):
      target_dtype = jnp.float8_e5m2
    else:
      raise ValueError(
          f'Unsupported qtype for _block_shift_to_fp8: {operand.qtype}'
      )
  (lhs_ca, rhs_ca), _ = dimension_numbers
  ca_axes = lhs_ca if for_lhs else rhs_ca
  shift_factor = _get_fp8_shift_factor(operand.qtype, target_dtype)

  ca = ca_axes[-1]
  k_dim = operand.qvalue.shape[ca]
  s_dim = operand.scale.shape[ca]

  if tile_size is None or s_dim <= 1 or tile_size >= k_dim:
    max_scale = jnp.max(operand.scale, axis=ca_axes, keepdims=True)
    max_scale = jnp.where(max_scale == 0, 1.0, max_scale)
    fp8_block_scale = max_scale / shift_factor
    rel_scale = operand.scale / fp8_block_scale
  else:
    fine_tile_size = k_dim // s_dim
    if tile_size < fine_tile_size:
      raise ValueError(
          f'Coarse tile_size ({tile_size}) cannot be smaller than the fine'
          f' block size ({fine_tile_size}).'
      )
    if tile_size % fine_tile_size != 0:
      raise ValueError(
          f'Coarse tile_size ({tile_size}) must be a multiple of the fine'
          f' block size ({fine_tile_size}).'
      )
    group_size = tile_size // fine_tile_size
    n_coarse = k_dim // tile_size

    scale = operand.scale
    if len(ca_axes) > 1:
      other_ca = tuple(a for a in ca_axes if a != ca)
      scale = jnp.max(scale, axis=other_ca, keepdims=True)

    scale_shape = list(scale.shape)
    grouped_shape = (
        scale_shape[:ca] + [n_coarse, group_size] + scale_shape[ca + 1 :]
    )
    scale_grouped = scale.reshape(grouped_shape)
    max_grouped = jnp.max(scale_grouped, axis=ca + 1, keepdims=True)
    max_grouped = jnp.where(max_grouped == 0, 1.0, max_grouped)

    coarse_shape = scale_shape[:ca] + [n_coarse] + scale_shape[ca + 1 :]
    fp8_coarse_scale = max_grouped.reshape(coarse_shape)
    fp8_block_scale = fp8_coarse_scale / shift_factor

    rel_grouped = (scale_grouped / max_grouped) * shift_factor
    rel_scale = rel_grouped.reshape(scale.shape)

  rel_scaled_val = qarray.call_with_generic_broadcast(
      jnp.multiply, operand.qvalue.astype(jnp.float32), rel_scale
  )
  return rel_scaled_val.astype(target_dtype), fp8_block_scale


def _quantize_fp8_for_int4_residual(
    x: jax.Array, how_fp8: qarray.HowToQuantize
) -> qarray.QArray:
  """Quantizes x to FP8 with a half-step boundary offset for INT4 residuals.

  In aligned FP8 + INT4 multi-pass quantization, each FP8 step
  Delta = 16 * s_res is subdivided into 16 asymmetric two's-complement INT4
  bins [-8, +7], which span [-8.5 * s_res, +7.5 * s_res] around each FP8
  anchor. Standard symmetric Pass 1 FP8 rounding places the handoff midpoint
  at a_L + 8.0 * s_res, causing values in (a_L + 7.5 * s_res, a_L + 8.0 * s_res)
  to round down to a_L and saturate at bin +7 (9.375% occupancy) while starving
  bin -8 of a_R (3.125% occupancy) and inducing a systematic -1/32 * s_res
  negative bias.

  Offsetting x by +0.5 * s_res in the 16-step top binade (|x| >= 128 * s_res,
  where Delta = 16 * s_res) before Pass 1 FP8 rounding shifts the midpoint
  boundary to a_L + 7.5 * s_res. When the residual r = x - dequantize(a0) is
  subsequently computed against the un-shifted x, residuals map uniformly
  across [-8.5 * s_res, +7.5 * s_res), equalizing all 16 INT4 bins at 6.25%
  and eliminating the quantization bias while preserving exact zero (0.0) and
  all base FP8 anchors. In lower binades (|x| < 128 * s_res, Delta <= 8 *
  s_res), residuals lie within [-4 * s_res, +4 * s_res] where both negative and
  positive endpoint codes (-4 and +4) exist symmetrically inside [-8, +7]
  without clipping, so no offset is needed.

  Args:
    x: Input unquantized array.
    how_fp8: Pass 1 FP8 quantization configuration.

  Returns:
    Quantized Pass 1 QArray with boundary-adjusted FP8 anchors.
  """
  calib = qarray.calibrate(x, how_fp8)
  s0, zp0 = qarray.compute_scale_zero_point(calib, how_fp8.qtype)
  # Determine the top-binade residual step size s_res = Delta / 16 from the
  # block's actual unscaled absmax relative to the 256 * s0 binade boundary
  # (accounting for mxfp8's 56/29 Outlier-Aware Scaling in
  # compute_scale_zero_point, where s0 can already be 1.0 on [128, 256] even
  # when calibration_method == 'absmax').
  _, *calib_args = how_fp8.calibration_method.lower().split(',')
  raw_absmax = (
      calib['absmax'].astype(jnp.float32) / float(calib_args[0])
      if calib_args
      else calib['absmax'].astype(jnp.float32)
  )
  s0_f32 = s0.astype(jnp.float32)
  s_res = jnp.where(raw_absmax > 256.0 * s0_f32, 2.0 * s0_f32, s0_f32)

  def _apply_offset(arr: jax.Array, sr: jax.Array) -> jax.Array:
    # Compute the +0.5 * sr boundary shift in float32 so bfloat16 inputs (whose
    # ULP in [128 * sr, 256 * sr) is 1.0 * sr) do not round 0.5 * sr to 0 or
    # +1.0 * sr before Pass 1 FP8 quantization.
    arr_f32 = arr.astype(jnp.float32)
    return jnp.where(
        jnp.abs(arr_f32) >= 128.0 * sr, arr_f32 + 0.5 * sr, arr_f32
    )

  x_adj = qarray.call_with_generic_broadcast(_apply_offset, x, s_res)
  q0 = qarray.quantize_with_scale_zero_point(x_adj, how_fp8.qtype, s0_f32, zp0)
  return qarray.QArray(
      qvalue=q0.qvalue,
      scale=s0.astype(x.dtype),
      zero_point=q0.zero_point,
      qtype=q0.qtype,
  )


def _hybrid_fp8_4bit_dot_general(
    lhs: jax.Array,
    rhs: jax.Array,
    multipass_mode: str,
    dimension_numbers: jax.lax.DotDimensionNumbers = (((1,), (0,)), ((), ())),
    compute_dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    tile_size: int | None = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    precision: jax.lax.PrecisionLike = None,
) -> jax.Array:
  """Hybrid 3-pass GEMM: FP8 for Pass 1, 4-bit for cross passes.

  Calibration and cross-pass precision selection:
  1. FP8 + FP4 / MXFP8 + MXFP4 ('three_pass_fp8_fp4/fp4_fp4/fp4',
  'three_pass_mxfp8_mxfp4/mxfp4_mxfp4/mxfp4'):
     - Pass 1 uses native absmax (cutoff 448.0). Downcasting by 64 maps
       [0, 448.0] to [0, 7.0], which matches the optimal Outlier-Aware
       Scaling (OAS) bound of 7.0 for FP4 (max 6.0).
     - Cross passes use FP4 / MXFP4 residuals and coarse operands.
     - Hardware execution: float8_e5m2 losslessly accommodates FP4 (with 28
       octaves of dynamic range for exponent shifting in _block_shift_to_fp8).
  2. FP8 + INT4 / MXFP8 + MXINT4 ('three_pass_fp8_int4/int4_int4/int4',
  'three_pass_mxfp8_mxint4/mxint4_mxint4/mxint4'):
     - Pass 1 uses cutoff 256.0 (absmax, 448/256) with boundary-offset FP8
       rounding (_quantize_fp8_for_int4_residual). Downcasting by 32 maps
       [-256.0, +256.0] to [-8.0, +8.0], matching the two's-complement INT4
       codebook [-8, 7].
     - Cross passes use INT4 / MXINT4 residuals calibrated with absmax, 7.5/8.5.
     - Hardware execution: float8_e5m2 represents integers [-8, 7] bit-
       exactly (<=2 mantissa bits), while 5 exponent bits provide wide
       dynamic range to absorb block scale shifts in _block_shift_to_fp8.
  3. FP8 + mixed 4-bit / MXFP8 + MXMixed4 ('three_pass_fp8_int4/fp4_fp4/int4',
  'three_pass_mxfp8_mxint4/mxfp4_mxfp4/mxint4',
  'three_pass_fp8_fp4/int4_int4/fp4',
  'three_pass_mxfp8_mxfp4/mxint4_mxint4/mxfp4'):
     - Downcast coarse operands and residuals independently select FP4/MXFP4
       or INT4/MXINT4.
     - Hardware execution: float8_e5m2 x float8_e5m2 on native hardware
       matrix units.

  Execution path and emulation notes:
  - Step 1 (Pass 1 Primary Quantization):
    In microscaled hybrid modes ('three_pass_mxfp8_*'), inputs are quantized
    into mxfp8 with block size 32 (establishing fine-grained microscales).
    In standard hybrid modes ('three_pass_fp8_*'), inputs are quantized into
    regular FP8 (float8_e4m3fn) with the specified tile_size (channel-wise by
    default when tile_size is None).
  - Step 2 (Pass 1 GEMM):
    In microscaled modes, _dot routes operands through _block_shift_to_fp8,
    coalescing block-32 scales into coarse tiles of tile_size (or channel-wise
    if tile_size is None) using the largest scaling factor within each coarse
    tile and absorbing them into float8_e4m3fn (allowing underflow in sub-blocks
    with smaller scales). The matrix multiplication executes directly on
    physical FP8.
  - Step 3 (Pass 2 & 3 Downcast and Residual Quantization):
    Coarse operands are downcast via _downcast_by_shift by simple power-of-2
    multiplication (scale * shift) for each block size 32. This reuses the first
    pass's block scales and eliminates an expensive second reduction pass across
    the tensor. Residuals are quantized using residual_qtype (mxfp4/mxint4 with
    block size 32 for microscaled modes; float4_e2m1fn/int4 with the specified
    tile_size for standard hybrid).
  - Step 4 (Cross-Pass GEMMs):
    In microscaled modes, _block_shift_to_fp8 absorbs the 4-bit block-32 scales
    into float8_e5m2 exponents (with 28 octaves of dynamic range). Additionally,
    blocks are merged to tile_size (or one per channel when tile_size is None).
    GEMMs execute directly on physical hardware FP8 matrix units.

  Args:
    lhs: Left-hand side unquantized array.
    rhs: Right-hand side unquantized array.
    multipass_mode: Hybrid multi-pass execution mode.
    dimension_numbers: Standard JAX dot_general dimension specification.
    compute_dtype: Accumulator/compute data type for matrix multiplication.
    tile_size: Optional block/tile size along contracting dimension.
    preferred_element_type: Preferred output element type.
    precision: Standard JAX precision specification.

  Returns:
    The resulting matrix product array.
  """
  is_microscale = multipass_mode.startswith('three_pass_mxfp8_')
  target_out_type = (
      preferred_element_type
      if preferred_element_type is not None
      else jnp.result_type(lhs.dtype, rhs.dtype)
  )

  def _dot(
      a: qarray.QArray,
      b: qarray.QArray,
  ) -> jax.Array:
    if is_microscale:
      a_fp8, s_a = _block_shift_to_fp8(
          a,
          dimension_numbers=dimension_numbers,
          for_lhs=True,
          tile_size=tile_size,
      )
      b_fp8, s_b = _block_shift_to_fp8(
          b,
          dimension_numbers=dimension_numbers,
          for_lhs=False,
          tile_size=tile_size,
      )
      a_coarse = qarray.QArray(
          qvalue=a_fp8,
          scale=s_a,
          qtype=a_fp8.dtype,
      )
      b_coarse = qarray.QArray(
          qvalue=b_fp8,
          scale=s_b,
          qtype=b_fp8.dtype,
      )
      # Call _fast_dot_general to ensure using physical FP8.
      return dg._fast_dot_general(  # pylint: disable=protected-access
          a_coarse,
          b_coarse,
          dimension_numbers=dimension_numbers,
          precision=precision,
          preferred_element_type=jnp.float32,
      )

    a_c = qarray.QArray(
        qvalue=a.qvalue.astype(compute_dtype),
        scale=a.scale,
        zero_point=a.zero_point,
        qtype=compute_dtype,
    )
    b_c = qarray.QArray(
        qvalue=b.qvalue.astype(compute_dtype),
        scale=b.scale,
        zero_point=b.zero_point,
        qtype=compute_dtype,
    )
    return dg.dot_general(
        a_c,
        b_c,
        dimension_numbers=dimension_numbers,
        precision=precision,
        preferred_element_type=jnp.float32,
    )

  fp4_type = 'mxfp4' if is_microscale else jnp.float4_e2m1fn
  int4_type = 'mxint4' if is_microscale else jnp.int4
  if multipass_mode in (
      'three_pass_fp8_fp4/fp4_fp4/fp4',
      'three_pass_mxfp8_mxfp4/mxfp4_mxfp4/mxfp4',
  ):
    downcast_qtype, residual_qtype = fp4_type, fp4_type
  elif multipass_mode in (
      'three_pass_fp8_int4/int4_int4/int4',
      'three_pass_mxfp8_mxint4/mxint4_mxint4/mxint4',
  ):
    downcast_qtype, residual_qtype = int4_type, int4_type
  elif multipass_mode in (
      'three_pass_fp8_int4/fp4_fp4/int4',
      'three_pass_mxfp8_mxint4/mxfp4_mxfp4/mxint4',
  ):
    downcast_qtype, residual_qtype = int4_type, fp4_type
  elif multipass_mode in (
      'three_pass_fp8_fp4/int4_int4/fp4',
      'three_pass_mxfp8_mxfp4/mxint4_mxint4/mxfp4',
  ):
    downcast_qtype, residual_qtype = fp4_type, int4_type
  else:
    raise ValueError(f'Unknown hybrid multipass_mode: {multipass_mode}')

  # Why Pass 1 uses calib = f'absmax,{448.0 / 256.0}' when downcasting coarse
  # operands to INT4/MXINT4:
  # - In Qwix, qarray.calibrate multiplies absmax(x) by 448.0 / 256.0, and
  #   qarray.compute_scale_zero_point divides by
  #   numerics.get_symmetric_bound(float8_e4m3fn) = 448.0, yielding:
  #     s0 = (absmax(x) * (448.0 / 256.0)) / 448.0 = absmax(x) / 256.0.
  # - Setting the Pass 1 FP8 ceiling to 256.0 (instead of 448.0) aligns both the
  #   coarse downcast grid and the top-binade residual step:
  #   1. Coarse INT4/MXINT4 downcast alignment: _downcast_by_shift divides the
  #      Pass 1 FP8 qvalue by shift = 32.0 (scaling the block scale to
  #      32.0 * s0). This maps [-256.0, +256.0] / 32.0 to [-8.0, +8.0], aligning
  #      with the two's-complement INT4 range [-8, +7] (with +7 capturing
  #      [7.0, 8.0]), whereas a 448.0 ceiling would map to 448.0 / 32.0 = 14.0
  #      and heavily saturate [7.5, 14.0] to +7.
  #   2. Top-binade residual step alignment: In float8_e4m3fn, the top binade
  #      [128.0, 256.0] under this calibration has anchor spacing
  #      Delta = 16.0 * s0, so subdividing Delta into 16 INT4 residual bins
  #      yields residual step size s_res = Delta / 16 = s0.
  calib = f'absmax,{448.0 / 256.0}' if downcast_qtype == int4_type else 'absmax'

  # Why the INT4/MXINT4 residual range is [-8.5 * s_res, +7.5 * s_res) and why
  # residual_calib = f'absmax,{7.5 / 8.5}' is mathematically exact:
  # - Each two's-complement INT4 code q1 in {-8, -7, ..., +7} has a unit-width
  #   rounding bucket [q1 - 0.5, q1 + 0.5) * s_res, so the union over all 16
  #   codes spans [-8.5 * s_res, +7.5 * s_res) around each FP8 anchor (where
  #   s_res = Delta / 16 subdivides the FP8 anchor step Delta into 16 equal
  #   parts).
  # - Because _quantize_fp8_for_int4_residual shifts x by +0.5 * s_res before
  #   Pass 1 FP8 rounding, the handoff midpoint between adjacent FP8 anchors
  #   a_L and a_R = a_L + 16 * s_res moves to a_L + 7.5 * s_res = a_R - 8.5 *
  #   s_res, bounding the exact residual r = x - dequantize(a0) to
  #   [-8.5 * s_res, +7.5 * s_res) with absmax(r) = 8.5 * s_res.
  # - In Qwix, qarray.calibrate multiplies absmax(r) by the factor 7.5 / 8.5,
  #   and qarray.compute_scale_zero_point divides by
  #   numerics.get_symmetric_bound(int4) = 7.5, yielding:
  #     scale = (absmax(r) * (7.5 / 8.5)) / 7.5 = absmax(r) / 8.5.
  #   For power-of-2 mxint4 block scales, 2 ** ceil(log2(absmax(r) / 8.5))
  #   snaps directly to s_res = Delta / 16 whenever absmax(r) in
  #   (4.25 * s_res, 8.5 * s_res] without triggering a 2x scale jump on the
  #   [-8.5, -7.5) * s_res negative tail, populating all 16 bins [-8, +7]
  #   uniformly at 6.25% with zero positive clipping. For floating-point int4
  #   tile scales, scale = absmax(r) / 8.5 equals s_res when the tile attains
  #   the worst-case bound absmax(r) = 8.5 * s_res (and scales proportionally
  #   to the tile's actual residual range on smaller tiles).
  is_int4_residual = residual_qtype == int4_type
  residual_calib = f'absmax,{7.5 / 8.5}' if is_int4_residual else 'absmax'

  if is_microscale:  # Input is quantized into mxfp8 with block size 32.
    fp8_qtype = 'mxfp8'
    fp8_tile_size = 32
    residual_tile_size = 32
  else:  # Standard hybrid mode: uses regular FP8 and user-specified tile_size.
    fp8_qtype = jnp.float8_e4m3fn
    fp8_tile_size = tile_size
    residual_tile_size = tile_size

  how_l_fp8 = dg.get_how_to_quantize(
      dimension_numbers=dimension_numbers,
      ndims=(lhs.ndim, rhs.ndim),
      for_lhs=True,
      qtype=fp8_qtype,
      tile_size=fp8_tile_size,
      calibration_method=calib,
  )
  how_r_fp8 = dg.get_how_to_quantize(
      dimension_numbers=dimension_numbers,
      ndims=(lhs.ndim, rhs.ndim),
      for_lhs=False,
      qtype=fp8_qtype,
      tile_size=fp8_tile_size,
      calibration_method=calib,
  )

  # Quantize inputs into Pass 1 operands and use _dot
  if is_int4_residual:
    a0 = _quantize_fp8_for_int4_residual(lhs, how_l_fp8)
    b0 = _quantize_fp8_for_int4_residual(rhs, how_r_fp8)
  else:
    a0 = qarray.quantize(lhs, how_l_fp8)
    b0 = qarray.quantize(rhs, how_r_fp8)
  c00 = _dot(a0, b0)

  # Compute residuals.
  r_a = lhs - qarray.dequantize(a0)
  r_b = rhs - qarray.dequantize(b0)

  # Cross pass quantizers for residuals
  how_r_residual = dg.get_how_to_quantize(
      dimension_numbers=dimension_numbers,
      ndims=(lhs.ndim, rhs.ndim),
      for_lhs=False,
      qtype=residual_qtype,
      tile_size=residual_tile_size,
      calibration_method=residual_calib,
  )
  how_l_residual = dg.get_how_to_quantize(
      dimension_numbers=dimension_numbers,
      ndims=(lhs.ndim, rhs.ndim),
      for_lhs=True,
      qtype=residual_qtype,
      tile_size=residual_tile_size,
      calibration_method=residual_calib,
  )

  # Convert coarse operands mxfp8 -> mxfp4 / mxint4 for each block size
  # 32 by simple multiplication (_downcast_by_shift) without a second reduction.
  # Downcast coarse operands inherit the block-32 scales (scale * shift).
  a0_downcast = _downcast_by_shift(a0, downcast_qtype)
  b1 = qarray.quantize(r_b, how_r_residual)
  # Cross-pass matmuls execute in hardware FP8 via _dot.
  c01 = _dot(a0_downcast, b1)

  a1 = qarray.quantize(r_a, how_l_residual)
  b0_downcast = _downcast_by_shift(b0, downcast_qtype)
  c10 = _dot(a1, b0_downcast)

  # Sum in f32 and cast once to avoid intermediate bf16 rounding.
  return (c00 + c01 + c10).astype(target_out_type)


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

  if multipass_mode in (
      'four_pass_int4',
      'three_pass_int4',
      'two_pass_lhs_int4',
      'two_pass_rhs_int4',
      'four_pass_zero_offset_int4',
      'six_pass_zero_offset_int4',
      'eight_pass_zero_offset_int4',
  ):
    # Note for the selected compute_dtype is fp8. The intended goal is emulation
    # for quality studies using TPU7x.
    return _int8_multipass_dot_general(
        lhs,
        rhs,
        dimension_numbers=dimension_numbers,
        compute_dtype=jnp.float8_e4m3fn,
        multipass_mode=multipass_mode,
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
  elif multipass_mode in (
      'three_pass_fp8_fp4/fp4_fp4/fp4',
      'three_pass_fp8_int4/int4_int4/int4',
      'three_pass_fp8_int4/fp4_fp4/int4',
      'three_pass_fp8_fp4/int4_int4/fp4',
      'three_pass_mxfp8_mxfp4/mxfp4_mxfp4/mxfp4',
      'three_pass_mxfp8_mxint4/mxint4_mxint4/mxint4',
      'three_pass_mxfp8_mxint4/mxfp4_mxfp4/mxint4',
      'three_pass_mxfp8_mxfp4/mxint4_mxint4/mxfp4',
  ):
    return _hybrid_fp8_4bit_dot_general(
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
