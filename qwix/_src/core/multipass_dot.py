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

- Hybrid types of multi-pass.
For the hybrid cases, the first pass A0 B0 is done in fp8, but 4 bit formats are
used for the A0 B1 and A1 B0 terms.
The nomenclature three_pass_fp8_{a0}/{b1}_{a1}/{b0} is used for these hybrid
types for clarity. To avoid complex naming there are aliases.
    1. 'three_pass_fp8_fp4' (longer name: 'three_pass_fp8_fp4/fp4_fp4/fp4')
    2. 'three_pass_fp8_int4' (longer name: 'three_pass_fp8_int4/int4_int4/int4')
    3. 'three_pass_fp8_mixed4' (longer name:
    'three_pass_fp8_int4/fp4_fp4/int4_int4')
Additionally we support micro-scaled versions of each.


Additionally it supports int8 by decomposing into int4. This could be useful on
devices with high int4 FLOPs. For the purposes of emulation we do this on the
fp8 native path. We support the following int8 emulation modes:
    1. 'four_pass_int4': full emulation for int8 x int8 matmul
    2. 'three_pass_int4': Truncated emulation, dropping the A1 B1 term
    3. 'two_pass_lhs_int4': Two passes for the LHS (emulating int8 x int4)
    4. 'two_pass_rhs_int4': Two passes for the RHS (emulating int4 x int8)

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

from collections.abc import Sequence
import functools
import itertools
from typing import Any, Literal, TypeAlias
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
    'karatsuba_int4',
    'three_pass_int4',
    'two_pass_lhs_int4',
    'two_pass_rhs_int4',
    # Hybrid FP8 + 4-bit modes and explicit pass-specification aliases.
    # Slash-separated aliases explicitly define operand formats per pass
    # (e.g. 'three_pass_fp8_fp4/fp4_fp4/fp4' specifies Pass 1 FP8 x FP8,
    # followed by cross-passes FP8 x FP4 and FP4 x FP8).
    'three_pass_fp8_fp4',
    'three_pass_fp8_fp4/fp4_fp4/fp4',
    'three_pass_fp8_int4',
    'three_pass_fp8_int4/int4_int4/int4',
    'three_pass_fp8_mixed4',
    'three_pass_fp8_int4/fp4_fp4/int4_int4',
    # Microscaled hybrid modes and explicit pass-specification aliases
    # (e.g. 'three_pass_mxfp8_16_mxfp4/mxfp4_mxfp4/mxfp4' specifies Pass 1
    # MXFP8_16 x MXFP8_16, followed by cross-passes MXFP4 x MXFP4).
    'three_pass_mxfp8_16_mxfp4',
    'three_pass_mxfp8_16_mxfp4/mxfp4_mxfp4/mxfp4',
    'three_pass_mxfp8_16_mxint4',
    'three_pass_mxfp8_16_mxint4/mxint4_mxint4/mxint4',
    'three_pass_mxfp8_16_mxmixed4',
    'three_pass_mxfp8_16_mxint4/mxfp4_mxfp4/mxint4',
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
    compute_dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    triangular: bool = False,
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array:
  """Emulated signed int8 GEMM using 4 (full) or 3 (triangular) uncoupled passes."""
  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers

  a_h, a_l = _prep_int8_parts(a)
  b_h, b_l = _prep_int8_parts(b)

  # Note: Despite converting to compute_type here, if native int4 is supported
  # it can be used for all passes.
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

  xy_product = (p11 << 8) + ((p10 + p01) << 4)
  if not triangular:
    p00 = _dot(a_l_c, b_l_c)
    xy_product = xy_product + p00

  k_size = 1
  for d in lhs_ca:
    k_size *= a.shape[d]

  sum_a = jnp.sum(a, axis=lhs_ca)
  sum_b = jnp.sum(b, axis=rhs_ca)

  lhs_rem_ndims = a.ndim - len(lhs_ca) - len(lhs_ba)
  rhs_rem_ndims = b.ndim - len(rhs_ca) - len(rhs_ba)

  for _ in range(rhs_rem_ndims):
    sum_a = jnp.expand_dims(sum_a, axis=-1)
  insert_pos = len(lhs_ba)
  for _ in range(lhs_rem_ndims):
    sum_b = jnp.expand_dims(sum_b, axis=insert_pos)

  res = xy_product + (sum_a * 8) + (sum_b * 8) - (64 * k_size)
  if preferred_element_type is not None:
    return res.astype(preferred_element_type)
  return res


def _triangular_signed_int8_dot_general(
    a: jax.Array,
    b: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers = (((1,), (0,)), ((), ())),
    *,
    compute_dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array:
  """Triangular signed int8 GEMM using 3 uncoupled native signed int4 passes."""
  return _emulated_signed_int8_dot_general(
      a,
      b,
      dimension_numbers=dimension_numbers,
      compute_dtype=compute_dtype,
      triangular=True,
      preferred_element_type=preferred_element_type,
  )


def _asymmetric_signed_int4_int8_dot_general(
    a: jax.Array,
    b: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers = (((1,), (0,)), ((), ())),
    *,
    compute_dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    multipass_mode: str = 'two_pass_lhs_int4',
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array:
  """Asymmetric signed INT4 x INT8 matrix multiplication via 2 INT4 passes."""
  a_int = a.astype(jnp.int32)
  b_int = b.astype(jnp.int32)
  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers

  # Note: Despite converting to compute_type here, if native int4 x int8 is
  # supported it can be used for both passes.
  def _dot(x: jax.Array, y: jax.Array) -> jax.Array:
    return jax.lax.dot_general(
        x.astype(compute_dtype),
        y.astype(compute_dtype),
        dimension_numbers=dimension_numbers,
        preferred_element_type=jnp.float32,
    ).astype(jnp.int32)

  if multipass_mode == 'two_pass_rhs_int4':
    b_h, b_l = _prep_int8_parts(b_int)

    p1 = _dot(a_int, b_h)
    p0 = _dot(a_int, b_l)

    sum_a = jnp.sum(a_int, axis=lhs_ca)
    rhs_rem_ndims = b.ndim - len(rhs_ca) - len(rhs_ba)
    for _ in range(rhs_rem_ndims):
      sum_a = jnp.expand_dims(sum_a, axis=-1)

    res = (p1 << 4) + p0 + (sum_a * 8)
  elif multipass_mode == 'two_pass_lhs_int4':
    a_h, a_l = _prep_int8_parts(a_int)

    p1 = _dot(a_h, b_int)
    p0 = _dot(a_l, b_int)

    sum_b = jnp.sum(b_int, axis=rhs_ca)
    lhs_rem_ndims = a.ndim - len(lhs_ca) - len(lhs_ba)
    insert_pos = len(lhs_ba)
    for _ in range(lhs_rem_ndims):
      sum_b = jnp.expand_dims(sum_b, axis=insert_pos)

    res = (p1 << 4) + p0 + (sum_b * 8)
  else:
    raise ValueError(
        f'Unsupported multipass_mode for asymmetric int8: {multipass_mode}'
    )

  if preferred_element_type is not None:
    return res.astype(preferred_element_type)
  return res


def _asymmetric_int_dot_general(
    lhs: jax.Array,
    rhs: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers = (((1,), (0,)), ((), ())),
    *,
    compute_dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    multipass_mode: str = 'two_pass_lhs_int4',
    tile_size: int | None = 256,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    **kwargs: Any,
) -> jax.Array:
  """Evaluates asymmetric integer matrix multiplication."""
  del kwargs
  return _int8_multipass_dot_general(
      lhs,
      rhs,
      dimension_numbers=dimension_numbers,
      compute_dtype=compute_dtype,
      multipass_mode=multipass_mode,
      tile_size=tile_size,
      preferred_element_type=preferred_element_type,
  )


def _int8_multipass_dot_general(
    lhs: jax.Array,
    rhs: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers = (((1,), (0,)), ((), ())),
    *,
    multipass_mode: str = 'four_pass_int4',
    compute_dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    tile_size: int | None = 256,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    **kwargs: Any,
) -> jax.Array:
  """Executes scaled INT8 (channelwise or subchannel) GEMM via INT4 passes."""
  del kwargs
  match multipass_mode:
    case 'four_pass_int4':
      lhs_qtype = jnp.int8
      rhs_qtype = jnp.int8
      dot_fn = _emulated_signed_int8_dot_general
    case 'three_pass_int4':
      lhs_qtype = jnp.int8
      rhs_qtype = jnp.int8
      dot_fn = _triangular_signed_int8_dot_general
    case 'two_pass_lhs_int4':
      lhs_qtype = jnp.int8
      rhs_qtype = jnp.int4
      dot_fn = functools.partial(
          _asymmetric_signed_int4_int8_dot_general,
          multipass_mode='two_pass_lhs_int4',
      )
    case 'two_pass_rhs_int4':
      lhs_qtype = jnp.int4
      rhs_qtype = jnp.int8
      dot_fn = functools.partial(
          _asymmetric_signed_int4_int8_dot_general,
          multipass_mode='two_pass_rhs_int4',
      )
    case _:
      raise ValueError(f'Unknown multipass_mode: {multipass_mode!r}')

  if not isinstance(lhs, qarray.QArray):
    lhs_how = dg.get_how_to_quantize(
        dimension_numbers=dimension_numbers,
        ndims=(lhs.ndim, rhs.ndim),
        for_lhs=True,
        tile_size=tile_size,
        qtype=lhs_qtype,
    )
    q_lhs = qarray.quantize(lhs, lhs_how)
  else:
    q_lhs = lhs

  if not isinstance(rhs, qarray.QArray):
    rhs_how = dg.get_how_to_quantize(
        dimension_numbers=dimension_numbers,
        ndims=(lhs.ndim, rhs.ndim),
        for_lhs=False,
        tile_size=tile_size,
        qtype=rhs_qtype,
    )
    q_rhs = qarray.quantize(rhs, rhs_how)
  else:
    q_rhs = rhs

  (lhs_ca, rhs_ca), _ = dimension_numbers
  lhs_value = q_lhs.qvalue
  rhs_value = q_rhs.qvalue
  lhs_scale = q_lhs.scale
  rhs_scale = q_rhs.scale

  lhs_tiled_axes = qarray.get_tiled_axes(q_lhs)
  rhs_tiled_axes = qarray.get_tiled_axes(q_rhs)

  ca_tile_counts = []
  for l, r in zip(lhs_ca, rhs_ca):
    tile_size = lhs_tiled_axes.get(l) or rhs_tiled_axes.get(r)
    ca_tile_counts.append(lhs_value.shape[l] // tile_size if tile_size else 1)

  lhs_scale_transpose, rhs_scale_transpose = (
      dg._get_scale_transpose(  # pylint: disable=protected-access
          dimension_numbers, (lhs_value.ndim, rhs_value.ndim)
      )
  )

  def take_slice(array, ca, ca_tile_indices):
    indices = []
    for i, s in enumerate(array.shape):
      if i not in ca or s == 1:
        indices.append(slice(None))
      else:
        idx = ca_tile_indices[ca.index(i)]
        count = ca_tile_counts[ca.index(i)]
        size = s // count
        indices.append(slice(idx * size, (idx + 1) * size))
    return array[tuple(indices)]

  acc = None
  for ca_tile_indices in itertools.product(*map(range, ca_tile_counts)):
    out = dot_fn(
        take_slice(lhs_value, lhs_ca, ca_tile_indices),
        take_slice(rhs_value, rhs_ca, ca_tile_indices),
        dimension_numbers=dimension_numbers,
        compute_dtype=compute_dtype,
        preferred_element_type=jnp.float32,
    )
    if lhs_scale is not None:
      scale = take_slice(lhs_scale, lhs_ca, ca_tile_indices)
      scale = qarray.transpose_array(scale, lhs_scale_transpose)
      out = qarray.call_with_generic_broadcast(jnp.multiply, out, scale)
    if rhs_scale is not None:
      scale = take_slice(rhs_scale, rhs_ca, ca_tile_indices)
      scale = qarray.transpose_array(scale, rhs_scale_transpose)
      out = qarray.call_with_generic_broadcast(jnp.multiply, out, scale)
    acc = out if acc is None else acc + out

  assert acc is not None
  if preferred_element_type is not None:
    return acc.astype(preferred_element_type)
  return acc


def _downcast_by_shift(
    qarr: qarray.QArray,
    target_qtype: jax.typing.DTypeLike,
) -> qarray.QArray:
  """Downcasts a calibrated FP8 QArray into 4-bit (FP4 or INT4) via scale-shifting."""
  if target_qtype in (jnp.float4_e2m1fn, 'mxfp4'):
    shift = 64.0
    scaled_val = qarr.qvalue.astype(jnp.float32) / shift
    new_qval = jnp.clip(scaled_val, -6.0, 6.0).astype(jnp.float4_e2m1fn)
    actual_qtype = (
        jnp.float4_e2m1fn if target_qtype == 'mxfp4' else target_qtype
    )
  else:
    shift = 32.0
    scaled_val = qarr.qvalue.astype(jnp.float32) / shift
    new_qval = jnp.clip(jnp.round(scaled_val), -7.0, 7.0).astype(jnp.int4)
    actual_qtype = jnp.int4 if target_qtype == 'mxint4' else target_qtype
  return qarray.QArray(
      qvalue=new_qval, scale=qarr.scale * shift, qtype=actual_qtype
  )


def _hybrid_fp8_4bit_dot_general(
    lhs: jax.Array,
    rhs: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers = (((1,), (0,)), ((), ())),
    *,
    compute_dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    multipass_mode: str = 'three_pass_fp8_fp4',
    tile_size: int | None = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    precision: jax.lax.PrecisionLike = None,
    **kwargs: Any,
) -> jax.Array:
  """Hybrid 3-pass GEMM: FP8 for Pass 1, 4-bit (FP4, INT4, or Mixed) for cross passes."""

  def _dot(a: qarray.QArray, b: qarray.QArray) -> jax.Array:
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
        preferred_element_type=preferred_element_type,
        **kwargs,
    )

  calib_override = kwargs.pop('calibration_method', None) or kwargs.pop(
      'calib', None
  )
  if multipass_mode in ('three_pass_fp8_fp4', 'three_pass_fp8_fp4/fp4_fp4/fp4'):
    calib = 'absmax'
    cross_qtype_lhs = jnp.float4_e2m1fn
    cross_qtype_rhs = jnp.float4_e2m1fn
  elif multipass_mode in (
      'three_pass_fp8_int4',
      'three_pass_fp8_int4/int4_int4/int4',
  ):
    calib = f'absmax,{448.0 / 256.0}'
    cross_qtype_lhs = jnp.int4
    cross_qtype_rhs = jnp.int4
  elif multipass_mode in (
      'three_pass_fp8_mixed4',
      'three_pass_fp8_int4/fp4_fp4/int4_int4',
  ):
    calib = f'absmax,{448.0 / 256.0}'
    cross_qtype_lhs = jnp.int4
    cross_qtype_rhs = jnp.float4_e2m1fn
  else:
    raise ValueError(f'Unknown hybrid multipass_mode: {multipass_mode}')

  if calib_override is not None:
    calib = calib_override

  how_l_fp8 = dg.get_how_to_quantize(
      dimension_numbers=dimension_numbers,
      ndims=(lhs.ndim, rhs.ndim),
      for_lhs=True,
      qtype=jnp.float8_e4m3fn,
      tile_size=tile_size,
      calibration_method=calib,
  )
  how_r_fp8 = dg.get_how_to_quantize(
      dimension_numbers=dimension_numbers,
      ndims=(lhs.ndim, rhs.ndim),
      for_lhs=False,
      qtype=jnp.float8_e4m3fn,
      tile_size=tile_size,
      calibration_method=calib,
  )

  a0 = qarray.quantize(lhs, how_l_fp8)
  b0 = qarray.quantize(rhs, how_r_fp8)
  c00 = _dot(a0, b0)

  # Residuals
  r_a = lhs - qarray.dequantize(a0)
  r_b = rhs - qarray.dequantize(b0)

  # Cross pass quantizers for residuals
  how_r_cross1 = dg.get_how_to_quantize(
      dimension_numbers=dimension_numbers,
      ndims=(lhs.ndim, rhs.ndim),
      for_lhs=False,
      qtype=cross_qtype_rhs,
      tile_size=tile_size,
  )
  if multipass_mode in (
      'three_pass_fp8_mixed4',
      'three_pass_fp8_int4/fp4_fp4/int4_int4',
  ):
    cross2_qtype_lhs = jnp.float4_e2m1fn
    cross2_qtype_rhs = jnp.int4
  else:
    cross2_qtype_lhs = cross_qtype_lhs
    cross2_qtype_rhs = cross_qtype_rhs

  how_l_cross2 = dg.get_how_to_quantize(
      dimension_numbers=dimension_numbers,
      ndims=(lhs.ndim, rhs.ndim),
      for_lhs=True,
      qtype=cross2_qtype_lhs,
      tile_size=tile_size,
  )

  # A0 B1
  a0_downcast = _downcast_by_shift(a0, cross_qtype_lhs)
  b1_downcast = qarray.quantize(r_b, how_r_cross1)
  c01 = _dot(a0_downcast, b1_downcast)

  # A1 B0
  a1_downcast = qarray.quantize(r_a, how_l_cross2)
  b0_downcast = _downcast_by_shift(b0, cross2_qtype_rhs)
  c10 = _dot(a1_downcast, b0_downcast)

  return c00 + c01 + c10


def _block_shift_to_fp8(
    q_tensor: qarray.QArray,
    target_dtype: jax.typing.DTypeLike,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    for_lhs: bool,
    block_size: int = 32,
) -> tuple[jax.Array, jax.Array]:
  """Absorbs block scales into FP8 exponents so matmul runs on hardware FP8 path."""
  del block_size
  (lhs_ca, rhs_ca), _ = dimension_numbers
  ca_axis = lhs_ca[0] if for_lhs else rhs_ca[0]
  max_scale = jnp.max(q_tensor.scale, axis=ca_axis, keepdims=True)
  max_scale = jnp.where(max_scale == 0, 1.0, max_scale)
  rel_scale = q_tensor.scale / max_scale
  rel_scaled_val = qarray.call_with_generic_broadcast(
      jnp.multiply, q_tensor.qvalue.astype(jnp.float32), rel_scale
  )
  return rel_scaled_val.astype(target_dtype), max_scale


def _microscaled_hybrid_fp8_4bit_dot_general(
    lhs: jax.Array,
    rhs: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers = (((1,), (0,)), ((), ())),
    *,
    multipass_mode: str = 'three_pass_mxfp8_16_mxfp4',
    tile_size: int | Sequence[int | None] | None = None,
    preferred_element_type: jax.typing.DTypeLike | None = None,
    precision: jax.lax.PrecisionLike = None,
    **kwargs: Any,
) -> jax.Array:
  """Microscaled Hybrid 3-Pass GEMM with hardware FP8 YOLO accumulation."""
  calib_override = kwargs.pop('calibration_method', None) or kwargs.pop(
      'calib', None
  )
  del kwargs
  is_int4_cross = 'mxint4' in multipass_mode and 'mxfp4' not in multipass_mode
  is_mixed_cross = (
      'mxmixed4' in multipass_mode or 'mxint4/mxfp4' in multipass_mode
  )

  if is_int4_cross:
    calib = f'absmax,{448.0 / 256.0}'
    cross_lhs_dtype = jnp.float8_e5m2
    cross_rhs_dtype = jnp.float8_e5m2
    cross_lhs_qtype = 'mxint4'
    cross_rhs_qtype = 'mxint4'
  elif is_mixed_cross:
    calib = f'absmax,{448.0 / 256.0}'
    cross_lhs_dtype = jnp.float8_e5m2
    cross_rhs_dtype = jnp.float8_e4m3fn
    cross_lhs_qtype = 'mxint4'
    cross_rhs_qtype = 'mxfp4'
  else:
    calib = 'absmax'
    cross_lhs_dtype = jnp.float8_e4m3fn
    cross_rhs_dtype = jnp.float8_e4m3fn
    cross_lhs_qtype = 'mxfp4'
    cross_rhs_qtype = 'mxfp4'

  if calib_override is not None:
    calib = calib_override

  how_l_fp8 = dg.get_how_to_quantize(
      dimension_numbers=dimension_numbers,
      ndims=(lhs.ndim, rhs.ndim),
      for_lhs=True,
      qtype='mxfp8_16',
      tile_size=16,
      calibration_method=calib,
  )
  how_r_fp8 = dg.get_how_to_quantize(
      dimension_numbers=dimension_numbers,
      ndims=(lhs.ndim, rhs.ndim),
      for_lhs=False,
      qtype='mxfp8_16',
      tile_size=16,
      calibration_method=calib,
  )

  a0 = qarray.quantize(lhs, how_l_fp8)
  b0 = qarray.quantize(rhs, how_r_fp8)
  c00 = dg.dot_general(
      a0,
      b0,
      dimension_numbers=dimension_numbers,
      precision=precision,
      preferred_element_type=preferred_element_type,
  )

  # Residuals
  r_a = lhs - qarray.dequantize(a0)
  r_b = rhs - qarray.dequantize(b0)
  if tile_size is None:
    bs = 32
  elif isinstance(tile_size, int):
    bs = tile_size
  else:
    bs = 32 if tile_size[1] is None else int(tile_size[1])

  how_r_cross = dg.get_how_to_quantize(
      dimension_numbers=dimension_numbers,
      ndims=(lhs.ndim, rhs.ndim),
      for_lhs=False,
      qtype=cross_rhs_qtype,
      tile_size=bs,
  )

  if is_mixed_cross:
    how_l_cross2 = dg.get_how_to_quantize(
        dimension_numbers=dimension_numbers,
        ndims=(lhs.ndim, rhs.ndim),
        for_lhs=True,
        qtype='mxfp4',
        tile_size=bs,
    )
    cross2_rhs_qtype = 'mxint4'
    cross2_lhs_dtype = jnp.float8_e4m3fn
    cross2_rhs_dtype = jnp.float8_e5m2
  else:
    how_l_cross2 = dg.get_how_to_quantize(
        dimension_numbers=dimension_numbers,
        ndims=(lhs.ndim, rhs.ndim),
        for_lhs=True,
        qtype=cross_lhs_qtype,
        tile_size=bs,
    )
    cross2_rhs_qtype = cross_rhs_qtype
    cross2_lhs_dtype = cross_lhs_dtype
    cross2_rhs_dtype = cross_rhs_dtype

  # Pass 2: A0 B1
  a0_down = _downcast_by_shift(a0, cross_lhs_qtype)
  b1_down = qarray.quantize(r_b, how_r_cross)
  a0_fp8, s_a0 = _block_shift_to_fp8(
      a0_down, cross_lhs_dtype, dimension_numbers, for_lhs=True, block_size=bs
  )
  b1_fp8, s_b1 = _block_shift_to_fp8(
      b1_down, cross_rhs_dtype, dimension_numbers, for_lhs=False, block_size=bs
  )
  c01_raw = jax.lax.dot_general(
      a0_fp8,
      b1_fp8,
      dimension_numbers,
      precision=precision,
      preferred_element_type=jnp.float32,
  )
  c01 = c01_raw * (s_a0 * s_b1)

  # Pass 3: A1 B0
  a1_down = qarray.quantize(r_a, how_l_cross2)
  b0_down = _downcast_by_shift(b0, cross2_rhs_qtype)
  a1_fp8, s_a1 = _block_shift_to_fp8(
      a1_down, cross2_lhs_dtype, dimension_numbers, for_lhs=True, block_size=bs
  )
  b0_fp8, s_b0 = _block_shift_to_fp8(
      b0_down, cross2_rhs_dtype, dimension_numbers, for_lhs=False, block_size=bs
  )
  c10_raw = jax.lax.dot_general(
      a1_fp8,
      b0_fp8,
      dimension_numbers,
      precision=precision,
      preferred_element_type=jnp.float32,
  )
  c10 = c10_raw * (s_a1 * s_b0)

  res = c00.astype(jnp.float32) + c01 + c10
  if preferred_element_type is not None:
    return res.astype(preferred_element_type)
  return res


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
      'three_pass_mxfp8_16_mxfp4',
      'three_pass_mxfp8_16_mxfp4/mxfp4_mxfp4/mxfp4',
      'three_pass_mxfp8_16_mxint4',
      'three_pass_mxfp8_16_mxint4/mxint4_mxint4/mxint4',
      'three_pass_mxfp8_16_mxmixed4',
      'three_pass_mxfp8_16_mxint4/mxfp4_mxfp4/mxint4',
  ):
    return _microscaled_hybrid_fp8_4bit_dot_general(
        lhs,
        rhs,
        dimension_numbers=dimension_numbers,
        multipass_mode=multipass_mode,
        tile_size=tile_size,
        preferred_element_type=preferred_element_type,
        precision=precision,
        **kwargs,
    )

  if multipass_mode in (
      'three_pass_fp8_fp4',
      'three_pass_fp8_fp4/fp4_fp4/fp4',
      'three_pass_fp8_int4',
      'three_pass_fp8_int4/int4_int4/int4',
      'three_pass_fp8_mixed4',
      'three_pass_fp8_int4/fp4_fp4/int4_int4',
  ):
    compute_dtype = kwargs.pop('compute_dtype', jnp.float8_e4m3fn)
    return _hybrid_fp8_4bit_dot_general(
        lhs,
        rhs,
        dimension_numbers=dimension_numbers,
        compute_dtype=compute_dtype,
        multipass_mode=multipass_mode,
        tile_size=tile_size,
        preferred_element_type=preferred_element_type,
        precision=precision,
        **kwargs,
    )

  if multipass_mode in (
      'four_pass_int4',
      'three_pass_int4',
      'two_pass_lhs_int4',
      'two_pass_rhs_int4',
  ):
    compute_dtype = kwargs.pop('compute_dtype', jnp.float8_e4m3fn)
    return _int8_multipass_dot_general(
        lhs,
        rhs,
        dimension_numbers=dimension_numbers,
        compute_dtype=compute_dtype,
        multipass_mode=multipass_mode,
        tile_size=tile_size,
        preferred_element_type=preferred_element_type,
        **kwargs,
    )

  # Except for the emulated int modes above, we only support FP8 for now.
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
        f'Unknown multipass mode: {multipass_mode!r}. Expected one of:'
        " 'three_pass_fp8', 'four_pass_fp8', 'two_pass_lhs_fp8',"
        " 'two_pass_rhs_fp8', 'four_pass_int4',"
        " 'three_pass_int4', 'two_pass_lhs_int4', 'two_pass_rhs_int4',"
        " 'three_pass_fp8_fp4', 'three_pass_fp8_int4', 'three_pass_fp8_mixed4'."
    )
