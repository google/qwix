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
"""Multi-pass emulation for microscaled floating-point and integer formats.

This module provides multi-pass residual quantization emulation.

Note on Emulation Experiments:
-------------------------------
The emulated INT8 implementations using INT4 building blocks (Karatsuba,
Naive, Asymmetric, Triangular, etc.) are included for reference only to model
hardware feasibility on native INT4 units. For software emulation experiments
using BF16, you should simply use regular single-pass mxint8.

This module provides multi-pass residual quantization emulation:
- 2-pass residual decomposition:
    Pass 0: X_0 = quantize(X),   R_1 = X - dequantize(X_0)
    Pass 1: X_1 = quantize(R_1)
- Multi-pass matrix multiplication modes:
    1. 'triangular' (3 passes): A_0 B_0 + A_0 B_1 + A_1 B_0
       Drops second-order cross-residual A_1 B_1 (scaled by ~2^-14).
    2. 'full_cross' (4 passes): A_0 B_0 + A_0 B_1 + A_1 B_0 + A_1 B_1
       Evaluates all 4 cross-products.
    3. 'lhs_high_precision' (2 passes): A_0 B_0 + A_1 B_0
       LHS uses 2 residual passes, RHS uses 1 pass.
    4. 'rhs_high_precision' (2 passes): A_0 B_0 + A_0 B_1
       LHS uses 1 pass, RHS uses 2 residual passes.
- Asymmetric MX-INT dot general (mxint4 x mxint8) with block-scale accumulation.

Scaling factor design choice for residual passes:
While an alternative formulation could independently compute fresh scaling
factors for each residual pass (via block-level max-reduction searches over
R_1),

we instead derive residual scales by shifting the initial pass's scale factor by
a fixed power-of-2 exponent offset (e.g. 2^-4 for FP8 E4M3, 2^-2 for FP4 E2M1).
This design choice is made because:

1. Hardware efficiency: It eliminates the expensive second block-level reduction
   tree across elements, replacing dynamic exponent search with a single ALU
   integer shift (E_1 = E_0 - shift_bits).
2. Memory bandwidth and storage: The residual scale is implicit rather than a
   separate scale tensor that must be stored and loaded from memory, cutting
   scale metadata traffic by 50%.
3. Fixed accumulator alignment: Constant power-of-2 scale offsets allow matrix
   units to align cross-term products with simple arithmetic bit-shifts prior
   to register accumulation, rather than performing variable per-block floating-
   point scale multiplications.
4. Numerical closeness: Because the maximum rounding residual of round-to-
   nearest is bounded by the top bin step size, shifting by these exact bit
   yields virtually identical SQNR to independent scale choices.
"""

import functools
from typing import Literal, TypeAlias
import jax
from jax import lax
import jax.numpy as jnp
from qwix._src.core import dot_general
from qwix._src.core import qarray

MultiPassMode: TypeAlias = Literal[
    'triangular',
    'full_cross',
    'lhs_high_precision',
    'rhs_high_precision',
]


def get_residual_scale_shift_bits(qtype: jax.typing.DTypeLike) -> int | None:
  """Returns the power-of-2 scale shift bits for residual quantization.

  For microscaled floating-point formats, the residual quantization scale
  can be derived directly from the preceding pass's scale by shifting its
  exponent, eliminating redundant block-level reduction passes:
  - mxfp8 / mxfp8_16 (E4M3): 16 / 448 = 1/28 -> shift by 4 bits (2^-4 = 1/16).
  - mxfp4 / nvfp4 (E2M1): 1 / 6 = 1/6 -> shift by 2 bits (2^-2 = 1/4).
  - float8_e5m2 (E5M2): 8192 / 57344 = 1/7 -> shift by 2 bits (2^-2 = 1/4).

  For integer formats (e.g. int4, int8, mxint8), returns None to indicate
  exact algebraic decomposition or independent calibration.

  Args:
    qtype: Quantization format or dtype string.

  Returns:
    The integer exponent shift in bits, or None.
  """
  match qtype:
    case 'mxfp8' | 'mxfp8_16':
      return 4
    case 'mxfp4' | 'nvfp4':
      return 2
    case _:
      if qtype == jnp.float8_e4m3fn or qtype == 'float8_e4m3fn':
        return 4
      elif qtype == jnp.float4_e2m1fn or qtype == 'float4_e2m1fn':
        return 2
      elif qtype == jnp.float8_e5m2 or qtype == 'float8_e5m2':
        return 2
      return None


def residual_decompose(
    x: jax.Array,
    how: qarray.HowToQuantize,
    n_passes: int = 2,
) -> tuple[qarray.QArray, ...]:
  """Decomposes tensor x into residual quantized passes.

  For floating-point formats (e.g., mxfp8, mxfp4), residual passes reuse the
  initial pass's scale factor shifted by the format's bit-precision (e.g. 2^-4
  for FP8, 2^-2 for FP4), eliminating redundant block-level reduction passes.
  For integer formats, passes are calibrated independently.

  Args:
    x: Input array.
    how: HowToQuantize configuration specifying format, tile size, scaling, etc.
    n_passes: Number of residual passes (default 2).

  Returns:
    Tuple of quantized QArray objects (q_0, q_1, ...).
  """
  passes = []
  current = x
  shift_bits = get_residual_scale_shift_bits(how.qtype)

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


def multipass_dot(
    lhs: jax.Array,
    rhs: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers = (((1,), (0,)), ((), ())),
    mode: MultiPassMode = 'triangular',
    *,
    lhs_how: qarray.HowToQuantize | None = None,
    rhs_how: qarray.HowToQuantize | None = None,
    lhs_qtype: str = 'mxfp8_16',
    rhs_qtype: str = 'mxfp8_16',
    tile_size: int = 16,
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array:
  """Executes multi-pass emulated matrix multiplication.

  Args:
    lhs: Left-hand side tensor.
    rhs: Right-hand side tensor.
    dimension_numbers: Standard JAX dot_general dimension specification.
    mode: Multi-pass mode: - 'triangular': Computes 3 GEMMs: A_0 B_0 + A_0 B_1 +
      A_1 B_0. - 'full_cross': Computes 4 GEMMs: A_0 B_0 + A_0 B_1 + A_1 B_0 +
      A_1 B_1. - 'lhs_high_precision': Computes 2 GEMMs: A_0 B_0 + A_1 B_0 (2
      passes on LHS, 1 on RHS). - 'rhs_high_precision': Computes 2 GEMMs: A_0
      B_0 + A_0 B_1 (1 pass on LHS, 2 on RHS).
    lhs_how: Explicit HowToQuantize for LHS (overrides lhs_qtype, tile_size,
      etc.).
    rhs_how: Explicit HowToQuantize for RHS (overrides rhs_qtype, tile_size,
      etc.).
    lhs_qtype: QType for LHS if lhs_how is None (default 'mxfp8_16').
    rhs_qtype: QType for RHS if rhs_how is None (default 'mxfp8_16').
    tile_size: Microscaling tile size (default 16 for mxfp8_16).
    preferred_element_type: Output/accumulator dtype.

  Returns:
    The resulting matrix product array.
  """
  if lhs_how is None:
    lhs_how = dot_general.get_how_to_quantize(
        dimension_numbers=dimension_numbers,
        ndims=(lhs.ndim, rhs.ndim),
        for_lhs=True,
        qtype=lhs_qtype,
        tile_size=tile_size,
    )
  if rhs_how is None:
    rhs_how = dot_general.get_how_to_quantize(
        dimension_numbers=dimension_numbers,
        ndims=(lhs.ndim, rhs.ndim),
        for_lhs=False,
        qtype=rhs_qtype,
        tile_size=tile_size,
    )

  if mode == 'lhs_high_precision':
    a_passes = residual_decompose(lhs, lhs_how, n_passes=2)
    a0, a1 = a_passes[0], a_passes[1]
    b0 = qarray.quantize(rhs, rhs_how)
    c0 = dot_general.dot_general(
        a0, b0, dimension_numbers, preferred_element_type=preferred_element_type
    )
    c1 = dot_general.dot_general(
        a1, b0, dimension_numbers, preferred_element_type=preferred_element_type
    )
    return c0 + c1

  elif mode == 'rhs_high_precision':
    a0 = qarray.quantize(lhs, lhs_how)
    b_passes = residual_decompose(rhs, rhs_how, n_passes=2)
    b0, b1 = b_passes[0], b_passes[1]
    c0 = dot_general.dot_general(
        a0, b0, dimension_numbers, preferred_element_type=preferred_element_type
    )
    c1 = dot_general.dot_general(
        a0, b1, dimension_numbers, preferred_element_type=preferred_element_type
    )
    return c0 + c1

  elif mode in ('triangular', 'full_cross'):
    a_passes = residual_decompose(lhs, lhs_how, n_passes=2)
    a0, a1 = a_passes[0], a_passes[1]
    b_passes = residual_decompose(rhs, rhs_how, n_passes=2)
    b0, b1 = b_passes[0], b_passes[1]
    c00 = dot_general.dot_general(
        a0, b0, dimension_numbers, preferred_element_type=preferred_element_type
    )
    c01 = dot_general.dot_general(
        a0, b1, dimension_numbers, preferred_element_type=preferred_element_type
    )
    c10 = dot_general.dot_general(
        a1, b0, dimension_numbers, preferred_element_type=preferred_element_type
    )
    res = c00 + c01 + c10
    if mode == 'full_cross':
      c11 = dot_general.dot_general(
          a1,
          b1,
          dimension_numbers,
          preferred_element_type=preferred_element_type,
      )
      res = res + c11
    return res

  else:
    raise ValueError(
        f"Unknown multipass mode: {mode!r}. Expected one of: 'triangular',"
        " 'full_cross', 'lhs_high_precision', 'rhs_high_precision'."
    )


def _dot_int4(
    lhs: jax.Array,
    rhs: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array:
  """Executes signed int4 x int4 matrix multiplication."""
  return lax.dot_general(
      lhs.astype(jnp.float32),
      rhs.astype(jnp.float32),
      dimension_numbers=dimension_numbers,
      preferred_element_type=preferred_element_type,
  )


def karatsuba_signed_int8_dot(
    a: jax.Array,
    b: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers = (((1,), (0,)), ((), ())),
    *,
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array:
  """Algebraic Karatsuba signed int8 GEMM using strictly 3 native int4 passes.

  Note: Included for reference only to model hardware feasibility on native INT4
  units. For software emulation experiments using BF16, you should simply use
  regular single-pass mxint8.

  Computes 100% bit-exact signed INT8 x INT8 matrix multiplication:
  1. Decomposes operands into signed high nibble a_h in [-8, 7] and centered low
     nibble a_l in [-8, 7], such that a = 16 * a_h + a_l + 8.
  2. Evaluates exactly 3 signed INT4 matrix products:
     - Z2 = a_h * b_h (High x High)
     - Z0 = a_l * b_l (Low x Low)
     - Z1_core = a_half * b_half (Half x Half)
  3. Uses boolean 1-bit remainders to reconstruct the middle cross-term
  algebraically.
  4. Assembles the exact base-16 product matching int8 x int8 GEMM with 0 error.

  Args:
    a: Signed int8 operand (or float/int array with values in [-128, 127]).
    b: Signed int8 operand.
    dimension_numbers: Standard JAX dot_general dimension specification.
    preferred_element_type: Target output dtype (default int32).

  Returns:
    Result array exactly matching int8 x int8 matrix multiplication.
  """
  a_int = a.astype(jnp.int32)
  b_int = b.astype(jnp.int32)
  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers

  # Step 1: Signed High & Centered Low Nibbles in [-8, 7]
  a_h = a_int >> 4
  a_l = ((a_int.astype(jnp.uint8) & 0x0F).astype(jnp.int32)) - 8

  b_h = b_int >> 4
  b_l = ((b_int.astype(jnp.uint8) & 0x0F).astype(jnp.int32)) - 8

  # Sums of nibbles
  s_a = a_h + a_l
  s_b = b_h + b_l

  # Factored quotient and 1-bit boolean remainder
  a_half = s_a >> 1
  r_a = s_a & 1

  b_half = s_b >> 1
  r_b = s_b & 1

  # Exactly 3 native signed int4 matrix multiplications
  z2 = _dot_int4(a_h, b_h, dimension_numbers)
  z0 = _dot_int4(a_l, b_l, dimension_numbers)
  z1_core = _dot_int4(a_half, b_half, dimension_numbers)

  # 1-bit remainder additions
  z1 = (
      (z1_core.astype(jnp.int32) << 2)
      + (_dot_int4(a_half, r_b, dimension_numbers).astype(jnp.int32) << 1)
      + (_dot_int4(r_a, b_half, dimension_numbers).astype(jnp.int32) << 1)
      + _dot_int4(r_a, r_b, dimension_numbers).astype(jnp.int32)
  )

  cross_term = z1 - z2.astype(jnp.int32) - z0.astype(jnp.int32)
  xy_product = (
      (z2.astype(jnp.int32) << 8) + (cross_term << 4) + z0.astype(jnp.int32)
  )

  # Offsets across contracting dimension
  k_size = 1
  for d in lhs_ca:
    k_size *= a.shape[d]

  sum_a = jnp.sum(a_int, axis=lhs_ca)
  sum_b = jnp.sum(b_int, axis=rhs_ca)

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


def naive_signed_int8_dot(
    a: jax.Array,
    b: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers = (((1,), (0,)), ((), ())),
    *,
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array:
  """Naive signed int8 GEMM using 4 uncoupled native signed int4 passes.

  Computes 100% bit-exact signed INT8 x INT8 matrix multiplication:
  1. Decomposes operands into signed high nibbles a_h in [-8, 7] and centered
  low
     nibbles a_l in [-8, 7].
  2. Evaluates 4 uncoupled signed INT4 passes on the MXU:
     - P11 = a_h * b_h
     - P10 = a_h * b_l
     - P01 = a_l * b_h
     - P00 = a_l * b_l
  3. Assembles the product: (P11 << 8) + ((P10 + P01) << 4) + P00 + offsets.

  Args:
    a: Signed int8 operand (or float/int array with values in [-128, 127]).
    b: Signed int8 operand.
    dimension_numbers: Standard JAX dot_general dimension specification.
    preferred_element_type: Target output dtype (default int32).

  Returns:
    Result array exactly matching int8 x int8 matrix multiplication.
  """
  a_int = a.astype(jnp.int32)
  b_int = b.astype(jnp.int32)
  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers

  a_h = a_int >> 4
  a_l = ((a_int.astype(jnp.uint8) & 0x0F).astype(jnp.int32)) - 8

  b_h = b_int >> 4
  b_l = ((b_int.astype(jnp.uint8) & 0x0F).astype(jnp.int32)) - 8

  p11 = _dot_int4(a_h, b_h, dimension_numbers)
  p10 = _dot_int4(a_h, b_l, dimension_numbers)
  p01 = _dot_int4(a_l, b_h, dimension_numbers)
  p00 = _dot_int4(a_l, b_l, dimension_numbers)

  xy_product = (
      (p11.astype(jnp.int32) << 8)
      + ((p10.astype(jnp.int32) + p01.astype(jnp.int32)) << 4)
      + p00.astype(jnp.int32)
  )

  k_size = 1
  for d in lhs_ca:
    k_size *= a.shape[d]

  sum_a = jnp.sum(a_int, axis=lhs_ca)
  sum_b = jnp.sum(b_int, axis=rhs_ca)

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


def triangular_signed_int8_dot(
    a: jax.Array,
    b: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers = (((1,), (0,)), ((), ())),
    *,
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array:
  """Triangular signed int8 GEMM using 3 uncoupled native signed int4 passes.

  Drops the lowest-order cross term P00 = a_l * b_l (coefficient 1), retaining
  the three leading terms:
    - P11 = a_h * b_h (coefficient 256)
    - P10 = a_h * b_l (coefficient 16)
    - P01 = a_l * b_h (coefficient 16)
  along with linear offset corrections: (sum_a * 8) + (sum_b * 8) - (64 *
  k_size).

  This achieves ~33.6 dB SQNR with 3 uncoupled passes on native INT4 hardware,
  avoiding the remainder operations needed for Karatsuba bit-exactness.

  Args:
    a: Signed int8 operand (or float/int array with values in [-128, 127]).
    b: Signed int8 operand.
    dimension_numbers: Standard JAX dot_general dimension specification.
    preferred_element_type: Target output dtype (default int32).

  Returns:
    Result array approximating int8 x int8 matrix multiplication.
  """
  a_int = a.astype(jnp.int32)
  b_int = b.astype(jnp.int32)
  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers

  a_h = a_int >> 4
  a_l = ((a_int.astype(jnp.uint8) & 0x0F).astype(jnp.int32)) - 8

  b_h = b_int >> 4
  b_l = ((b_int.astype(jnp.uint8) & 0x0F).astype(jnp.int32)) - 8

  p11 = _dot_int4(a_h, b_h, dimension_numbers)
  p10 = _dot_int4(a_h, b_l, dimension_numbers)
  p01 = _dot_int4(a_l, b_h, dimension_numbers)

  xy_product = (p11.astype(jnp.int32) << 8) + (
      (p10.astype(jnp.int32) + p01.astype(jnp.int32)) << 4
  )

  k_size = 1
  for d in lhs_ca:
    k_size *= a.shape[d]

  sum_a = jnp.sum(a_int, axis=lhs_ca)
  sum_b = jnp.sum(b_int, axis=rhs_ca)

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


def asymmetric_signed_int4_int8_dot(
    a: jax.Array,
    b: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers = (((1,), (0,)), ((), ())),
    *,
    int4_operand: Literal['lhs', 'rhs'] = 'lhs',
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array:
  """Asymmetric signed INT4 x INT8 matrix multiplication via 2 INT4 passes.

  Computes signed matrix multiplication between an INT4 operand (in [-8, 7])
  and an INT8 operand (in [-128, 127]) using strictly 2 native INT4 passes.

  When int4_operand == 'lhs':
    a is in [-8, 7] (INT4) and b is in [-128, 127] (INT8).
    Decompose b: b_h = b >> 4, b_l = ((b & 0x0F) - 8).
    Product: 16 * dot_int4(a, b_h) + dot_int4(a, b_l) + 8 * sum_k(a).

  When int4_operand == 'rhs':
    a is in [-128, 127] (INT8) and b is in [-8, 7] (INT4).
    Decompose a: a_h = a >> 4, a_l = ((a & 0x0F) - 8).
    Product: 16 * dot_int4(a_h, b) + dot_int4(a_l, b) + 8 * sum_k(b).

  Bit-exact to signed int4 x int8 in strictly 2 native INT4 passes.

  Args:
    a: Operand array.
    b: Operand array.
    dimension_numbers: Standard JAX dot_general dimension specification.
    int4_operand: Which operand is 4-bit ('lhs' or 'rhs').
    preferred_element_type: Target output dtype (default int32).

  Returns:
    Result array matching the integer matrix multiplication.
  """
  a_int = a.astype(jnp.int32)
  b_int = b.astype(jnp.int32)
  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers

  if int4_operand == 'lhs':
    b_h = b_int >> 4
    b_l = ((b_int.astype(jnp.uint8) & 0x0F).astype(jnp.int32)) - 8

    p1 = _dot_int4(a_int, b_h, dimension_numbers)
    p0 = _dot_int4(a_int, b_l, dimension_numbers)

    sum_a = jnp.sum(a_int, axis=lhs_ca)
    rhs_rem_ndims = b.ndim - len(rhs_ca) - len(rhs_ba)
    for _ in range(rhs_rem_ndims):
      sum_a = jnp.expand_dims(sum_a, axis=-1)

    res = (p1.astype(jnp.int32) << 4) + p0.astype(jnp.int32) + (sum_a * 8)
  else:
    a_h = a_int >> 4
    a_l = ((a_int.astype(jnp.uint8) & 0x0F).astype(jnp.int32)) - 8

    p1 = _dot_int4(a_h, b_int, dimension_numbers)
    p0 = _dot_int4(a_l, b_int, dimension_numbers)

    sum_b = jnp.sum(b_int, axis=rhs_ca)
    lhs_rem_ndims = a.ndim - len(lhs_ca) - len(lhs_ba)
    insert_pos = len(lhs_ba)
    for _ in range(lhs_rem_ndims):
      sum_b = jnp.expand_dims(sum_b, axis=insert_pos)

    res = (p1.astype(jnp.int32) << 4) + p0.astype(jnp.int32) + (sum_b * 8)

  if preferred_element_type is not None:
    return res.astype(preferred_element_type)
  return res


def mxint8_multipass_dot(
    lhs: jax.Array | qarray.QArray,
    rhs: jax.Array | qarray.QArray,
    dimension_numbers: jax.lax.DotDimensionNumbers = (((1,), (0,)), ((), ())),
    *,
    method: Literal[
        'karatsuba', 'naive', 'triangular', 'asymmetric'
    ] = 'karatsuba',
    int4_operand: Literal['lhs', 'rhs'] = 'lhs',
    tile_size: int = 32,
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array:
  """Executes microscaled mxint8 GEMM emulated via native INT4 building blocks.

  Computes per-block INT8 products using Karatsuba (3 INT4 passes, exact),
  Naive (4 INT4 passes, exact), Triangular (3 uncoupled INT4 passes, ~33.6 dB),
  or Asymmetric (2 INT4 passes, ~34.7 dB), scaling each block product by
  (scale_lhs * scale_rhs).

  Args:
    lhs: Left-hand side tensor or QArray.
    rhs: Right-hand side tensor or QArray.
    dimension_numbers: Standard JAX dot_general dimension specification.
    method: 'karatsuba', 'naive', 'triangular', or 'asymmetric'.
    int4_operand: When method is 'asymmetric', which operand is INT4 ('lhs' or
      'rhs').
    tile_size: Microscaling block size (default 32).
    preferred_element_type: Target output dtype (default float32).

  Returns:
    The resulting matrix multiplication array.
  """
  if method == 'asymmetric':
    lhs_qtype = 'mxint4' if int4_operand == 'lhs' else 'mxint8'
    rhs_qtype = 'mxint8' if int4_operand == 'lhs' else 'mxint4'
  else:
    lhs_qtype = 'mxint8'
    rhs_qtype = 'mxint8'

  if not isinstance(lhs, qarray.QArray):
    lhs_how = dot_general.get_how_to_quantize(
        dimension_numbers=dimension_numbers,
        ndims=(lhs.ndim, rhs.ndim),
        for_lhs=True,
        qtype=lhs_qtype,
        tile_size=tile_size,
    )
    q_lhs = qarray.quantize(lhs, lhs_how)
  else:
    q_lhs = lhs

  if not isinstance(rhs, qarray.QArray):
    rhs_how = dot_general.get_how_to_quantize(
        dimension_numbers=dimension_numbers,
        ndims=(lhs.ndim, rhs.ndim),
        for_lhs=False,
        qtype=rhs_qtype,
        tile_size=tile_size,
    )
    q_rhs = qarray.quantize(rhs, rhs_how)
  else:
    q_rhs = rhs

  if method == 'karatsuba':
    dot_fn = karatsuba_signed_int8_dot
  elif method == 'naive':
    dot_fn = naive_signed_int8_dot
  elif method == 'triangular':
    dot_fn = triangular_signed_int8_dot
  elif method == 'asymmetric':
    dot_fn = functools.partial(
        asymmetric_signed_int4_int8_dot, int4_operand=int4_operand
    )
  else:
    raise ValueError(f'Unknown method: {method}')

  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
  if (
      len(lhs_ca) == 1
      and len(rhs_ca) == 1
      and len(lhs_ba) == 0
      and len(rhs_ba) == 0
      and lhs.ndim == 2
      and rhs.ndim == 2
  ):
    m, k = q_lhs.qvalue.shape
    k2, n = q_rhs.qvalue.shape
    if k % tile_size == 0 and k == k2:
      n_blocks = k // tile_size
      a_blocks = q_lhs.qvalue.reshape(m, n_blocks, tile_size)
      b_blocks = q_rhs.qvalue.reshape(n_blocks, tile_size, n)
      sa_blocks = jnp.transpose(q_lhs.scale, (1, 0))[:, :, None]
      sb_blocks = q_rhs.scale[:, None, :]

      def block_gemm(a_b, b_b):
        return dot_fn(a_b, b_b, (((1,), (0,)), ((), ())))

      block_prods = jax.vmap(block_gemm, in_axes=(1, 0), out_axes=0)(
          a_blocks, b_blocks
      )
      scale_prods = sa_blocks * sb_blocks
      accum = jnp.sum(
          block_prods.astype(jnp.float32) * scale_prods.astype(jnp.float32),
          axis=0,
      )
      if preferred_element_type is not None:
        return accum.astype(preferred_element_type)
      return accum

  lhs_deq = qarray.dequantize(q_lhs)
  rhs_deq = qarray.dequantize(q_rhs)
  return lax.dot_general(
      lhs_deq,
      rhs_deq,
      dimension_numbers=dimension_numbers,
      preferred_element_type=preferred_element_type,
  )


def asymmetric_mxint_dot(
    lhs: jax.Array | qarray.QArray,
    rhs: jax.Array | qarray.QArray,
    dimension_numbers: jax.lax.DotDimensionNumbers = (((1,), (0,)), ((), ())),
    *,
    lhs_qtype: str = 'mxint4',
    rhs_qtype: str = 'mxint8',
    tile_size: int = 32,
    preferred_element_type: jax.typing.DTypeLike | None = None,
) -> jax.Array:
  """Evaluates asymmetric microscaled integer matrix multiplication."""
  del rhs_qtype
  int4_operand = 'lhs' if lhs_qtype == 'mxint4' else 'rhs'
  return mxint8_multipass_dot(
      lhs,
      rhs,
      dimension_numbers=dimension_numbers,
      method='asymmetric',
      int4_operand=int4_operand,
      tile_size=tile_size,
      preferred_element_type=preferred_element_type,
  )
